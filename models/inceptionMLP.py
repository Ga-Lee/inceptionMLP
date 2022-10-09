import torch
import torch.nn as nn
import torch.nn.functional as F
# from models.utils import Mlp
import math
from torch.nn import init

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class DwConv(nn.Module): #输入为 B C H W
    def __init__(self, invert = 1,  reduce_ratio = 1, dim = 96, kernel_size = 3, stride = 1, padding = 1,bias = False):
        super().__init__()

        self.invert = invert
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.reduce_ratio = reduce_ratio

        if self.invert != 1 :
            self.linear1 = nn.Conv2d(dim, int(self.invert * dim * self.reduce_ratio), kernel_size=1,stride=1,bias = self.bias)
            self.relu1 = nn.ReLU(inplace=True)
            self.bn1 = nn.BatchNorm2d(int(self.invert * dim * self.reduce_ratio))  #不确定bn和relu的顺序 默认在后面

            self.conv = nn.Conv2d(in_channels = int(self.invert * dim * self.reduce_ratio) , out_channels = int(self.invert * dim * self.reduce_ratio) , kernel_size = self.kernel_size, stride = self.stride, padding = self.padding, groups= int(self.invert * dim * self.reduce_ratio), bias= self.bias )
            self.relu2 = nn.ReLU(inplace=True)
            self.bn2 = nn.BatchNorm2d(int(self.invert * dim * self.reduce_ratio))

            self.linear2 = nn.Conv2d(int(self.invert * dim * self.reduce_ratio), int(dim * self.reduce_ratio), kernel_size=1,stride=1,bias = self.bias)
            self.relu3 = nn.ReLU(inplace=True)
            self.bn3 = nn.BatchNorm2d(dim)

        else:
            self.linear1 = nn.Conv2d(dim,int(self.reduce_ratio * dim), kernel_size=1, stride=1,bias = self.bias)  ## 这个linear 到底放在卷积前还是卷积后 有影响吗？
            self.relu1 = nn.ReLU(inplace=True)
            self.bn1 = nn.BatchNorm2d(int(self.reduce_ratio * dim))

            self.conv = nn.Conv2d(in_channels = int(self.reduce_ratio * dim) , out_channels = int(self.reduce_ratio * dim) , kernel_size = self.kernel_size, stride = self.stride, padding = self.padding, groups = int(self.reduce_ratio * dim), bias= self.bias)
            self.relu2 = nn.ReLU(inplace=True)
            self.bn2 = nn.BatchNorm2d(int(self.reduce_ratio * dim))
            

    def forward(self, x):
        x = self.bn1(self.relu1(self.linear1(x))) ## B C H W -> B C*RATION H W  
        x = self.bn2(self.relu2(self.conv(x)))
        if self.invert != 1:
            x = self.bn3(self.relu3(self.linear2(x)))  # B C*RATION H W  

        return x

class Maxpool(nn.Module): # INPUT SIZE : B C H W
    def __init__(self, dim = 96, reduce_ratio = 1, kernel_size = 3, stride = 1, padding = 1, bias = False):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.reduce_ratio = reduce_ratio

        self.maxpool = nn.MaxPool2d(self.kernel_size, self.stride, self.padding)  #接受输入为B, C, H, W
        self.conv = nn.Conv2d(dim, int(dim*self.reduce_ratio), kernel_size=1, stride=1, bias= self.bias)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(int(dim*self.reduce_ratio))
            
    def forward(self, x):
        x = self.maxpool(x) # B C H W
        x = self.conv(x) # B C*RATIO H W 
        x = self.relu(x)
        x = self.bn(x) # B C*RATIO H W (C CHANNEL)

        return x



class CoordAtt(nn.Module): # INPUT TENSOR : B C H W
    def __init__(self, inp, oup, groups=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // groups)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.relu = h_swish()

    def forward(self, x):
        identity = x
        n,c,h,w = x.size() ### size 的要求
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y) 
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()
        x_h = x_h.expand(-1, -1, h, w)
        x_w = x_w.expand(-1, -1, h, w)

        y = identity * x_w * x_h # B C H W

        return y


class MLPAtten(nn.Module): # B C H W 
    def __init__(self, dim, resolution, drop = 0.0, mlp_ratio = 2, reduce_ratio = 1, if_mlp = True, att_switch = False, groups=32, bias = False):
        super().__init__()

        self.reduce_ratio = reduce_ratio 
        self.bias = bias
        self.num_patch = resolution**2
        # self.oup = dim * self.reduce_ratio
        self.hidden_patches = int(mlp_ratio * self.num_patch)
        self.group = groups
        self.dropout = drop
        self.att_switch = att_switch

        if if_mlp:
            self.mlp = nn.Sequential(                                  ##spatial dimension的混合  在第1、2stage patch number平方倍关系，mlp若为True,mlp_ratio应该小于1            
                nn.Conv1d(self.num_patch, self.hidden_patches, kernel_size=1, stride=1),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Conv1d(self.hidden_patches, self.num_patch, kernel_size=1, stride=1),
                nn.Dropout(self.dropout)
            )
        else:
            self.mlp = nn.Sequential(         ##spatial dimension的混合 不进行spatial维度的expansion 1:1
                nn.Conv1d(self.num_patch, self.num_patch, kernel_size=1, stride=1),
                nn.GELU(),
                nn.Dropout(self.dropout))

        if self.reduce_ratio < 1:   # channel dimension降维
            self.reduce = nn.Conv2d(dim, int(dim* self.reduce_ratio), kernel_size=1, stride=1, padding=0)
            self.bn = nn.BatchNorm2d(int(dim* self.reduce_ratio))
            self.relu = nn.ReLU(inplace=True) 

        if self.att_switch:
            self.attention = CoordAtt(int(dim* self.reduce_ratio), int(dim* self.reduce_ratio), self.group)

    def forward(self, x): # B C H W
        # identity = x 
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1)  # b h w c
        x = x.view(b, -1, c) # B num_patch C
        x = self.mlp(x) # b num_patch c
        ## 转回 B C H W 
        x = x.permute(0, 2 , 1)
        x = x.view( b, c, h, w )

        if self.reduce_ratio < 1: 
            x = self.relu(self.bn(self.reduce(x)))
        
        if self.att_switch:
            x = self.attention(x)   #attention的输入是 mlp（+reduce）的输出
        return x


class SKFusion(nn.Module): #输入的各分支尺寸和通道数必须一样  输入 B C H W
    def __init__(self, features, resolution, reduce_ratio = 1, M=3, r=16, kernel_size=3, stride=1 , if_mlp =True, invert = 1, attn_drop=0., mlp_ratio = 2, att_switch = False, proj_drop=0, L=16):
        """ Constructor
        Args:
            features: input channel dimensionality.
            M: the number of branchs.
            G: num of convolution groups.
            r: the ratio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SKFusion, self).__init__()
        self.M = M #分支数
        self.features = features
        self.kernel_size = kernel_size
        self.reduce_ratio = reduce_ratio
        d = max(int(self.features * self.reduce_ratio/r), L) #降维下界
        self.stride = stride
        self.mlp_ratio = mlp_ratio
        self.invert = invert
        self.att_switch = att_switch
        self.attn_drop = attn_drop
        self.proj_drop =proj_drop
        self.resolution = resolution
        self.if_mlp = if_mlp

        self.convs = nn.ModuleList([])
        self.maxpool = Maxpool(dim = self.features, reduce_ratio = self.reduce_ratio, kernel_size = self.kernel_size, stride = self.stride, padding = 1, bias = False) 
        self.convs.append(self.maxpool)
        self.dwconv = DwConv(self.invert, reduce_ratio = self.reduce_ratio, dim = self.features, kernel_size = self.kernel_size, stride = self.stride, padding = 1,bias = False)
        self.convs.append(self.dwconv)
        self.mlpatten = MLPAtten(self.features, self.resolution, drop = self.attn_drop, mlp_ratio = self.mlp_ratio, if_mlp = self.if_mlp, reduce_ratio = self.reduce_ratio, att_switch = self.att_switch, groups=32, bias = False)
        self.convs.append(self.mlpatten)
        # for i in range(M):
        #     self.convs.append(nn.Sequential(
        #         nn.Conv2d(features, features, kernel_size=3, stride=stride, padding=1+i, dilation=1+i, groups=G, bias=False),
        #         nn.BatchNorm2d(features),
        #         nn.ReLU(inplace=True)
        #     ))
        self.gap = nn.AdaptiveAvgPool2d((1,1)) 
        self.fc = nn.Sequential(nn.Conv2d(int(self.features * self.reduce_ratio), d, kernel_size=1, stride=1, bias=False),
                                nn.BatchNorm2d(d),
                                nn.ReLU(inplace=True)) #降维
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                 nn.Conv2d(d, int(self.features * self.reduce_ratio), kernel_size=1, stride=1)  #升维
            )
        self.softmax = nn.Softmax(dim=1)
        if self.reduce_ratio < 1:
            self.proj = nn.Linear(int(self.features * self.reduce_ratio), self.features)
            self.proj_drop = nn.Dropout(self.proj_drop)
        
    def forward(self, x):
        
        batch_size = x.shape[0] ## BLOCK 的输入是  B C H W
        
        feats = [conv(x) for conv in self.convs]   #各分支结果   
        feats = torch.cat(feats, dim=1)  
        feats = feats.view(batch_size, self.M, int(self.features * self.reduce_ratio), feats.shape[2], feats.shape[3])
        
        feats_U = torch.sum(feats, dim=1) #元素相加
        feats_S = self.gap(feats_U)# 全局池化
        feats_Z = self.fc(feats_S)# 降维

        attention_vectors = [fc(feats_Z) for fc in self.fcs] #升维
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.M, int(self.features * self.reduce_ratio), 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        
        feats_V = torch.sum(feats*attention_vectors, dim=1) # B C H W 
        feats_V = feats_V.permute(0,2,3,1) # B  H  W  C*RATIO

        if self.reduce_ratio < 1:
            feats_V = self.proj(feats_V)
            feats_V = self.proj_drop(feats_V)
   
        return feats_V #输出是 B H W C
