import torch.nn as nn
import torch
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_

from models.utils import PatchEmbed, Downsample, Mlp
from models.inceptionMLP import MSFF


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .96, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 'classifier': 'head',
        **kwargs
    }

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma



class VisionBlock(nn.Module): # 注意各个部分的维度 

    def __init__(self, dim, resolution, reduce_ratio,channel_ratio = 4.,M=3, init_values=None, drop=0., drop_path=0.,  act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip_lam=1.0, mlp_fn=MSFF, cat_proj = False, shift_list=[1,2,3],split_channel=[1,1,2],proj_ratio = 3):
        super().__init__()
        self.norm1 = norm_layer(dim) # 对最后一维进行
        self.attn = mlp_fn(resolution=resolution, features= dim, reduce_ratio = reduce_ratio, M=M, shift_list= shift_list, split_channel = split_channel,
                            cat_proj =cat_proj , proj_drop= drop, proj_ratio = proj_ratio)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * channel_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.skip_lam = skip_lam

    def forward(self, x): # x : b h w c
        # print('model device: ', self.ls1.gamma.device)
        # print(f'x device: {x.device}')
        # exit()
        x = self.norm1(x) # b h w c
        x = x.permute(0,3,1,2) # b c h w
        x = self.attn(x) # b c h w
        x = x.permute(0,2,3,1) # b h w c
        # x = x + self.drop_path(x) / self.skip_lam 
        x = x + self.drop_path1(self.ls1(x)) # b h w c

        # x = x + self.drop_path(self.mlp(self.norm2(x))) / self.skip_lam # B H W C
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x)))) ## b h w c
        # print(x.shape)
        return x  # b h w c 


def basic_blocks(dim, index, layers, resolution, reduce_ratio, M, channel_ratio = 3., \
                 drop=0., drop_path_rate=0., init_values = None, skip_lam=1.0, mlp_fn=MSFF, cat_proj = False, shift_list=[1,2,3], split_channel=None, norm_layer=nn.LayerNorm,proj_ratio=3,**kwargs): ##MSFF
    blocks = []

    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(VisionBlock(dim, resolution, reduce_ratio = reduce_ratio, M = M, channel_ratio= channel_ratio, \
                                  drop=drop, init_values = init_values, drop_path=block_dpr,skip_lam=skip_lam, mlp_fn=mlp_fn, cat_proj =cat_proj,shift_list=shift_list,split_channel=split_channel,norm_layer=norm_layer,proj_ratio=proj_ratio ))

    blocks = nn.Sequential(*blocks)

    return blocks


class VisionModel(nn.Module):

    def __init__(self, layers, img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dims=None, reduce_ratios=None,
                transitions=None, resolutions=None, num_heads=None, init_values = None , channel_ratios=None, skip_lam=1.0, M=3,
                drop_rate=0., drop_path_rate=0.,cat_proj = False,
                norm_layer=nn.LayerNorm, mlp_fn=MSFF, shift_list=[1,2,3], split_channels=None, overlap=False,proj_ratio=[3,2,1,1], **kwargs): ##

        super().__init__()
        self.num_classes = num_classes

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                                      embed_dim=embed_dims[0], overlap=overlap)

        network = []
        for i in range(len(layers)):
            stage = basic_blocks(embed_dims[i], i, layers, resolutions[i], reduce_ratios[i],M, 
                                init_values = init_values, channel_ratio=channel_ratios[i], drop = drop_rate,
                                 drop_path_rate=drop_path_rate, norm_layer=norm_layer, skip_lam=skip_lam, mlp_fn=mlp_fn, cat_proj =cat_proj,shift_list=shift_list,split_channel=split_channels[i],proj_ratio=proj_ratio[i])
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if transitions[i] or embed_dims[i] != embed_dims[i + 1]: ## merging or downsample
                patch_size = 2 if transitions[i] else 1
                network.append(Downsample(embed_dims[i], embed_dims[i + 1], patch_size, overlap=overlap))

        self.network = nn.ModuleList(network)

        self.norm = norm_layer(embed_dims[-1])

        # Classifier head
        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m): ## 
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if (isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d)) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_embeddings(self, x): 
        x = self.patch_embed(x) ## 输入 B C H W  输出 B H W C
        ##  B,C,H,W-> B,H,W,C
        # x = x.permute(0, 2, 3, 1)
        return x

    def forward_tokens(self, x): ## 输入 B H W C
        for idx, block in enumerate(self.network):
            x = block(x)  ## B H W C
        B, H, W, C = x.shape 
        # x = x.permute(0,2,3,1) ## B H W C
        x = x.reshape(B, -1, C)
        return x

    def forward(self, x):# B C H W
        x = self.forward_embeddings(x)   # B H W C
        x = self.forward_tokens(x) # B N C
        x = self.norm(x)
        return self.head(x.mean(1))




class VisionModelERF(nn.Module):

    def __init__(self, layers, img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dims=None, reduce_ratios=None,
                transitions=None, resolutions=None, num_heads=None, init_values = None , channel_ratios=None, skip_lam=1.0, M=3,
                drop_rate=0., drop_path_rate=0.,cat_proj = False,
                norm_layer=nn.LayerNorm, mlp_fn=MSFF, shift_list=[1,2,3], split_channels=None, overlap=False,proj_ratio=[3,2,1,1], **kwargs): ##

        super().__init__()
        self.num_classes = num_classes

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                                      embed_dim=embed_dims[0], overlap=overlap)

        network = []
        for i in range(len(layers)):
            stage = basic_blocks(embed_dims[i], i, layers, resolutions[i], reduce_ratios[i],M, 
                                init_values = init_values, channel_ratio=channel_ratios[i], drop = drop_rate,
                                 drop_path_rate=drop_path_rate, norm_layer=norm_layer, skip_lam=skip_lam, mlp_fn=mlp_fn, cat_proj =cat_proj,shift_list=shift_list,split_channel=split_channels[i],proj_ratio=proj_ratio[i])
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if transitions[i] or embed_dims[i] != embed_dims[i + 1]: ## merging or downsample
                patch_size = 2 if transitions[i] else 1
                network.append(Downsample(embed_dims[i], embed_dims[i + 1], patch_size, overlap=overlap))

        self.network = nn.ModuleList(network)

        self.norm = norm_layer(embed_dims[-1])

        # Classifier head
        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m): ## 
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if (isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d)) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_embeddings(self, x): 
        x = self.patch_embed(x) ## 输入 B C H W  输出 B H W C
        ##  B,C,H,W-> B,H,W,C
        # x = x.permute(0, 2, 3, 1)
        return x

    def forward_tokens(self, x): ## 输入 B H W C
        # print(len(self.network))
        for idx, block in enumerate(self.network):
            x = block(x)  ## B H W C
            # if idx > 1: # stage2
            #     break
            if idx > 5: # stage3
                break

        B, H, W, C = x.shape 
        # print(x.shape)
        x = x.permute(0,3, 1,2)

        return x

    def forward(self, x):# B C H W
        x = self.forward_embeddings(x)   # B H W C
        x = self.forward_tokens(x) # B N C
        return x