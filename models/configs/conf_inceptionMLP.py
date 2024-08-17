from models.vision_model import VisionModel, _cfg
from timm.models.registry import register_model
from models.inceptionMLP import MSFF

default_cfgs = {
    'InceptionMLP_S': _cfg(crop_pct=0.9),
    'InceptionMLP_M': _cfg(crop_pct=0.9),
    'InceptionMLP_L': _cfg(crop_pct=0.875),
}

@register_model # overlap = False  
def inceptionmlp_s(pretrained=False, **kwargs):
    img_size =224
    patch_size = 4
    # L = 12 #
    init_values = 1e-5
    # patch_size = 7
    layers = [3, 3, 9, 3]
    # if_mlp = [True, True, True, True]
    transitions = [True, True, True, True]
    resolutions = [56, 28, 14, 7]
    # resolutions = [32, 16, 8, 4]
    num_heads = [8, 16, 16, 16]
    channel_ratios = [4, 4, 4, 4] ## channel mixing ratio
    # mlp_ratios = [0.25, 1, 2, 3] ## patch mixing ratio
    embed_dims = [96, 192, 324, 396] # stage 
    # rs = [8, 12, 16, 16]
    reduce_ratios=[1, 1, 1, 1]  ##多brach结构中，为了节省参数量，Channel降维比例
    # inverts=[1,1,1,1]
    # att_switches=[False, False, True, True]
    shift_list = [1,2,3]
    split_channels = [[2,1,1],[2,1,1],[1,1,1],[1,1,4]]
    M=3
    proj_ratio=[3,3,2,1]

    model = VisionModel(layers=layers,embed_dims=embed_dims, patch_size= patch_size, reduce_ratios = reduce_ratios, transitions=transitions,
                        resolutions=resolutions, num_heads=num_heads, init_values= init_values,  channel_ratios = channel_ratios,M=M,
                        mlp_fn=MSFF, shift_list=shift_list, split_channels=split_channels,proj_ratio=proj_ratio,**kwargs)
    model.default_cfg = default_cfgs['InceptionMLP_S']
    return model




@register_model # overlap = False
def inceptionmlp_m(pretrained=False, **kwargs):
    img_size =224
    patch_size = 4
    # L = 12 #
    init_values = 1e-5
    # patch_size = 7
    layers = [3, 4, 12, 6]
    # if_mlp = [True, True, True, True]
    transitions = [True, True, True, True]
    resolutions = [56, 28, 14, 7]
    num_heads = [8, 16, 16, 16]
    channel_ratios = [4, 4, 4, 4] ## channel mixing ratio
    # mlp_ratios = [0.5, 1, 2, 3] ## patch mixing ratio
    embed_dims = [96, 192, 384, 522] # stage 
    # rs = [8, 16, 16, 16]
    reduce_ratios=[1, 1, 1, 1]  
    # inverts=[1,1,1,1]
    # att_switches=[False, False, True, True]
    shift_list = [1,2,3]
    split_channels = [[2,1,1],[2,1,1],[1,2,1],[1,1,4]]
    M=3
    proj_ratio=[3,3,2,2]

    model = VisionModel(layers=layers,embed_dims=embed_dims, patch_size= patch_size, reduce_ratios = reduce_ratios, transitions=transitions,
                        resolutions=resolutions, num_heads=num_heads, init_values= init_values,  channel_ratios = channel_ratios,M=M,
                        mlp_fn=MSFF, shift_list=shift_list, split_channels=split_channels,proj_ratio=proj_ratio,**kwargs)
    model.default_cfg = default_cfgs['InceptionMLP_M']
    return model

@register_model # overlap = False
def inceptionmlp_l(pretrained=False, **kwargs):
    img_size =224
    patch_size = 4
    L = 16 #
    init_values = 1e-5
    # patch_size = 7
    layers = [4, 6, 14, 8]
    # if_mlp = [True, True, True, True]
    transitions = [True, True, True, True]
    resolutions = [56, 28, 14, 7]
    # resolutions = [32, 16, 8, 4]
    num_heads = [8, 16, 16, 16]
    channel_ratios = [4, 4, 4, 4] ## channel mixing ratio
    # mlp_ratios = [0.5, 1, 2, 3] ## patch mixing ratio
    embed_dims = [96, 192, 450, 648] # stage 
    # rs = [8, 12, 16, 16]
    reduce_ratios=[1, 1, 1, 1]  ##多brach结构中，为了节省参数量，Channel降维比例
    # inverts=[1,1,1,1]
    # att_switches=[False, False, True, True]
    shift_list = [1,2,3]
    split_channels = [[2,1,1],[2,1,1],[1,2,2],[1,1,4]]
    M=3
    proj_ratio=[2,4,1,2]

    model = VisionModel(layers=layers,embed_dims=embed_dims, patch_size= patch_size, reduce_ratios = reduce_ratios, transitions=transitions,
                        resolutions=resolutions, num_heads=num_heads, init_values= init_values,  channel_ratios = channel_ratios,M=M,
                        mlp_fn=MSFF, shift_list=shift_list, split_channels=split_channels,proj_ratio=proj_ratio,**kwargs)
    model.default_cfg = default_cfgs['InceptionMLP_L']
    return model