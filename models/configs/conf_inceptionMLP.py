from models.vision_model import VisionModel, _cfg
from timm.models.registry import register_model
from models.inceptionMLP import SKFusion

default_cfgs = {
    'InceptionMLP_S': _cfg(crop_pct=0.9),
    'InceptionMLP_M': _cfg(crop_pct=0.9),
    'InceptionMLP_L': _cfg(crop_pct=0.875),
}



@register_model # overlap = False  参数主要集中在第一层 和resolution关系更大
def inceptionmlp_s(pretrained=False, **kwargs):
    patch_size = 4
    L = 12 #
    init_values = 1e-5
    # patch_size = 7
    layers = [3, 3, 9, 3]
    if_mlp = [True, True, True, True]
    # layers = [1, 1, 1, 1]
    transitions = [True, True, True, True]
    resolutions = [56, 28, 14, 7]
    # resolutions = [32, 16, 8, 4]
    num_heads = [8, 16, 16, 16]
    channel_ratios = [3, 3, 2, 1] ## channel mixing ratio
    mlp_ratios = [0.25, 1, 2, 3] ## patch mixing ratio
    embed_dims = [96, 192, 320, 384] # stage 
    rs = [8, 12, 16, 16]
    reduce_ratios=[1, 1, 1, 1]  ##多brach结构中，为了节省参数量，Channel降维比例
    inverts=[1,1,1,1]
    att_switches=[False, False, True, True]

    model = VisionModel(layers, embed_dims=embed_dims, patch_size= patch_size, reduce_ratios = reduce_ratios, rs = rs, inverts = inverts, att_switches = att_switches, transitions=transitions,
                        resolutions=resolutions, num_heads=num_heads, init_values= init_values, mlp_ratios=mlp_ratios, if_mlp = if_mlp, channel_ratios = channel_ratios, L = L,
                        mlp_fn=SKFusion, **kwargs)
    model.default_cfg = default_cfgs['InceptionMLP_S']
    return model


@register_model # overlap = False
def inceptionmlp_m(pretrained=False, **kwargs):
    patch_size = 4
    L = 12 #
    init_values = 1e-5
    # patch_size = 7
    layers = [3, 4, 12, 6]
    if_mlp = [True, True, True, True]
    transitions = [True, True, True, True]
    resolutions = [56, 28, 14, 7]
    num_heads = [8, 16, 16, 16]
    channel_ratios = [3, 3, 2, 1] ## channel mixing ratio
    mlp_ratios = [0.5, 1, 2, 3] ## patch mixing ratio
    embed_dims = [128, 224, 384, 512] # stage 
    rs = [8, 16, 16, 16]
    reduce_ratios=[1, 1, 1, 1]  
    inverts=[1,1,1,1]
    att_switches=[False, False, True, True]

    model = VisionModel(layers, embed_dims=embed_dims, patch_size= patch_size, reduce_ratios = reduce_ratios, rs = rs, inverts = inverts, att_switches = att_switches, transitions=transitions,
                        resolutions=resolutions, num_heads=num_heads, init_values= init_values, mlp_ratios=mlp_ratios, if_mlp = if_mlp, channel_ratios = channel_ratios, L = L,
                        mlp_fn=SKFusion, **kwargs)
    model.default_cfg = default_cfgs['InceptionMLP_M']
    return model

@register_model # overlap = False
def inceptionmlp_l(pretrained=False, **kwargs):
    patch_size = 4
    L = 16 #
    init_values = 1e-5
    # patch_size = 7
    layers = [4, 6, 14, 8]
    if_mlp = [True, True, True, True]
    transitions = [True, True, True, True]
    resolutions = [56, 28, 14, 7]
    # resolutions = [32, 16, 8, 4]
    num_heads = [8, 16, 16, 16]
    channel_ratios = [3, 3, 2, 1] ## channel mixing ratio
    mlp_ratios = [0.5, 1, 2, 3] ## patch mixing ratio
    embed_dims = [196, 256, 448, 600] # stage 
    rs = [8, 12, 16, 16]
    reduce_ratios=[1, 1, 1, 1]  ##多brach结构中，为了节省参数量，Channel降维比例
    inverts=[1,1,1,1]
    att_switches=[False, False, True, True]

    model = VisionModel(layers, embed_dims=embed_dims, patch_size= patch_size, reduce_ratios = reduce_ratios, rs = rs, inverts = inverts, att_switches = att_switches, transitions=transitions,
                        resolutions=resolutions, num_heads=num_heads, init_values = init_values, mlp_ratios=mlp_ratios, if_mlp = if_mlp, channel_ratios = channel_ratios, L = L,
                        mlp_fn=SKFusion, **kwargs)
    model.default_cfg = default_cfgs['InceptionMLP_L']
    return model