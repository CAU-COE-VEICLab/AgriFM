# --------------------------------------------------------
# Agri420K: A Large-Scale Benchmark Dataset for Agricultural Image Recognition
# Licensed under The MIT License [see LICENSE for details]
# Written by Guorun Li and Yucong Wang
# --------------------------------------------------------

from .swin_transformer import SwinTransformer
from .convnext import ConvNeXt
from .swin_transformer_v2 import SwinTransformerV2
from .ResNet import resnetX
from .Transformer import ViT
from .VGG19 import VGG19
from .VGG16 import VGG16
from .MobileNetv2 import MobileNetV2
from .MobileNetv3 import MobileNetV3_Large, MobileNetV3_Small
from .Xception import Xception
from .PVT_V2 import PyramidVisionTransformerV2
from .efficientnet_v2 import get_efficientnet_v2
from .Densenet import densenet201

def build_model(config, is_pretrain=False):
    model_type = config.MODEL.TYPE

    # accelerate layernorm
    if config.FUSED_LAYERNORM:
        try:
            import apex as amp
            layernorm = amp.normalization.FusedLayerNorm
        except:
            layernorm = None
            print("To use FusedLayerNorm, please install apex.")
    else:
        import torch.nn as nn
        layernorm = nn.LayerNorm

    if model_type == 'swinv2':
        model = SwinTransformerV2(img_size=config.DATA.IMG_SIZE,
                                  patch_size=config.MODEL.SWINV2.PATCH_SIZE,
                                  in_chans=config.MODEL.SWINV2.IN_CHANS,
                                  num_classes=config.MODEL.NUM_CLASSES,
                                  embed_dim=config.MODEL.SWINV2.EMBED_DIM,
                                  depths=config.MODEL.SWINV2.DEPTHS,
                                  num_heads=config.MODEL.SWINV2.NUM_HEADS,
                                  window_size=config.MODEL.SWINV2.WINDOW_SIZE,
                                  mlp_ratio=config.MODEL.SWINV2.MLP_RATIO,
                                  qkv_bias=config.MODEL.SWINV2.QKV_BIAS,
                                  drop_rate=config.MODEL.DROP_RATE,
                                  drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                  ape=config.MODEL.SWINV2.APE,
                                  patch_norm=config.MODEL.SWINV2.PATCH_NORM,
                                  use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                  pretrained_window_sizes=config.MODEL.SWINV2.PRETRAINED_WINDOW_SIZES)
    elif model_type == 'swin':
        model = SwinTransformer(img_size=config.DATA.IMG_SIZE,
                                patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                in_chans=config.MODEL.SWIN.IN_CHANS,
                                num_classes=config.MODEL.NUM_CLASSES,
                                embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                depths=config.MODEL.SWIN.DEPTHS,
                                num_heads=config.MODEL.SWIN.NUM_HEADS,
                                window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                qk_scale=config.MODEL.SWIN.QK_SCALE,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                ape=config.MODEL.SWIN.APE,
                                norm_layer=layernorm,
                                patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                fused_window_process=config.FUSED_WINDOW_PROCESS)


    elif model_type == 'convnext':
        model = ConvNeXt(
            in_chans=config.MODEL.CONVNEXT.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            depths=config.MODEL.CONVNEXT.DEPTHS,
            dims=config.MODEL.CONVNEXT.DIMS,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            layer_scale_init_value=config.MODEL.CONVNEXT.LAYER_SCALE_INIT_VALUE,
            )

    elif model_type == 'resnet':
        model = resnetX(
            layers=config.MODEL.RESNET.LAYERS,
            num_classes=config.MODEL.NUM_CLASSES,
        )

    elif model_type == 'vit':
        model = ViT(
            image_size=config.DATA.IMG_SIZE,
            num_classes=config.MODEL.NUM_CLASSES,
            patches=config.MODEL.VIT.PATCHES,
            dim=config.MODEL.VIT.EMBED_DIM,
            ff_dim=config.MODEL.VIT.FEEDFORWARD_DIM,
            num_heads=config.MODEL.VIT.NUM_HEADS,
            num_layers=config.MODEL.VIT.NUM_LAYERS,
            attention_dropout_rate=config.MODEL.VIT.ATTENTION_DROPOUT_RATE,
            dropout_rate=config.MODEL.VIT.DROPOUT_RATE,
            representation_size=config.MODEL.VIT.REPRESENTATION_SIZE,
            load_repr_layer=config.MODEL.VIT.LOAD_REPR_LAYER,
            classifier=config.MODEL.VIT.CLASSIFIER,
            positional_embedding=config.MODEL.VIT.POSITIONAL_EMBEDDING,
            in_channels=config.MODEL.VIT.IN_CHANS,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
        )

    elif model_type == 'vgg19':
        model = VGG19(
            num_labels=config.MODEL.NUM_CLASSES,
        )

    elif model_type == 'vgg16':
        model = VGG16(
            num_labels=config.MODEL.NUM_CLASSES,
        )

    elif model_type == 'mobilenetv3_l':
        model = MobileNetV3_Large(
            num_classes=config.MODEL.NUM_CLASSES,
        )
    elif model_type == 'mobilenetv3_s':
        model = MobileNetV3_Small(
            num_classes=config.MODEL.NUM_CLASSES,
        )
    elif model_type == 'mobilenetv2':
        model = MobileNetV2(
            num_classes=config.MODEL.NUM_CLASSES,
        )
    elif model_type == 'xception':
        model = Xception(
            num_classes=config.MODEL.NUM_CLASSES,
        )

    elif model_type == 'pvt':
        model = PyramidVisionTransformerV2(
                img_size=config.DATA.IMG_SIZE, 
                patch_size=config.MODEL.PVT.PATCH_SIZE, 
                in_chans=config.MODEL.PVT.IN_CHANS, 
                num_classes=config.MODEL.NUM_CLASSES, 
                embed_dims=config.MODEL.PVT.EMBED_DIMS,
                num_heads=config.MODEL.PVT.NUM_HEADS, 
                mlp_ratios=config.MODEL.PVT.MLP_RATIOS, 
                qkv_bias=config.MODEL.PVT.QKV_BIAS, 
                qk_scale=config.MODEL.PVT.QK_SCALE, 
                drop_rate=config.MODEL.PVT.DROP_RATE,
                attn_drop_rate=config.MODEL.PVT.ATTN_DROP_RATE, 
                drop_path_rate=config.MODEL.PVT.DROP_PATH_RATE,
                depths=config.MODEL.PVT.DEPTHS, 
                sr_ratios=config.MODEL.PVT.SR_RATIOS, 
                num_stages=config.MODEL.PVT.NUM_STAGES, 
                linear=config.MODEL.PVT.LINEAR,
        )
        
    elif model_type == 'efficientnet_v2':
        model = get_efficientnet_v2(
                model_name=config.MODEL.EFFICIENTNET_V2.NAME,
                nclass=config.MODEL.NUM_CLASSES,
                dropout=config.MODEL.EFFICIENTNET_V2.DROPOUT,
                stochastic_depth=config.MODEL.EFFICIENTNET_V2.STOCHASTIC_DEPTH,
        )

    elif model_type == 'densenet201':
        model = densenet201(
            num_classes=config.MODEL.NUM_CLASSES,
        )
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
