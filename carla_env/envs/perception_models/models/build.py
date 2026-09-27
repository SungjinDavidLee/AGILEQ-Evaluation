from .intern_image import InternImage


def build_model(config,in_c=3):
    model_type = config.MODEL.TYPE
    if model_type == 'intern_image':
        model = InternImage(
            in_chans=in_c,
            core_op=config.MODEL.INTERN_IMAGE.CORE_OP,
            num_classes=config.MODEL.NUM_CLASSES,
            channels=config.MODEL.INTERN_IMAGE.CHANNELS,
            depths=config.MODEL.INTERN_IMAGE.DEPTHS,
            groups=config.MODEL.INTERN_IMAGE.GROUPS,
            layer_scale=config.MODEL.INTERN_IMAGE.LAYER_SCALE,
            offset_scale=config.MODEL.INTERN_IMAGE.OFFSET_SCALE,
            post_norm=config.MODEL.INTERN_IMAGE.POST_NORM,
            mlp_ratio=config.MODEL.INTERN_IMAGE.MLP_RATIO,
            with_cp=config.TRAIN.USE_CHECKPOINT,
            res_post_norm=config.MODEL.INTERN_IMAGE.RES_POST_NORM,
            dw_kernel_size=config.MODEL.INTERN_IMAGE.DW_KERNEL_SIZE,
            use_clip_projector=config.MODEL.INTERN_IMAGE.USE_CLIP_PROJECTOR,
            level2_post_norm=config.MODEL.INTERN_IMAGE.LEVEL2_POST_NORM,
            level2_post_norm_block_ids=config.MODEL.INTERN_IMAGE.LEVEL2_POST_NORM_BLOCK_IDS,
            center_feature_scale=config.MODEL.INTERN_IMAGE.CENTER_FEATURE_SCALE,
            remove_center=config.MODEL.INTERN_IMAGE.REMOVE_CENTER,
        )
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
