
def add_panoptic_pyconv_config(cfg):
    """
    Add config for panoptic_pyconv
    cfg: the YAML configuration for the default config.
    """
    # SemSegMultiScalePyConvHead settings
    cfg.MODEL.SEM_SEG_HEAD.PYCONV_OUT_SIZE_LOCAL_CONTEXT = 512
    cfg.MODEL.SEM_SEG_HEAD.PYCONV_OUT_SIZE_GLOBAL_CONTEXT = 512
    cfg.MODEL.SEM_SEG_HEAD.PYCONV_MERGE_OUT_SIZE = 256
    cfg.MODEL.SEM_SEG_HEAD.PYCONV_LOCAL_REDUCTION = 4
    cfg.MODEL.SEM_SEG_HEAD.PYCONV_GLOBAL_BINS = 9
    cfg.MODEL.SEM_SEG_HEAD.PYCONV_CLS_DROPOUT = 0.1
    # Determine how to merge the outputs from PyConvSemHeads at different FPN levels.
    # "SUM" means summing up them while "CAT" stands for concatenating them along the depth dimension.
    cfg.MODEL.SEM_SEG_HEAD.PYCONV_FUSE_MODE = "SUM"
    # The variant of the Versatile Convolution. Any of conv_type = [VerConvSeparated, VerConv, PyConvSE]
    cfg.MODEL.SEM_SEG_HEAD.VERCONV = "VerConvSeparated"
    # The reduction rate in the fc layer of the SE-like module.
    cfg.MODEL.SEM_SEG_HEAD.REDUCT_RATE = 16


