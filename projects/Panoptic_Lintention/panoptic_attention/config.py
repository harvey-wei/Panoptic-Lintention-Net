
def add_panoptic_ppa_config(cfg):
    """
    Add config for panoptic PPAttention
    cfg: the YAML configuration for the default config.
    """

    # cfg.MODEL.SEM_SEG_HEAD.PPA_UNIT_TYPE = "PPA"
    cfg.MODEL.SEM_SEG_HEAD.PPA_NUM_HEADS = 4
    cfg.MODEL.SEM_SEG_HEAD.PPA_NUM_PATCHES = 16
    cfg.MODEL.SEM_SEG_HEAD.PPA_QUERY_PROJECT = False
    cfg.MODEL.SEM_SEG_HEAD.PPA_PATCHES_PROJECT = False
    cfg.MODEL.SEM_SEG_HEAD.PPA_POSITION_EMBED = False
    cfg.MODEL.SEM_SEG_HEAD.PPA_DROPOUT = 0.1
    cfg.MODEL.SEM_SEG_HEAD.PPA_FFN_EXPANSION = 4


