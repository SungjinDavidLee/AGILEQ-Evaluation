import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()


_C.BASE = ['']


_C.DATA = CN()

_C.DATA.BATCH_SIZE = 128

_C.DATA.DATA_PATH = ''

_C.DATA.DATASET = 'imagenet'

_C.DATA.IMG_SIZE = 224

_C.DATA.INTERPOLATION = 'bicubic'


_C.DATA.ZIP_MODE = False

_C.DATA.CACHE_MODE = 'part'

_C.DATA.PIN_MEMORY = True

_C.DATA.NUM_WORKERS = 8

_C.DATA.IMG_ON_MEMORY = False


_C.MODEL = CN()

_C.MODEL.TYPE = 'INTERN_IMAGE'

_C.MODEL.NAME = 'intern_image'


_C.MODEL.PRETRAINED = ''

_C.MODEL.RESUME = ''

_C.MODEL.NUM_CLASSES = 1000

_C.MODEL.DROP_RATE = 0.0

_C.MODEL.DROP_PATH_RATE = 0.1

_C.MODEL.DROP_PATH_TYPE = 'linear'

_C.MODEL.LABEL_SMOOTHING = 0.1


_C.MODEL.INTERN_IMAGE = CN()
_C.MODEL.INTERN_IMAGE.DEPTHS = [4, 4, 18, 4]
_C.MODEL.INTERN_IMAGE.GROUPS = [4, 8, 16, 32]
_C.MODEL.INTERN_IMAGE.CHANNELS = 64
_C.MODEL.INTERN_IMAGE.LAYER_SCALE = None
_C.MODEL.INTERN_IMAGE.OFFSET_SCALE = 1.0
_C.MODEL.INTERN_IMAGE.MLP_RATIO = 4.0
_C.MODEL.INTERN_IMAGE.CORE_OP = 'DCNv3'
_C.MODEL.INTERN_IMAGE.POST_NORM = False
_C.MODEL.INTERN_IMAGE.RES_POST_NORM = False
_C.MODEL.INTERN_IMAGE.DW_KERNEL_SIZE = None
_C.MODEL.INTERN_IMAGE.USE_CLIP_PROJECTOR = False
_C.MODEL.INTERN_IMAGE.LEVEL2_POST_NORM = False
_C.MODEL.INTERN_IMAGE.LEVEL2_POST_NORM_BLOCK_IDS = None
_C.MODEL.INTERN_IMAGE.CENTER_FEATURE_SCALE = False
_C.MODEL.INTERN_IMAGE.REMOVE_CENTER = False


_C.MODEL.SWIN = CN()
_C.MODEL.SWIN.PATCH_SIZE = 4
_C.MODEL.SWIN.IN_CHANS = 3
_C.MODEL.SWIN.EMBED_DIM = 96
_C.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN.WINDOW_SIZE = 7
_C.MODEL.SWIN.MLP_RATIO = 4.
_C.MODEL.SWIN.QKV_BIAS = True
_C.MODEL.SWIN.QK_SCALE = None
_C.MODEL.SWIN.APE = False
_C.MODEL.SWIN.PATCH_NORM = True


_C.MODEL.SWIN_MOE = CN()
_C.MODEL.SWIN_MOE.PATCH_SIZE = 4
_C.MODEL.SWIN_MOE.IN_CHANS = 3
_C.MODEL.SWIN_MOE.EMBED_DIM = 96
_C.MODEL.SWIN_MOE.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN_MOE.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN_MOE.WINDOW_SIZE = 7
_C.MODEL.SWIN_MOE.MLP_RATIO = 4.
_C.MODEL.SWIN_MOE.QKV_BIAS = True
_C.MODEL.SWIN_MOE.QK_SCALE = None
_C.MODEL.SWIN_MOE.APE = False
_C.MODEL.SWIN_MOE.PATCH_NORM = True
_C.MODEL.SWIN_MOE.MLP_FC2_BIAS = True
_C.MODEL.SWIN_MOE.INIT_STD = 0.02
_C.MODEL.SWIN_MOE.PRETRAINED_WINDOW_SIZES = [0, 0, 0, 0]
_C.MODEL.SWIN_MOE.MOE_BLOCKS = [[-1], [-1], [-1], [-1]]
_C.MODEL.SWIN_MOE.NUM_LOCAL_EXPERTS = 1
_C.MODEL.SWIN_MOE.TOP_VALUE = 1
_C.MODEL.SWIN_MOE.CAPACITY_FACTOR = 1.25
_C.MODEL.SWIN_MOE.COSINE_ROUTER = False
_C.MODEL.SWIN_MOE.NORMALIZE_GATE = False
_C.MODEL.SWIN_MOE.USE_BPR = True
_C.MODEL.SWIN_MOE.IS_GSHARD_LOSS = False
_C.MODEL.SWIN_MOE.GATE_NOISE = 1.0
_C.MODEL.SWIN_MOE.COSINE_ROUTER_DIM = 256
_C.MODEL.SWIN_MOE.COSINE_ROUTER_INIT_T = 0.5
_C.MODEL.SWIN_MOE.MOE_DROP = 0.0
_C.MODEL.SWIN_MOE.AUX_LOSS_WEIGHT = 0.01


_C.MODEL.SWIN_MLP = CN()
_C.MODEL.SWIN_MLP.PATCH_SIZE = 4
_C.MODEL.SWIN_MLP.IN_CHANS = 3
_C.MODEL.SWIN_MLP.EMBED_DIM = 96
_C.MODEL.SWIN_MLP.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN_MLP.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN_MLP.WINDOW_SIZE = 7
_C.MODEL.SWIN_MLP.MLP_RATIO = 4.
_C.MODEL.SWIN_MLP.APE = False
_C.MODEL.SWIN_MLP.PATCH_NORM = True


_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 300
_C.TRAIN.WARMUP_EPOCHS = 20
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 5e-4
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6

_C.TRAIN.CLIP_GRAD = 5.0

_C.TRAIN.AUTO_RESUME = True


_C.TRAIN.ACCUMULATION_STEPS = 0


_C.TRAIN.USE_CHECKPOINT = False


_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'

_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30

_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1


_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'

_C.TRAIN.OPTIMIZER.EPS = 1e-8

_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)

_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

_C.TRAIN.OPTIMIZER.USE_ZERO = False

_C.TRAIN.OPTIMIZER.FREEZE_BACKBONE = None

_C.TRAIN.OPTIMIZER.DCN_LR_MUL = None


_C.TRAIN.EMA = CN()
_C.TRAIN.EMA.ENABLE = False
_C.TRAIN.EMA.DECAY = 0.9998


_C.TRAIN.LR_LAYER_DECAY = False
_C.TRAIN.LR_LAYER_DECAY_RATIO = 0.875


_C.TRAIN.RAND_INIT_FT_HEAD = False


_C.AUG = CN()

_C.AUG.COLOR_JITTER = 0.4

_C.AUG.AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1'

_C.AUG.REPROB = 0.25

_C.AUG.REMODE = 'pixel'

_C.AUG.RECOUNT = 1

_C.AUG.MIXUP = 0.8

_C.AUG.CUTMIX = 1.0

_C.AUG.CUTMIX_MINMAX = None

_C.AUG.MIXUP_PROB = 1.0

_C.AUG.MIXUP_SWITCH_PROB = 0.5

_C.AUG.MIXUP_MODE = 'batch'

_C.AUG.RANDOM_RESIZED_CROP = False
_C.AUG.MEAN = (0.485, 0.456, 0.406)
_C.AUG.STD = (0.229, 0.224, 0.225)


_C.TEST = CN()

_C.TEST.CROP = True


_C.TEST.SEQUENTIAL = False


_C.AMP_OPT_LEVEL = ''

_C.OUTPUT = ''

_C.TAG = 'default'

_C.SAVE_FREQ = 1

_C.PRINT_FREQ = 10

_C.EVAL_FREQ = 1

_C.SEED = 0

_C.EVAL_MODE = False

_C.THROUGHPUT_MODE = False

_C.LOCAL_RANK = 0
_C.EVAL_22K_TO_1K = False

_C.AMP_TYPE = 'float16'


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg))
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_file = os.path.join(current_dir, 'internimage_t_1k_224.yaml')

    _update_config_from_file(config, cfg_file)

    config.defrost()


    config.MODEL.NAME = 'internimage_t_1k_224'
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME)

    config.freeze()


def get_config_t():


    config = _C.clone()
    update_config(config)

    return config

def update_config2(config):
    _update_config_from_file(config, 'configs/internimage_b_1k_224.yaml')

    config.defrost()


    config.MODEL.NAME = 'configs/internimage_b_1k_224.yaml'.split('/')[-1].replace('.yaml', '')
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME)

    config.freeze()


def get_config_b():


    config = _C.clone()
    update_config2(config)

    return config
