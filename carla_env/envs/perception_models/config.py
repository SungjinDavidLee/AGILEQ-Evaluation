# --------------------------------------------------------
# InternImage
# Copyright (c) 2022 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 128
# Path to dataset, could be overwritten by command line argument
_C.DATA.DATA_PATH = ''
# Dataset name
_C.DATA.DATASET = 'imagenet'
# Input image size
_C.DATA.IMG_SIZE = 224
# Interpolation to resize image (random, bilinear, bicubic)
_C.DATA.INTERPOLATION = 'bicubic'
# Use zipped dataset instead of folder dataset
# could be overwritten by command line argument
_C.DATA.ZIP_MODE = False
# Cache Data in Memory, could be overwritten by command line argument
_C.DATA.CACHE_MODE = 'part'
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 8
# Load data to memory
_C.DATA.IMG_ON_MEMORY = False

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type
_C.MODEL.TYPE = 'INTERN_IMAGE'
# Model name
_C.MODEL.NAME = 'intern_image'
# Pretrained weight from checkpoint, could be imagenet22k pretrained weight
# could be overwritten by command line argument
_C.MODEL.PRETRAINED = ''
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ''
# Number of classes, overwritten in data preparation
_C.MODEL.NUM_CLASSES = 1000
# Dropout rate
_C.MODEL.DROP_RATE = 0.0
# Drop path rate
_C.MODEL.DROP_PATH_RATE = 0.1
# Drop path type
_C.MODEL.DROP_PATH_TYPE = 'linear'  # linear, uniform
# Label Smoothing
_C.MODEL.LABEL_SMOOTHING = 0.1

# INTERN_IMAGE parameters
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

# Swin Transformer MoE parameters
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

# Swin MLP parameters
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

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 300
_C.TRAIN.WARMUP_EPOCHS = 20
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 5e-4
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 5.0
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = True
# Gradient accumulation steps
# could be overwritten by command line argument
_C.TRAIN.ACCUMULATION_STEPS = 0
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
_C.TRAIN.USE_CHECKPOINT = False

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9
# ZeRO
_C.TRAIN.OPTIMIZER.USE_ZERO = False
# freeze backbone
_C.TRAIN.OPTIMIZER.FREEZE_BACKBONE = None
# dcn lr
_C.TRAIN.OPTIMIZER.DCN_LR_MUL = None

# EMA
_C.TRAIN.EMA = CN()
_C.TRAIN.EMA.ENABLE = False
_C.TRAIN.EMA.DECAY = 0.9998

# LR_LAYER_DECAY
_C.TRAIN.LR_LAYER_DECAY = False
_C.TRAIN.LR_LAYER_DECAY_RATIO = 0.875

# FT head init weights
_C.TRAIN.RAND_INIT_FT_HEAD = False

# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
# Color jitter factor
_C.AUG.COLOR_JITTER = 0.4
# Use AutoAugment policy. "v0" or "original"
_C.AUG.AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1'
# Random erase prob
_C.AUG.REPROB = 0.25
# Random erase mode
_C.AUG.REMODE = 'pixel'
# Random erase count
_C.AUG.RECOUNT = 1
# Mixup alpha, mixup enabled if > 0
_C.AUG.MIXUP = 0.8
# Cutmix alpha, cutmix enabled if > 0
_C.AUG.CUTMIX = 1.0
# Cutmix min/max ratio, overrides alpha and enables cutmix if set
_C.AUG.CUTMIX_MINMAX = None
# Probability of performing mixup or cutmix when either/both is enabled
_C.AUG.MIXUP_PROB = 1.0
# Probability of switching to cutmix when both mixup and cutmix enabled
_C.AUG.MIXUP_SWITCH_PROB = 0.5
# How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
_C.AUG.MIXUP_MODE = 'batch'
# RandomResizedCrop
_C.AUG.RANDOM_RESIZED_CROP = False
_C.AUG.MEAN = (0.485, 0.456, 0.406)
_C.AUG.STD = (0.229, 0.224, 0.225)

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Whether to use center crop when testing
_C.TEST.CROP = True

# Whether to use SequentialSampler as validation sampler
_C.TEST.SEQUENTIAL = False

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# Mixed precision opt level, if O0, no amp is used ('O0', 'O1', 'O2')
# overwritten by command line argument
_C.AMP_OPT_LEVEL = ''
# Path to output folder, overwritten by command line argument
_C.OUTPUT = ''
# Tag of experiment, overwritten by command line argument
_C.TAG = 'default'
# Frequency to save checkpoint
_C.SAVE_FREQ = 1
# Frequency to logging info
_C.PRINT_FREQ = 10
# eval freq
_C.EVAL_FREQ = 1
# Fixed random seed
_C.SEED = 0
# Perform evaluation only, overwritten by command line argument
_C.EVAL_MODE = False
# Test throughput only, overwritten by command line argument
_C.THROUGHPUT_MODE = False
# local rank for DistributedDataParallel, given by command line argument
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

    # output folder
    config.MODEL.NAME = 'internimage_t_1k_224'
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME)

    config.freeze()


def get_config_t():
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config)

    return config

def update_config2(config):
    _update_config_from_file(config, 'configs/internimage_b_1k_224.yaml')

    config.defrost()

    # output folder
    config.MODEL.NAME = 'configs/internimage_b_1k_224.yaml'.split('/')[-1].replace('.yaml', '')
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME)

    config.freeze()


def get_config_b():
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config2(config)

    return config
