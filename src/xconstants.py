import os
import torch
from typing import Tuple
from enum import Enum, unique

@unique
class AccFuncType(Enum):
    acc = 1
    mse = 2
    rmse = 3
    criterion = 4

FAST: bool = True

REMOTE_SESS: bool = True # Changes matplotlib backend
PROGRESS_PLOTS: bool = not FAST
SAVE_TREE: bool = not FAST
SAVE_CDT_TREE: bool = False

DENSE_PROGRESS_LOG_EPOCH_EVERY: int = 10
REG_ACC = [AccFuncType.mse, AccFuncType.rmse, AccFuncType.criterion][1]
DENORMALIZE_RESULTS: bool = True

DEBUG_MODE: bool = not FAST
DEBUG_LOG_NET_EPOCH_EVERY: int = 10 if DEBUG_MODE else 0 # print weights
DEBUG_INPUT_DATA: bool = True # Print layer outputs and control DEBUG_Y?
COMPARE_LABEL_DIST_EPOCH_EVERY: int = DEBUG_MODE

JOIN_TRAIN_VAL: bool = False
TAO: bool = False
HPSEARCH_STATS: bool = True

################ main.py ################
EXP_OUT_DIR = '../out'

DESC_SEP: str = '@'
CONFIGS_ROOT_DIR: str = 'configs'
CONFIG_LOGS_DIR: str = 'logs'
CONFIG_PLOTS_DIR: str = 'plots'
EXP_LOGS_DIR: str = 'logs'
MSTD_DIR: str = 'meanstd-exps'
MSTD_SUMMARY_STATS_FILE: str = 'stats.json'
CONFIG_REPRO_DATA_FILE: str = 'repro-data'
MAIN_PROGRESS_FILE = 'main-progress.log'

LOG_BASE_PERF: bool = True

DENSE_LOG_BB1P_COUNT = 2000
################ xmodelrunner.py ################
LOG_DATASET_STATS: bool = True

SAVE_PLOTS: bool = False # Saves data plots and model decision plots
POST_TRAIN_PLOT_TEST: bool = False # Post-train decision surfaces using test or train samples?

IGNORE_LOW_SAT_CONFIGS: bool = False

################ models/xdgt.py ################
DEBUG_PREDICATES: bool = False

LR_SCHED_STEPS:int = 10
INIT_MODEL_WITH_CART: bool = False

ZERO_INIT_PRED_BIAS: bool = True#False
APPLY_NORM_TO_PREDPOOL: bool = False

# Compare pred and targ label distributions
LOG_PRED_TARG_DIFF: bool = False

# Controls printing weights, DEBUG_INPUT_DATA, DEBUG_Y
LOG_INIT_FIN_WTS: bool = not FAST
DEBUG_LOG_NET_BATCH_EVERY: int = 0
DEBUG_LOG_NET_BATCH_FIRST: int = 1
DEBUG_Y: bool = False

SAT_INFO_EPOCH_EVERY: int = 0 if DEBUG_MODE else 0
# Number of int_nodes upto which sat info is provided individually for each node,
# after which the mean over nodes is used.
SAT_INFO_NODE_MAX: int = 64
# How to discretize the saturation info distribution (the one with abs diff from 0.5)
# The numbers here define upper bounds for the intervals
SAT_INFO_DIST_DISC: Tuple = (0.25, 0.4, 0.49)

FPTYPE: torch.dtype = torch.float32 # applicable only to DGT

SIGSLOPE_MAX: int = 100000000

@unique
class BB_ENUM(Enum):
    BB_SMART_1P = 'smart1p'
    BB_SMART_2P = 'smart2p'
    BB_DIRECT = 'direct'
    BB_CLASS = 'class'
    BB_QUANT_DT = 'quantdt'
    BB_QUANT_CDT = 'quantcdt'

    def __str__(self):
        return self.value

    def __eq__(self, o):
        return str(self) == str(o)

PROFILE_TIME: bool = False

################ xcommon.py ################
COMPUTE_BALANCED_ACC: bool = False
CMAP = 'RdYlGn'
MARKER_SIZE = 1
DETERMINISTIC: bool = True
INT_NODE_REPR_MAX_FEATURES: int = 15

################ xdata.py ################
DATASETS_ROOT = '../datasets'

################ xmstd.py ################
NSEEDS: int = 10