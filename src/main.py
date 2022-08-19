import functools
import os
import shutil
import time
from datetime import timedelta as td
from typing import List, Optional, cast

import numpy as np
import torch

from models.skcart import SkCARTPredictor
from models.sklin import SkLinPredictor
from models.xdgt import DGTPredictor
from models.xlin import LinPredictor
from models.xvw import VWPredictor
from xcommon import Logger, Utils
from xconstants import (CONFIGS_ROOT_DIR, DESC_SEP,
                        EXP_LOGS_DIR, HPSEARCH_STATS, LOG_BASE_PERF, MAIN_PROGRESS_FILE,
                        MSTD_DIR, SAVE_PLOTS, EXP_OUT_DIR, NSEEDS, JOIN_TRAIN_VAL)
from xdata import DataLoader
from xhp import HpUtils
from xparser import Opts
from xmodelrunner import get_best_model
from xmstd import MstdUtils

modelnamedict = {
    'DGT' : DGTPredictor,
    'SkCART' : SkCARTPredictor,
    'Lin' : LinPredictor,
    'SkLin' : SkLinPredictor,
    'VW': VWPredictor
}

def run_tasks(exp_dir: str, log_path: str, args):
    Utils.cmlog = functools.partial(Utils.safe_log, path='{}/{}/cm.log'.format(exp_dir, EXP_LOGS_DIR))
    Utils.write_json(vars(args), f'{exp_dir}/{EXP_LOGS_DIR}/cmdline-args.json')

    model_class = modelnamedict[args.model]
    dataset_names = [args.dataset]
    mstd_group_by: Optional[List[str]] = ['height', 'sigquant', 'softquant', 'black_box', 'gamma', 'use_last_model']

    for dataset_name in dataset_names:
        normalize_x_kwargs = None
        normalize_y_kwargs = None

        if args.datanorm:
            normalize_x_kwargs = {'category': 'zscore'}

            if DataLoader.stats[dataset_name]['n_classes'] <= 0:
                normalize_y_kwargs = {'category': 'minmax', 'interval': (0, 1)}

        get_dataset_kwargs = {
            'dataset_name': dataset_name,
            'normalize_x_kwargs': normalize_x_kwargs,
            'normalize_y_kwargs': normalize_y_kwargs,
            'shuffle_seed': 1
        }

        train, validn, test = DataLoader.get_dataset(**get_dataset_kwargs)
        setattr(args, 'train_size', train._x.shape[0])
        if JOIN_TRAIN_VAL:
            train = Utils.join_datasets(train, validn)

        assert len(cast(np.ndarray, train['y']).shape) == 1
        assert len(cast(np.ndarray, validn['y']).shape) == 1
        assert len(cast(np.ndarray, test['y']).shape) == 1

        if LOG_BASE_PERF:
            logger = Logger(log_path)
            logger.log('\n{}\n'.format(DataLoader.get_base_perf(train, validn, test)))
            logger.close()

        if SAVE_PLOTS:
            train.visualize(save_path='{}/train-data.png'.format(exp_dir))
            test.visualize(save_path='{}/test-data.png'.format(exp_dir))
        mss = HpUtils.get_model_search_set(model_class, dataset_name, args)
        if mss.model_class in [DGTPredictor, LinPredictor]:
            train, validn, test = train.to_tensor(), validn.to_tensor(), test.to_tensor()

        get_best_model(
            log_path, mss, train, validn, test, get_dataset_kwargs,
            exp_dir=exp_dir, devices_info=args.DEVICES_INFO, show_hpsearch_stats=HPSEARCH_STATS
        )

        if args.compute_mstd:
            if args.model == 'SkCART' or args.model == 'SkLin':
                nseeds = NSEEDS
            else:
                nseeds = NSEEDS
            for i in os.listdir('{}/{}'.format(exp_dir, EXP_LOGS_DIR)):
                if '{}-search-summary'.format(dataset_name) in i:
                    MstdUtils.summary_file('{}/{}/{}'.format(exp_dir, EXP_LOGS_DIR, i), args, group_by=mstd_group_by, metric='validn_acc', nseeds=nseeds, nshuffleseeds=args.ndshuffles)
                    # MstdUtils.summary_file('{}/{}/{}'.format(exp_dir, EXP_LOGS_DIR, i), args, group_by=mstd_group_by, metric='dt_validn_acc', nshuffleseeds=args.ndshuffles)
                    # MstdUtils.summary_file('{}/{}/{}'.format(exp_dir, EXP_LOGS_DIR, i), group_by=mstd_group_by, metric='cdt_validn_acc', nshuffleseeds=args.ndshuffles)

def copy_src_files(exp_dir):
    src_files_dir = exp_dir + '/src-files'
    os.mkdir(src_files_dir)
    for f in os.listdir('.'):
        if os.path.isfile(f) and f.endswith('.py'):
            shutil.copy(f, src_files_dir)

    models_files_dir = src_files_dir + '/models'
    os.mkdir(models_files_dir)
    for f in os.listdir('./models'):
        if os.path.isfile('./models/{}'.format(f)) and f.endswith('.py'):
            shutil.copy('./models/{}'.format(f), models_files_dir)

"""
exp_dir
    /{config}
        /logs
        /plots
    /logs (EXP_LOGS_DIR)
        ...
    /src-files
        ...
"""
def main():
    assert os.path.split(os.getcwd())[1] == 'src', 'Set working directory to src to proceed'
    start_time = time.time()

    args = Opts()

    desc = args.description
    desc = '{}{}{}'.format(desc.strip(), DESC_SEP, str(start_time).split('.')[0])
    exp_dir = '{}/exp{}{}'.format(EXP_OUT_DIR, DESC_SEP, desc)
    os.mkdir(exp_dir)
    os.mkdir('{}/{}'.format(exp_dir, CONFIGS_ROOT_DIR))
    os.mkdir('{}/{}'.format(exp_dir, EXP_LOGS_DIR))
    os.mkdir('{}/{}'.format(exp_dir, MSTD_DIR))
    copy_src_files(exp_dir)

    log_path = '{}/{}/{}'.format(exp_dir, EXP_LOGS_DIR, MAIN_PROGRESS_FILE)
    logger = Logger(log_path)
    logger.log('Experiment dir: {}\n'.format(os.path.abspath(exp_dir)))
    logger.log('Logging to: {}\n\n'.format(os.path.abspath(log_path)))

    run_tasks(exp_dir, log_path, args)

    logger.log('\nExperiment dir: {}\n'.format(os.path.abspath(exp_dir)))
    logger.log('Logging to: {}\n'.format(os.path.abspath(log_path)))
    logger.log('Program runtime: {}\n\n'.format(td(seconds=time.time() - start_time)))
    logger.close()

if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(False)
    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError as e:
        if 'context has already been set' not in str(e):
            raise e
    main()
