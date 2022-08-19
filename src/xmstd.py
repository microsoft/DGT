import os
import pickle
import time
from typing import List, Optional

import pandas as pd
from xcommon import Logger, Utils
from typing import List, Optional

from xconstants import (CONFIG_REPRO_DATA_FILE, CONFIGS_ROOT_DIR, DESC_SEP,
                        EXP_LOGS_DIR, IGNORE_LOW_SAT_CONFIGS, MAIN_PROGRESS_FILE,
                        MSTD_DIR, MSTD_SUMMARY_STATS_FILE, NSEEDS, JOIN_TRAIN_VAL)
from xdata import DataLoader
from xhp import ModelSearchSet
from xmodelrunner import get_best_model
from models.xdgt import DGTPredictor
from models.xlin import LinPredictor

class MstdUtils:
    """
    Get the best config in a summary file and get mstd stats.
    The manyruns stats for the config largest according to 'metric' in every group given
    by 'group_by' is computed.
    summary_file is assumed to be part of the usual experiment hierarchy
    """
    @staticmethod
    def summary_file(path: str, args, group_by: Optional[List[str]]=None, metric: Optional[str]=None, nseeds: int=NSEEDS, nshuffleseeds: int=1):
        df = pd.read_csv(path)
        if 'black_box' in df.columns:
            df['black_box'] = df['black_box'].fillna('None')
        if 'ignore_config' in df.columns and IGNORE_LOW_SAT_CONFIGS:
            df = df[df['ignore_config'] == 0]

        assert len(df) > 0
        assert df['dataset'].nunique(dropna=False) == 1
        dataset = df['dataset'].iloc[0]
        assert df['shuffle_seed'].nunique(dropna=False) == 1
        shuffle_seed = df['shuffle_seed'].iloc[0]

        is_classification = DataLoader.is_classification(dataset)

        ms = ([
            'dt_validn_auc',
            'dt_validn_acc',
            'validn_auc',
            'validn_acc'
        ])
        if metric is None:
            for m in ms:
                if m in df.columns:
                    metric = m
                    break
        else:
            if metric not in df.columns:
                return
        assert metric is not None

        if group_by is not None:
            group_by = list(set(group_by).intersection(set(df.columns)))
            if len(group_by) == 0:
                group_by = None

        if group_by is None:
            df = df.loc[[df[metric].idxmax()]] if is_classification else df.loc[[df[metric].idxmin()]]
        else:
            gb = df.groupby(by=group_by, dropna=False)[metric]
            df = df.loc[gb.idxmax()] if is_classification else df.loc[gb.idxmin()]

        exp_dir = os.path.abspath('{}/../..'.format(path))
        for idx, row in df.iterrows():
            desc = [f'{dataset}-{shuffle_seed}']
            desc.append(metric)
            if group_by is not None:
                for gb_col in group_by:
                    desc.append(MstdUtils.short_col[gb_col] + '-' + str(row[gb_col]))
            desc = DESC_SEP.join(desc)
            config_dir = '{}/configs/{}'.format(exp_dir, row['config_dir'].split('configs/' if 'configs/' in row['config_dir'] else 'configs\\')[1].strip()) # to handle cases where exp dir as a whole is moved (say to a different machine)
            MstdUtils.config(config_dir, args, desc=desc, verify_row=row, nseeds=nseeds, group_by_metric=metric, nshuffleseeds=nshuffleseeds)

    """
    Get mstd stats for a config

    Given a config, creates a new experiment where that config is run many times and the summary file is parsed
    to give final stats in the main progress file of this new experiment.
    """
    @staticmethod
    def config(config_dir: str, args, exp_root=None, desc: str='', verify_row: Optional[pd.Series]=None, nseeds: int=NSEEDS, group_by_metric=None, nshuffleseeds: int=1):
        time_str = str(time.time()).split(".")[0]

        # Load datasets and other data needed
        if config_dir[-1] in ['/', '\\']:
            config_dir = config_dir[:-1]

        with open('{}/{}.pkl'.format(config_dir, CONFIG_REPRO_DATA_FILE), 'rb') as f:
            repro_data = pickle.load(f)

        model_class = repro_data['model_class']
        hp = repro_data['hp']
        get_dataset_kwargs = repro_data['get_dataset_kwargs']

        # Compute mstd stats
        hps = {}
        for k, v in hp.items():
            hps[k] = [v]
        hps['seed'] = list(range(1, nseeds + 1))

        if verify_row is not None:
            assert repro_data['seed'] == verify_row['seed']
        if repro_data['seed'] not in hps['seed']:
            hps['seed'][0] = repro_data['seed']

        mss = ModelSearchSet(model_class, [hps])

        if exp_root is None:
            exp_root = f'{Utils.get_exp_dir(config_dir)}/{MSTD_DIR}/{desc}'
        try:
            os.makedirs(exp_root)
        except FileExistsError:
            pass

        desc = f'{desc.strip()}{DESC_SEP}{time_str}'
        shuffle_seeds = list(range(1, nshuffleseeds + 1))
        if get_dataset_kwargs['shuffle_seed'] not in shuffle_seeds:
            shuffle_seeds[0] = get_dataset_kwargs['shuffle_seed']
        orig_shuffle_seed = get_dataset_kwargs['shuffle_seed']

        local_mstd_summary_file = f'{exp_root}/local-meanstd-run-summary.csv'
        concat_local_mstd_summary_file = f'{exp_root}/../concat-local-meanstd-run-summary.csv'
        mstd_summary_file = f'{exp_root}/../meanstd-run-summary.csv'

        gethp_output = model_class(**hp).get_hyperparams()
        mstd_dict_aux = {}
        for shuffle_seed in shuffle_seeds:

            get_dataset_kwargs['shuffle_seed'] = shuffle_seed
            train_data, validn_data, test_data = DataLoader.get_dataset(**get_dataset_kwargs)
            if JOIN_TRAIN_VAL:
                train_data = Utils.join_datasets(train_data, validn_data)
            if model_class in [DGTPredictor, LinPredictor]:
                train_data, validn_data, test_data = train_data.to_tensor(), validn_data.to_tensor(), test_data.to_tensor()

            # curr_desc = f'{get_dataset_kwargs["dataset_name"]}-{shuffle_seed}{DESC_SEP}{desc.split(DESC_SEP, maxsplit=1)[1]}'
            curr_desc = f'shuffle_seed-{shuffle_seed}'
            exp_dir = f'{exp_root}/exp{DESC_SEP}{curr_desc}{DESC_SEP}{time_str}'

            os.mkdir(exp_dir)
            os.mkdir('{}/{}'.format(exp_dir, CONFIGS_ROOT_DIR))
            logs_dir = '{}/{}'.format(exp_dir, EXP_LOGS_DIR)
            os.mkdir(logs_dir)
            log_path = '{}/{}'.format(logs_dir, MAIN_PROGRESS_FILE)

            get_best_model(
                log_path, mss, train_data, validn_data, test_data, get_dataset_kwargs, exp_dir=exp_dir, devices_info=args.DEVICES_INFO, show_hpsearch_stats=False)

            summary_file = '{}/{}/{}-search-summary{}{}.csv'.format(exp_dir, EXP_LOGS_DIR, get_dataset_kwargs['dataset_name'], DESC_SEP, time_str)

            # Verify
            if (shuffle_seed == orig_shuffle_seed) and (verify_row is not None):
                df = pd.read_csv(summary_file)
                new = df.loc[df['seed'] == repro_data['seed']]
                assert len(new) == 1
                new = new.iloc[0]
                orig = verify_row
                assert set(orig.index) == set(new.index)

                sel = set([
                    'train_acc', 'validn_acc', 'test_acc',
                    'train_auc', 'validn_auc', 'test_auc',
                    'dt_train_acc', 'dt_validn_acc', 'dt_test_acc',
                    'dt_train_auc', 'dt_validn_auc', 'dt_test_auc',
                    'cdt_train_acc', 'cdt_validn_acc', 'cdt_test_acc'
                ]).intersection(set(orig.index)).intersection(set(new.index))
                diff = new.loc[sel] - orig.loc[sel]

                s = ''
                s += '\n{}\n'.format(Utils.get_padded_text('Verifying'))
                s += '\n--------\nOriginal\n--------\n{}'.format(orig)
                s += '\n---\nNew\n---\n{}'.format(new)
                s += '\n---------------\nDiff (new-orig)\n---------------\n{}'.format(diff)
                s += '\n{}\n'.format(Utils.get_padded_text(''))

                logger = Logger(log_path)
                logger.log(s, stdout=False)
                logger.close()

            summary_stats = MstdUtils.log_stats(summary_file, logs_dir, verify_row['config_idx'])

            # add a line to local-meanstd-run-summary.csv
            local_mstd_dict = {'shuffle_seed': shuffle_seed}
            for metric in MstdUtils.metrics:
                if metric in summary_stats['metrics']:
                    local_mstd_dict[f'{metric}_mean'] = summary_stats['metrics'][metric]['mean']
                    local_mstd_dict[f'{metric}_median'] = summary_stats['metrics'][metric]['median']
                    local_mstd_dict[f'{metric}_std'] = summary_stats['metrics'][metric]['std']

                    if metric not in mstd_dict_aux:
                        mstd_dict_aux[metric] = []
                    mstd_dict_aux[metric].extend(summary_stats['metrics'][metric]['vals'])

            local_mstd_dict.update(gethp_output)
            local_mstd_dict['group_by_metric'] = group_by_metric

            Utils.append_linedict_to_csv(local_mstd_dict, local_mstd_summary_file)
            Utils.append_linedict_to_csv(local_mstd_dict, concat_local_mstd_summary_file)

        # add a line to meanstd-run-summary.csv
        mstd_dict = {}
        for metric in mstd_dict_aux:
            mstd_dict[f'{metric}_mean'] = pd.Series(mstd_dict_aux[metric]).mean()
            mstd_dict[f'{metric}_std'] = pd.Series(mstd_dict_aux[metric]).std(ddof=0)
        mstd_dict.update(gethp_output)
        mstd_dict['group_by_metric'] = group_by_metric
        for metric in mstd_dict_aux:
            mstd_dict[f'{metric}_nancount'] = int(pd.Series(mstd_dict_aux[metric]).isna().sum())
        Utils.append_linedict_to_csv(mstd_dict, mstd_summary_file)

    """
    Compute and log stats of a summary file whose configs represent results from various seeds/starting points
    """
    @staticmethod
    def log_stats(summary_file: str, logs_dir: str, config_idx: Optional[int]=None):
        log_path = '{}/{}'.format(logs_dir, MAIN_PROGRESS_FILE)
        logger = Logger(log_path)
        logger.log('config_idx={}\n'.format(config_idx))

        df = pd.read_csv(summary_file)

        summary_stats = {'config_idx': config_idx, 'metrics': {}}

        for idx, metric in enumerate(MstdUtils.metrics):
            if metric in df.columns:
                mean = df[metric].mean()
                median = df[metric].median()
                std = df[metric].std(ddof=0)
                mn = df[metric].min()
                mx = df[metric].max()
                nancount = int(df[metric].isna().sum())
                vals = df[metric].tolist()

                summary_stats['metrics'][metric] = { 'mean': mean, 'median': median, 'std': std, 'min': mn, 'max': mx, 'nancount': nancount, 'vals': vals }
                logger.log(
                    '{:<14} - mean:{:.5f}, median:{:.5f}, std:{:.5f}, min:{:.5f}, max:{:.5f}, nancount:{}\n'.format(
                        metric, mean, median, std, mn, mx, nancount), stdout=True)
                if idx % 3 == 2:
                    logger.log('\n')

        logger.close()
        Utils.write_json(summary_stats, '{}/{}'.format(logs_dir, MSTD_SUMMARY_STATS_FILE))
        return summary_stats

    metrics = ([
        'train_acc', 'validn_acc', 'test_acc',
        'train_auc', 'validn_auc', 'test_auc',
        'dt_train_acc', 'dt_validn_acc', 'dt_test_acc',
        'dt_train_auc', 'dt_validn_auc', 'dt_test_auc',
        'cdt_train_acc', 'cdt_validn_acc', 'cdt_test_acc',
    ])

    short_col = {
        'height': 'h',
        'optimizer': 'opt',
        'black_box': 'bb',
        'use_last_model': 'ulm',

        'gamma' : 'gamma',

        'h': 'height',
        'opt': 'optimizer',
        'bb': 'black_box',
        'ulm': 'use_last_model',


        'sigquant' : 'sigq',
        'softquant' : 'sofq',
        'sigq' : 'sigquant',
        'sofq' : 'softquant',


        'batch_norm': 'bn',
        'bn': 'batch_norm',

        'batch_size': 'bs',
        'bs': 'batch_size',

        'optimizer_kwargs': 'optkw',
        'optkw': 'optimizer_kwargs',

        'lr_scheduler': 'lrsched',
        'lrsched': 'lr_scheduler',

        'lr_scheduler_kwargs': 'lrkw',
        'lrkw': 'lr_scheduler_kwargs',

        'l2_lambda': 'l2',
        'l2': 'l2_lambda',

        'l1_lambda': 'l1',
        'l1': 'l1_lambda',

        'over_param': 'op',
        'op': 'over_param',
    }
