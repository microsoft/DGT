import functools
import multiprocessing
import os
import pickle
import time
from collections import OrderedDict
from datetime import timedelta as td
from typing import Any, Dict, List, Tuple, Type, cast

import numpy as np
import torch
from xconstants import REMOTE_SESS
if REMOTE_SESS:
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from torch.multiprocessing import Lock, Value, spawn

from models.skcart import SkCARTPredictor
from models.xdgt import DGTPredictor
from models.xvw import VWPredictor
from xcommon import (Dataset, DTExtractablePredictor, LearnablePredictor,
                     Logger, Utils)
from xconstants import (CONFIG_LOGS_DIR, CONFIG_PLOTS_DIR, SAVE_CDT_TREE,
                        CONFIG_REPRO_DATA_FILE, CONFIGS_ROOT_DIR, DESC_SEP,
                        EXP_LOGS_DIR, LOG_DATASET_STATS, POST_TRAIN_PLOT_TEST,
                        SAVE_PLOTS, SAVE_TREE, DENORMALIZE_RESULTS)
from xhp import ModelSearchSet
from xdata import DataLoader
from xhpsearch_stats import hpsearch_stats

CPU_MODELS = [VWPredictor]

r"""
Setup logger for the model and return output paths
"""
def setup_config_dir(
    model: LearnablePredictor, dataset_name: str, dseed: int, exp_dir: str, config_idx: int
) -> Tuple[str, str, str]:

    config = '{}{}{}'.format(type(model).__name__, DESC_SEP, dataset_name)
    for k, v in model.get_hyperparams().items():
        config += '{}{}={}'.format(DESC_SEP, k, Utils.shorten_mid(str(v), begin_keep=1000, end_keep=1000) if (isinstance(v, list) or isinstance(v, np.ndarray)) else v)

    # TODO: config will be used to create directory and it may have illegal characters
    config_dir = f'{exp_dir}/{CONFIGS_ROOT_DIR}/{config_idx}{DESC_SEP}{dataset_name}-{dseed}{DESC_SEP}{type(model).__name__}'
    logs_dir = f'{config_dir}/{CONFIG_LOGS_DIR}'
    plots_dir = f'{config_dir}/{CONFIG_PLOTS_DIR}'
    os.mkdir(config_dir)
    os.mkdir(logs_dir)
    os.mkdir(plots_dir)
    model.log_f = Logger('{}/progress.log'.format(logs_dir)).log

    model.logs_dir = logs_dir
    model.plots_dir = plots_dir

    img_save_prefix = plots_dir
    return config, config_dir, img_save_prefix

def process_model(
    model: LearnablePredictor,
    train_data: Dataset,
    validn_data: Dataset,
    test_data: Dataset,
    config_dir: str,
    config_idx: int,
    seed: int,
    compute_auc: bool
) -> Dict:
    if SAVE_PLOTS:
        save_path = '{}/plots/model-decisions-post-train.png'.format(config_dir)
        model.visualize_decisions(test_data['x'] if POST_TRAIN_PLOT_TEST else train_data['x'], save_path=save_path)

    train_acc = model.acc(train_data, denormalize=DENORMALIZE_RESULTS)
    validn_acc = model.acc(validn_data, denormalize=DENORMALIZE_RESULTS)
    test_acc = model.acc(test_data, denormalize=DENORMALIZE_RESULTS)

    if compute_auc:
        train_auc = model.auc(train_data)
        validn_auc = model.auc(validn_data)
        test_auc = model.auc(test_data)

    # Get dt_acc
    if isinstance(model, DTExtractablePredictor):
        if not model._is_pure_dt:
            dt_c = model.extract_dt_predictor()

            dt_c.acc_func = model.acc_func
            dt_c.acc_func_type = model.acc_func_type

            dt_train_acc = dt_c.acc(train_data, denormalize=DENORMALIZE_RESULTS)
            dt_validn_acc = dt_c.acc(validn_data, denormalize=DENORMALIZE_RESULTS)
            dt_test_acc = dt_c.acc(test_data, denormalize=DENORMALIZE_RESULTS)

            if compute_auc:
                dt_train_auc = dt_c.auc(train_data)
                dt_validn_auc = dt_c.auc(validn_data)
                dt_test_auc = dt_c.auc(test_data)

        if (SAVE_PLOTS or SAVE_TREE) and (model._is_pure_dt):
            dt_c = model.extract_dt_predictor()
        if SAVE_PLOTS:
            dt_c.visualize_decisions(cast(torch.Tensor, test_data['x'] if POST_TRAIN_PLOT_TEST else train_data['x']), save_path='{}/plots/model-tree-decisions-post-train.png'.format(config_dir))
        if SAVE_TREE:
            dt_c.visualize_tree(save_path='{}/{}/model-tree.svg'.format(config_dir, CONFIG_PLOTS_DIR), data=train_data)

    if isinstance(model, DTExtractablePredictor) and isinstance(model, DGTPredictor):
        if not model._is_pure_dt:
            cdt_c = model.extract_cdt_predictor(train_data)
            cdt_c.acc_func = model.acc_func
            cdt_c.acc_func_type = model.acc_func_type

            cdt_train_acc = cdt_c.acc(train_data, denormalize=DENORMALIZE_RESULTS)
            cdt_validn_acc = cdt_c.acc(validn_data, denormalize=DENORMALIZE_RESULTS)
            cdt_test_acc = cdt_c.acc(test_data, denormalize=DENORMALIZE_RESULTS)

        if SAVE_CDT_TREE:
            if model._is_pure_dt:
                cdt_c = model.extract_cdt_predictor(train_data)
            cdt_c.visualize_tree(
                save_path='{}/{}/model-ctree.svg'.format(config_dir, CONFIG_PLOTS_DIR), data=train_data
            )

    if isinstance(model, SkCARTPredictor) and SAVE_TREE:
        fig, ax = plt.subplots(dpi=400)
        plot_tree(model._model, ax=ax, precision=5)
        plt.savefig('{}/{}/model-tree.png'.format(config_dir, CONFIG_PLOTS_DIR))
        plt.close(fig)

    # Collect stats
    stats: Dict[str, Any] = OrderedDict()
    stats['train_acc'] = train_acc
    stats['validn_acc'] = validn_acc
    stats['test_acc'] = test_acc
    if compute_auc:
        stats['train_auc'] = train_auc
        stats['validn_auc'] = validn_auc
        stats['test_auc'] = test_auc
    if isinstance(model, DTExtractablePredictor):
        if not model._is_pure_dt:
            stats['dt_train_acc'] = dt_train_acc
            stats['dt_validn_acc'] = dt_validn_acc
            stats['dt_test_acc'] = dt_test_acc
            if compute_auc:
                stats['dt_train_auc'] = dt_train_auc
                stats['dt_validn_auc'] = dt_validn_auc
                stats['dt_test_auc'] = dt_test_auc
    if isinstance(model, DTExtractablePredictor) and isinstance(model, DGTPredictor):
        if not model._is_pure_dt:
            stats['cdt_train_acc'] = cdt_train_acc
            stats['cdt_validn_acc'] = cdt_validn_acc
            stats['cdt_test_acc'] = cdt_test_acc
    stats['config_idx'] = config_idx
    stats['model_name'] = type(model).__name__
    stats['dataset'] = train_data.name
    stats['seed'] = seed
    stats['shuffle_seed'] = train_data.shuffle_seed

    stats.update(model.get_hyperparams())
    for key, val in stats.items():
        if isinstance(val, list) or isinstance(val, np.ndarray):
            stats[key] = Utils.shorten_mid(str(val), begin_keep=1000, end_keep=1000)

    """
    if isinstance(model, DGTPredictor):
        sat_info = model.get_sat_info(train_data)
        stats['sat_info'] = sat_info
        stats['ignore_config'] = 0
    """

    stats['config_dir'] = os.path.abspath(config_dir)

    return stats

def get_best_model_aux(
    proc_num: int,
    l: 'multiprocessing.synchronize.Lock', # lock for coordinating MAIN_PROGRESS_FILE logging
    cmlock: 'multiprocessing.synchronize.Lock', # common lock for coordinating all other logging (typically for small messages)
    started_hps: Value,
    started_hps_lk: 'multiprocessing.synchronize.Lock',
    finished_hps: Value,
    total_hps: int,
    start_time: float,
    model_class: Type[LearnablePredictor],
    proc_hps: List[List[Dict[str, Any]]],
    train_data: Dataset,
    validn_data: Dataset,
    test_data: Dataset,
    get_dataset_kwargs: Dict[str, Any],
    exp_dir: str,
    proc_device_id: List[int],
    log_path: str,
    model_search_summary_path: str,
    use_lforb: bool
):
    # Prep process-specific data
    hps = proc_hps[proc_num]
    device_ids = None if proc_device_id[proc_num] == -1 else [proc_device_id[proc_num]]

    assert train_data.n_labels == 1
    compute_auc = Utils.is_binary_labels(train_data.to_ndarray()['y'])

    Utils.cmlog = functools.partial(Utils.safe_log, path='{}/{}/cm.log'.format(exp_dir, EXP_LOGS_DIR), l=cmlock)

    logger = Logger(log_path)
    for i, hp in enumerate(hps):
        with started_hps_lk:
            config_idx = started_hps.value
            started_hps.value += 1

        hp = hp.copy()
        seed = hp['seed']
        # print('seed: {}, type(seed): {}'.format(seed, type(seed)), flush=True)
        np.random.seed(seed)
        torch.manual_seed(seed)
        hp.pop('seed')

        # Prep
        try:
            model = model_class(**hp, device_ids=device_ids) # type: ignore
        except TypeError:
            model = model_class(**hp) # type: ignore
        model._use_lforb = use_lforb
        config, config_dir, img_save_prefix = setup_config_dir(model, train_data.name, train_data.shuffle_seed, exp_dir, config_idx)

        # Save data for easy reproducibility (currently used in mstd computation)
        repro_data = {
            'get_dataset_kwargs': get_dataset_kwargs,
            'model_class': model_class,
            'seed': seed,
            'hp': hp
        }
        with open('{}/{}.pkl'.format(config_dir, CONFIG_REPRO_DATA_FILE), 'wb') as f:
            pickle.dump(repro_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Set what model.acc() should do
        acc_func, acc_func_type = Utils.get_acc_def(DataLoader.is_classification(get_dataset_kwargs['dataset_name']), hp.get('criterion', None))
        model.acc_func = acc_func
        model.acc_func_type = acc_func_type

        model.train(train_data, validn_data, test_data)
        stats = process_model(model, train_data, validn_data, test_data, config_dir, config_idx, seed, compute_auc)

        # Log
        l.acquire()

        finished_hps.value += 1
        logger.log('\n>> [{:.2f}% ({}/{})]:\nRan: (cidx={}): {}\n'.format(
            finished_hps.value * 100 / total_hps, finished_hps.value, total_hps, config_idx, config))
        logger.log('Config dir: {}\n'.format(os.path.abspath(config_dir)))

        logger.log('\ntrain_acc={:.5f}%\n'.format(stats['train_acc']))
        logger.log('validn_acc={:.5f}%\n'.format(stats['validn_acc']))
        logger.log('test_acc={:.5f}%\n'.format(stats['test_acc']))
        if compute_auc:
            logger.log('train_auc={:.5f}\n'.format(stats['train_auc']))
            logger.log('validn_auc={:.5f}\n'.format(stats['validn_auc']))
            logger.log('test_auc={:.5f}\n'.format(stats['test_auc']))

        if isinstance(model, DTExtractablePredictor):
            if not model._is_pure_dt:
                logger.log('dt_train_acc={:.5f}%\n'.format(stats['dt_train_acc']))
                logger.log('dt_validn_acc={:.5f}%\n'.format(stats['dt_validn_acc']))
                logger.log('dt_test_acc={:.5f}%\n'.format(stats['dt_test_acc']))
                if compute_auc:
                    logger.log('dt_train_auc={:.5f}\n'.format(stats['dt_train_auc']))
                    logger.log('dt_validn_auc={:.5f}\n'.format(stats['dt_validn_auc']))
                    logger.log('dt_test_auc={:.5f}\n'.format(stats['dt_test_auc']))

        if isinstance(model, DTExtractablePredictor) and isinstance(model, DGTPredictor):
            if not model._is_pure_dt:
                logger.log('cdt_train_acc={:.5f}%\n'.format(stats['cdt_train_acc']))
                logger.log('cdt_validn_acc={:.5f}%\n'.format(stats['cdt_validn_acc']))
                logger.log('cdt_test_acc={:.5f}%\n'.format(stats['cdt_test_acc']))

        total_time = time.time() - start_time
        per_hp_time = total_time / finished_hps.value
        rem_time = per_hp_time * (total_hps - finished_hps.value)
        logger.log('Time: per hp={}, total={}, rem={}\n'.format(
            td(seconds=per_hp_time), td(seconds=total_time), td(seconds=rem_time)))

        Utils.append_linedict_to_csv(stats, model_search_summary_path)
        l.release()

        if use_lforb:
            assert hp['use_last_model']

            old_logs_dir = model.logs_dir
            config_idx += total_hps
            _, config_dir, _ = setup_config_dir(model, train_data.name, train_data.shuffle_seed, exp_dir, config_idx)
            model.load_best_model(old_logs_dir)

            hp['use_last_model'] = False
            repro_data = {
                'get_dataset_kwargs': get_dataset_kwargs,
                'model_class': model_class,
                'seed': seed,
                'hp': hp
            }
            with open('{}/{}.pkl'.format(config_dir, CONFIG_REPRO_DATA_FILE), 'wb') as f:
                pickle.dump(repro_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            stats = process_model(model, train_data, validn_data, test_data, config_dir, config_idx, seed, compute_auc)

            l.acquire()
            Utils.append_linedict_to_csv(stats, model_search_summary_path)
            l.release()

    logger.close()

def get_best_model(
    log_path: str,
    mss: ModelSearchSet,
    train_data: Dataset,
    validn_data: Dataset,
    test_data: Dataset,
    get_dataset_kwargs: Dict[str, Any],
    exp_dir: str='../out',
    devices_info: List[Tuple[int, int]]=[(-1, 1)], # (device_id, max_proc_cnt), -1 for cpu
    show_hpsearch_stats: bool=True
):
    logger = Logger(log_path)
    if LOG_DATASET_STATS:
        logger.log(train_data.get_stats(title='Train Data'))
        logger.log(validn_data.get_stats(title='Validation Data'))
        logger.log(test_data.get_stats(title='Test Data'))

    # Prepare process-specific device ids, hyperparams and datasets
    proc_device_id: List[int]
    nprocs: int

    # Prep proc_device_id
    proc_device_id = []
    device_ids_helper = {}
    for info in devices_info:
        device_ids_helper[info[0]] = info[1]
    rem = True
    while rem:
        rem = False
        for device_id in device_ids_helper:
            if device_ids_helper[device_id] > 0:
                rem = True
                proc_device_id.append(device_id)
                device_ids_helper[device_id] -= 1

    # Prep proc_hps
    proc_hps = []
    if mss.model_class not in CPU_MODELS:
        total_proc_cnt = len(proc_device_id)
        proc_hp_cnt = np.full(total_proc_cnt, len(mss.hps) // total_proc_cnt)
        proc_hp_cnt[: len(mss.hps) % total_proc_cnt] += 1
        agg = 0
        for cnt in proc_hp_cnt:
            if cnt > 0:
                proc_hps.append(mss.hps[agg: agg + cnt])
                agg += cnt
    else:
        proc_hps = [mss.hps]
    nprocs = len(proc_hps)
    proc_device_id = proc_device_id[: nprocs]

    # Run the processes
    l: 'multiprocessing.synchronize.Lock' = Lock()
    cmlock: 'multiprocessing.synchronize.Lock' = Lock()
    started_hps = Value('i', 0)
    started_hps_lk: 'multiprocessing.synchronize.Lock' = Lock()
    finished_hps = Value('i', 0)
    total_hps = len(mss.hps)
    start_time = time.time()
    assert (train_data.name == validn_data.name) and (validn_data.name == test_data.name) and (test_data.name == get_dataset_kwargs['dataset_name'])
    assert (train_data.shuffle_seed == validn_data.shuffle_seed) and (validn_data.shuffle_seed == test_data.shuffle_seed) and (test_data.shuffle_seed == get_dataset_kwargs['shuffle_seed'])

    model_search_summary_path = '{}/{}/{}-search-summary{}{}.csv'.format(exp_dir, EXP_LOGS_DIR, get_dataset_kwargs['dataset_name'], DESC_SEP, exp_dir.split(DESC_SEP)[-1])
    assert not os.path.exists(model_search_summary_path)

    logger.log(f'\nExperiment dir: {os.path.abspath(exp_dir)}\n')
    logger.log(f'Hyperparam configs: {total_hps}\n')
    logger.log(f'Number of processes: {nprocs}\n')

    args = (
        l, cmlock, started_hps, started_hps_lk, finished_hps, total_hps, start_time, mss.model_class, proc_hps,
        train_data, validn_data, test_data, get_dataset_kwargs, exp_dir, proc_device_id,
        log_path, model_search_summary_path, mss.use_lforb
    )
    if mss.model_class not in CPU_MODELS:
        spawn(get_best_model_aux, args=args, nprocs=nprocs)
    else:
        get_best_model_aux(0, *args)

    if show_hpsearch_stats:
        hpsearch_stats(model_search_summary_path, get_dataset_kwargs['dataset_name'])

    logger.log('\n==================================\n')
    logger.log('Model search summary saved to: {}\n'.format(os.path.abspath(model_search_summary_path)))
    logger.log('==================================\n\n')
    logger.close()
