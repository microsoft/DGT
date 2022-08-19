from typing import Any, Dict, List, Optional, Type, cast

import numpy as np
import torch.nn as nn
import torch.optim as optim

from models.skcart import SkCARTPredictor
from models.sklin import SkLinPredictor
from models.xdgt import DGTPredictor
from models.xlin import LinPredictor
from models.xvw import VWPredictor
from xcommon import LearnablePredictor, Utils
from xdata import DataLoader
from xconstants import BB_ENUM,LR_SCHED_STEPS

class ModelSearchSet():
    def __init__(self, model_class: Type[LearnablePredictor], hps: List[Dict[str, Any]], expand: bool=True, use_lforb: bool=False):
        self.hps: List[Dict[str, Any]]
        if expand:
            self.hps = Utils.get_list_dict_combinations(hps)
        else:
            self.hps = hps

        self.model_class = model_class
        self.search_size = len(self.hps)
        self.use_lforb = use_lforb # use ulm=True run to compute stats for ulm=False

class HpUtils():
    @staticmethod
    def get_entropy_net_hps(dataset_name: str, opts) -> ModelSearchSet:
        n_features = DataLoader.stats[dataset_name]['n_features']
        n_classes = DataLoader.stats[dataset_name]['n_classes']

        use_lforb = False
        if (True in opts.ulm) and (False in opts.ulm):
            opts.ulm = [True]
            use_lforb = True

        hps = []
        for height in opts.height:
            for op in opts.over_param:
                reglist = opts.reglist
                outreglist = []

                if len(op)==0:
                    outreglist.append((0,0))
                else:
                    if opts.use_no_reg: outreglist.append((0,0))
                    if opts.use_l1_reg: outreglist.extend([(i, 0) for i in reglist])
                    if opts.use_l2_reg: outreglist.extend([(0,i) for i in reglist])

                for l1, l2 in (outreglist):
                    for optimizer in opts.optimizer:
                        for lr_scheduler in opts.lr_sched:
                            for bs in opts.batch_sizes:
                                ### Training hyperparameters (defined above since sigslopes etc require epocs defn)
                                epochs = opts.epochs
                                vepochs = 1

                                l1_lambda = [(l1, l1)]
                                l2_lambda = [(l2, l2)]


                                black_box = [opts.black_box]
                                br = opts.br
                                if opts.black_box == BB_ENUM.BB_CLASS:
                                    gamma = [0.02, 0.1, 0.2, 0.3]
                                else:
                                    gamma = [0.05]

                                if optimizer == 'Adam':
                                    curroptimizer = [optim.Adam]
                                    if opts.optimizer_kw == 'default':
                                        optimizer_kwargs = [{'eps':1e-5, 'betas': (0.9,0.999)}]
                                    elif opts.optimizer_kw == 'medium':
                                        optimizer_kwargs = [
                                            {'eps':1e-5, 'betas':(0.8,0.999)},
                                            {'eps':1e-5, 'betas':(0.9,0.999)},
                                        ]
                                    elif opts.optimizer_kw == 'big':
                                        optimizer_kwargs = [
                                            {'amsgrad': True, 'eps':1e-5, 'betas':(0.7,0.999)},
                                            {'amsgrad': True, 'eps':1e-5, 'betas':(0.8,0.999)},
                                            {'amsgrad': True, 'eps':1e-5, 'betas':(0.9,0.999)},
                                            {'eps':1e-5, 'betas':(0.7,0.999)},
                                            {'eps':1e-5, 'betas':(0.8,0.999)},
                                            {'eps':1e-5, 'betas':(0.9,0.999)},
                                        ]
                                    elif opts.optimizer_kw == 'xbig':
                                        optimizer_kwargs = [
                                            {'amsgrad': True},
                                            {'amsgrad': True, 'eps':1e-5, 'betas':(0.7,0.999)},
                                            {'amsgrad': True, 'eps':1e-5, 'betas':(0.8,0.999)},
                                            {'amsgrad': True, 'eps':1e-5, 'betas':(0.9,0.999)},
                                            {},
                                            {'eps':1e-5, 'betas':(0.7,0.999)},
                                            {'eps':1e-5, 'betas':(0.8,0.999)},
                                            {'eps':1e-5, 'betas':(0.9,0.999)},
                                        ]
                                    elif opts.optimizer_kw == 'old':
                                        optimizer_kwargs = [
                                            {'amsgrad': True},
                                            {'amsgrad': False}
                                        ]
                                    elif opts.optimizer_kw == 'singleold':
                                        optimizer_kwargs = [
                                            {'amsgrad': True}
                                        ]
                                    else:
                                        raise ValueError

                                elif optimizer == 'RMS':
                                    curroptimizer = [optim.RMSprop]
                                    if opts.optimizer_kw == 'default':
                                        optimizer_kwargs = [{'eps':1e-5, 'momentum': 0},]
                                    elif opts.optimizer_kw == 'medium':
                                        optimizer_kwargs = [
                                                {'eps':1e-5, 'momentum': 0},
                                                {'eps':1e-5, 'momentum': 0.3},
                                            ]
                                    elif opts.optimizer_kw == 'big':
                                        optimizer_kwargs = [
                                                {'eps':1e-5, 'momentum': 0},
                                                {'eps':1e-5, 'momentum': 0.2},
                                                {'eps':1e-5, 'momentum': 0.3},
                                                {'eps':1e-5, 'momentum': 0.4},
                                                {'eps':1e-5, 'momentum': 0.6},
                                                {'eps':1e-5, 'momentum': 0.8},
                                            ]
                                    elif opts.optimizer_kw == 'xbig':
                                        optimizer_kwargs = [
                                                {},
                                                {'eps':1e-5, 'momentum': 0},
                                                {'eps':1e-5, 'momentum': 0.1},
                                                {'eps':1e-5, 'momentum': 0.2},
                                                {'eps':1e-5, 'momentum': 0.3},
                                                {'eps':1e-5, 'momentum': 0.5},
                                                {'eps':1e-5, 'momentum': 0.7},
                                            ]
                                    elif opts.optimizer_kw == 'old':
                                        optimizer_kwargs = [{'eps':1e-5, 'momentum': 0.3},]
                                    else:
                                        raise ValueError
                                elif optimizer == 'SGD':
                                    curroptimizer = [optim.SGD]
                                    if opts.optimizer_kw == 'default':
                                        optimizer_kwargs = [{}, {'momentum': 0.3}, {'momentum': 0.7}]
                                    elif opts.optimizer_kw == 'old':
                                        optimizer_kwargs = [{'momentum': 0.9}]
                                    else:
                                        raise ValueError
                                else:
                                    raise NotImplementedError

                                ### NOTE -- not using opts.lr_sched_kw currently
                                if lr_scheduler == 'Step':
                                    lr_scheduler_out = [optim.lr_scheduler.StepLR]
                                    if opts.lr_sched_kw == 'default':
                                        lr_scheduler_kwargs = [{'step_size': int(epochs*0.9), 'gamma': 0.1},]
                                    elif opts.lr_sched_kw == 'old':
                                        lr_scheduler_kwargs = [{'step_size': epochs, 'gamma': 1}]
                                    elif opts.lr_sched_kw == 'h1':
                                        lr_scheduler_kwargs = [{'step_size': epochs // 10, 'gamma': 0.5}]
                                    elif opts.lr_sched_kw == 'h2':
                                        lr_scheduler_kwargs = [{'step_size': epochs // 5, 'gamma': 0.5}]
                                    elif opts.lr_sched_kw == 'const':
                                        lr_scheduler_kwargs = [{'step_size': epochs, 'gamma': 1}]
                                    else:
                                        raise ValueError

                                elif lr_scheduler == 'Cosine':
                                    lr_scheduler_out = [optim.lr_scheduler.CosineAnnealingLR]
                                    if opts.lr_sched_kw == 'default':
                                        lr_scheduler_kwargs = [
                                                {'T_max': epochs, 'eta_min': 1e-4},
                                                {'T_max': epochs, 'eta_min': 1e-6},
                                            ]
                                    else:
                                        raise ValueError

                                elif lr_scheduler == 'CosineWarm':
                                    lr_scheduler_out = [optim.lr_scheduler.CosineAnnealingWarmRestarts]
                                    steps_per_epoch = opts.train_size // bs + (0 if opts.train_size%bs==0 else 1)
                                    total_lr_steps = epochs*steps_per_epoch//LR_SCHED_STEPS+1
                                    if opts.lr_sched_kw == 'default':
                                        lr_scheduler_kwargs = [
                                                {'T_0': total_lr_steps//2+2, 'eta_min': 1e-4},
                                                {'T_0': total_lr_steps//4+2, 'eta_min': 1e-4},
                                                {'T_0': total_lr_steps//3+2, 'T_mult': 2, 'eta_min': 1e-4},
                                            ][1:2]
                                    elif opts.lr_sched_kw == 'ablation':
                                        lr_scheduler_kwargs = [
                                                {'T_0': total_lr_steps//4+2, 'eta_min': 1e-4},
                                                {'T_0': total_lr_steps//4+2, 'eta_min': 1e-2},
                                            ]
                                    elif opts.lr_sched_kw == 'old':
                                        lr_scheduler_kwargs = [
                                                {'T_0': total_lr_steps//2+2, 'eta_min': 1e-4},
                                                {'T_0': total_lr_steps//3+2, 'eta_min': 1e-4},
                                                {'T_0': total_lr_steps//4+2, 'eta_min': 1e-4},
                                                {'T_0': total_lr_steps//5+2, 'eta_min': 1e-4},
                                            ]
                                    elif opts.lr_sched_kw == 'h1':
                                        lr_scheduler_kwargs = [
                                                {'T_0': total_lr_steps//2+2, 'eta_min': 1e-4},
                                                {'T_0': total_lr_steps//3+2, 'eta_min': 1e-4},
                                                {'T_0': total_lr_steps//4+2, 'eta_min': 1e-4},
                                            ]
                                    else:
                                        raise ValueError
                                else:
                                    raise NotImplementedError



                                #####################
                                # Architecture params
                                learn_and_bias_for = opts.lab
                                eps = [opts.eps]

                                if opts.and_act == 'softmax':
                                    and_act_fn = [nn.Softmax(dim=-1)]
                                elif opts.and_act == 'relu':
                                    and_act_fn = [nn.ReLU()]
                                else:
                                    raise NotImplementedError

                                # Constraining parameters
                                sigslopes: List[Optional[List[float]]] = []
                                sigslope_routine = opts.sigslope_routine

                                if sigslope_routine == 'constant':
                                    sigslopes.append([1] * epochs)
                                elif sigslope_routine == 'multiconstant':
                                    sigslopes.append([1] * epochs)
                                    sigslopes.append([.1] * epochs)
                                    sigslopes.append([10] * epochs)
                                    sigslopes.append([100] * epochs)
                                elif sigslope_routine == 'increase':
                                    sigslopes.append(np.linspace(1, 50, epochs))
                                    sigslopes.append(np.linspace(1, 200, epochs))
                                elif sigslope_routine == 'midincrease':
                                    sigslopes.append(np.linspace(1, 50, epochs))
                                    sigslopes.append(np.linspace(1, 100, epochs))
                                    sigslopes.append(np.linspace(1, 300, epochs))
                                    sigslopes.append(np.linspace(1, 1000, epochs))
                                elif sigslope_routine == 'largeincrease':
                                    sigslopes.append(np.linspace(1, 10, epochs))
                                    sigslopes.append(np.linspace(1, 50, epochs))
                                    sigslopes.append(np.linspace(1, 100, epochs))
                                    sigslopes.append(np.linspace(1, 300, epochs))
                                    sigslopes.append(np.linspace(1, 1000, epochs))
                                    sigslopes.append(np.linspace(1, 3000, epochs))


                                softslopes = []
                                softslope_routine = opts.softslope_routine

                                if softslope_routine == 'constant':
                                    softslopes.append([1] * epochs)
                                elif softslope_routine == 'multiconstant':
                                    softslopes.append([1] * epochs)
                                    softslopes.append([.1] * epochs)
                                    softslopes.append([10] * epochs)
                                    softslopes.append([100] * epochs)
                                elif softslope_routine == 'increase':
                                    softslopes.append(np.linspace(1, 10, epochs))
                                    softslopes.append(np.linspace(1, 50, epochs))
                                    softslopes.append(np.linspace(1, 100, epochs))
                                    softslopes.append(np.linspace(1, 300, epochs))
                                elif softslope_routine == 'largeincrease':
                                    softslopes.append(np.linspace(1, 5, epochs))
                                    softslopes.append(np.linspace(1, 10, epochs))
                                    softslopes.append(np.linspace(1, 50, epochs))
                                    softslopes.append(np.linspace(1, 100, epochs))
                                    softslopes.append(np.linspace(1, 300, epochs))
                                    softslopes.append(np.linspace(1, 1000, epochs))
                                elif softslope_routine == 'logincrease':
                                    softslopes.append(np.logspace(-3, 1, epochs))
                                    softslopes.append(np.logspace(-3, 1.75, epochs))
                                    softslopes.append(np.logspace(-3, 2, epochs))
                                elif softslope_routine == 'largelogincrease':
                                    softslopes.append(np.logspace(-4, 1, epochs))
                                    softslopes.append(np.logspace(-3, 1, epochs))
                                    softslopes.append(np.logspace(-2, 1, epochs))
                                    softslopes.append(np.logspace(-4, 1.75, epochs))
                                    softslopes.append(np.logspace(-3, 1.75, epochs))
                                    softslopes.append(np.logspace(-2, 1.75, epochs))
                                    softslopes.append(np.logspace(-4, 2, epochs))
                                    softslopes.append(np.logspace(-3, 2, epochs))
                                    softslopes.append(np.logspace(-2, 2, epochs))

                                thresh: List[List[float]] = []
                                if opts.lab > 0:
                                    thresh.append([-height])
                                else:
                                    if opts.thresholds == 'constant':
                                        thresh.append([height, height])
                                    elif opts.thresholds == 'general':
                                        thresh.append([0, -height])
                                        thresh.append([height, -height])

                                sigquant = opts.sigquant
                                softquant = opts.softquant
                                sigrandbin = [np.linspace(-1,-1,epochs)]
                                softrandbin = [np.linspace(-1,-1,epochs)]
                                if sigquant[0] is not None and 'Noise' in sigquant[0]:
                                    sigrandbin = [[0.6, 0.6],[0.8, 0.8],[0,1],[0.2,1],[0.4,1]]
                                    sigrandbin = [np.linspace(*x, epochs) for x in sigrandbin]
                                if softquant[0] is not None and 'Noise' in softquant[0]:
                                    softrandbin = [[0.6, 0.6],[0.8, 0.8],[0,1],[0.2,1],[0.4,1]]
                                    softrandbin = [np.linspace(*x, epochs) for x in softrandbin]

                                if opts.criterion is not None:
                                    if opts.criterion == 'huber':
                                        criterion = [Utils.huber_loss]
                                        ## can be extended for getting multiple losses from cmd line
                                    else:
                                        raise NotImplementedError
                                else:
                                    if n_classes > 0:
                                        criterion = [nn.CrossEntropyLoss(reduction='none')]
                                    else:
                                        criterion = [nn.MSELoss(reduction='none')]

                                hps.append(cast(Dict[str, List[Any]], {
                                    'n_features': [n_features],
                                    'n_classes': [n_classes],
                                    'height': [height],
                                    'over_param': [op],
                                    'learn_and_bias_for': [learn_and_bias_for],
                                    'sigslopes': sigslopes,
                                    'softslopes' : softslopes,
                                    'thresholds': thresh,
                                    'eps': eps,
                                    'and_act_fn': and_act_fn,
                                    'sigquant' : sigquant,
                                    'sigrandbin' : sigrandbin,
                                    'softquant' : softquant,
                                    'softrandbin' : softrandbin,
                                    'batch_norm': opts.batchnorm,
                                    'pred_reg_only': opts.pred_reg,

                                    'epochs': [epochs],
                                    'vepochs': [vepochs],
                                    'criterion': criterion,
                                    'batch_size': [bs],
                                    'l1_lambda': l1_lambda,
                                    'l2_lambda': l2_lambda,
                                    'optimizer': curroptimizer,
                                    'optimizer_kwargs': optimizer_kwargs,
                                    'lr1': opts.lr1,
                                    'lr2': opts.lr2,
                                    'lr_scheduler': lr_scheduler_out,
                                    'lr_scheduler_kwargs': lr_scheduler_kwargs,
                                    'grad_clip': opts.grad_clips,

                                    'black_box': black_box,
                                    'br': br,
                                    'gamma' : gamma,
                                    'use_last_model': opts.ulm,

                                    # Doesn't reach the model's init, used by driver code
                                    'seed': opts.seed,
                                }))

        return ModelSearchSet(DGTPredictor, hps, use_lforb=use_lforb)

    @staticmethod
    def get_lin_hps(dataset_name: str, opts) -> ModelSearchSet:
        n_features = DataLoader.stats[dataset_name]['n_features']
        n_classes = DataLoader.stats[dataset_name]['n_classes']

        hps = []
        epochs = opts.epochs

        if opts.criterion is not None:
            if opts.criterion == 'huber':
                criterion = [Utils.huber_loss]
                ## can be extended for getting multiple losses from cmd line
            else:
                raise NotImplementedError
        else:
            criterion = [nn.MSELoss(reduction='none')]

        for optimizer in opts.optimizer:
            for lr_scheduler in opts.lr_sched:
                for bs in opts.batch_sizes:
                    if optimizer == 'Adam':
                        curroptimizer = [optim.Adam]
                        if opts.optimizer_kw == 'default':
                            optimizer_kwargs = [{'eps':1e-5, 'betas': (0.9,0.999)}]
                        else:
                            raise ValueError

                    elif optimizer == 'RMS':
                        curroptimizer = [optim.RMSprop]
                        if opts.optimizer_kw == 'default':
                            optimizer_kwargs = [
                                    {'eps':1e-5, 'momentum': 0.8,},
                                    {'eps':1e-5, 'momentum': 0.4,},
                                    {'eps':1e-5, 'momentum': 0.,},
                                ]
                        else:
                            raise ValueError
                    elif optimizer == 'SGD':
                        curroptimizer = [optim.SGD]
                        if opts.optimizer_kw == 'default':
                            optimizer_kwargs = [
                                    {'momentum': 0.8, 'nesterov': False},
                                    {'momentum': 0.4, 'nesterov': False},
                                    {'momentum': 0., 'nesterov': False},
                                ]
                        else:
                            raise ValueError
                    else:
                        raise NotImplementedError

                    hps.append(cast(Dict[str, List[Any]], {
                        'n_features': [n_features],

                        'epochs': [epochs],
                        'l1_lambda': [0],
                        'l2_lambda': opts.reglist+[0],
                        'lr': opts.lr1,
                        'batch_size': [bs],

                        'criterion': criterion,
                        'optimizer': curroptimizer,
                        'optimizer_kwargs': optimizer_kwargs,
                        'lr_scheduler': [optim.lr_scheduler.StepLR],
                        'lr_scheduler_kwargs': [{'step_size': epochs, 'gamma': 1}],

                        'black_box': [opts.black_box],
                        'br': opts.br,

                        'seed': opts.seed,
                    }))

        return ModelSearchSet(LinPredictor, hps)

    @staticmethod
    def get_skcart_hps(dataset_name: str, opts) -> ModelSearchSet:
        n_features = DataLoader.stats[dataset_name]['n_features']
        n_classes = DataLoader.stats[dataset_name]['n_classes']

        vals = (
            list(range(2, 11)) +
            list(range(11, 31, 2)) +
            list(range(31, 61, 3)) +
            list(range(61, 101, 4)) +
            list(range(101, 151, 5))
        )
        vals = (
            list(range(2, 21, 2)) +
            list(range(21, 101, 4)) +
            list(range(101,151,5))
        )
        # vals = (
        #     list(range(2, 50, 3)) +
        #     list(np.logspace(-10, -1, base=3, num=15))
        # )

        hps = []
        hps.append(cast(Dict[str, List[Any]], {
            'n_features': [n_features],
            'n_classes': [n_classes],

            'max_depth': [2, 4, 6, 8, 10],
            'min_samples_leaf': [1] + vals,
            #'min_samples_split': vals,

            'seed': opts.seed,
        }))

        return ModelSearchSet(SkCARTPredictor, hps)

    @staticmethod
    def get_sklin_hps(dataset_name: str, opts) -> ModelSearchSet:
        n_features = DataLoader.stats[dataset_name]['n_features']
        n_classes = DataLoader.stats[dataset_name]['n_classes']

        hps = []
        hps.append(cast(Dict[str, List[Any]], {
            'n_features': [n_features],
            'n_classes': [n_classes],

            'l2_lambda': (
                [0, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4] +
                #[         2e-4, 2e-3, 2e-2, 2e-1, 2, 2e1, 2e2, 2e3, 2e4] +
                [         5e-4, 5e-3, 5e-2, 5e-1, 5, 5e1, 5e2, 5e3, 5e4]
            ),
            # 'l2_lambda': [0],

            'seed': opts.seed,
        }))

        return ModelSearchSet(SkLinPredictor, hps)

    @staticmethod
    def get_vw_hps(dataset_name: str, opts) -> ModelSearchSet:
        n_features = DataLoader.stats[dataset_name]['n_features']
        n_classes = DataLoader.stats[dataset_name]['n_classes']

        l1_lambda, l2_lambda = [0], [0]
        if opts.use_l1_reg:
            l1_lambda = opts.reglist
        if opts.use_l2_reg:
            l2_lambda = opts.reglist

        hps = []
        hps.append(cast(Dict[str, List[Any]], {
            'n_features': [n_features],
            'n_classes': [n_classes],

            'epochs': [opts.epochs],
            'lr': opts.lr1,
            'epsilon': opts.epsilon,
            'l1_lambda': l1_lambda,
            'l2_lambda': l2_lambda,
            'num_tlogs': [opts.num_tlogs],

            'seed': opts.seed,
        }))

        return ModelSearchSet(VWPredictor, hps)

    @staticmethod
    def get_model_search_set(model_class: Type[LearnablePredictor], dataset_name: str, opts) -> ModelSearchSet:
        if model_class == DGTPredictor:
            return HpUtils.get_entropy_net_hps(dataset_name, opts)
        if model_class == LinPredictor:
            return HpUtils.get_lin_hps(dataset_name, opts)
        if model_class == SkCARTPredictor:
            return HpUtils.get_skcart_hps(dataset_name, opts)
        if model_class == SkLinPredictor:
            return HpUtils.get_sklin_hps(dataset_name, opts)
        if model_class == VWPredictor:
            return HpUtils.get_vw_hps(dataset_name, opts)

        raise ValueError('Unexpected model_class: {}'.format(model_class))
