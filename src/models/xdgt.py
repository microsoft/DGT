import itertools
import time
from collections import OrderedDict
from datetime import timedelta as td
from typing import (Any, Callable, Dict, Iterator, List, Optional, Tuple, Type,
                    Union, cast)

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.tree import DecisionTreeRegressor
from xcommon import (Batcher, Dataset, DTExtractablePredictor, DTPredictor,
                     LearnablePredictor, NTArray, Utils)
from xconstants import (APPLY_NORM_TO_PREDPOOL, BB_ENUM,
                        COMPARE_LABEL_DIST_EPOCH_EVERY, DEBUG_INPUT_DATA,
                        DEBUG_LOG_NET_BATCH_EVERY, DEBUG_LOG_NET_BATCH_FIRST,
                        DEBUG_LOG_NET_EPOCH_EVERY, DEBUG_Y,
                        DENORMALIZE_RESULTS, DENSE_PROGRESS_LOG_EPOCH_EVERY,
                        PROGRESS_PLOTS, FPTYPE, INIT_MODEL_WITH_CART,
                        LOG_INIT_FIN_WTS, LOG_PRED_TARG_DIFF, PROFILE_TIME,
                        SAT_INFO_DIST_DISC, SAT_INFO_EPOCH_EVERY,
                        SAT_INFO_NODE_MAX, SAVE_PLOTS, ZERO_INIT_PRED_BIAS,
                        AccFuncType, LR_SCHED_STEPS, DENSE_LOG_BB1P_COUNT, DEBUG_PREDICATES)

from models.xdgthelper import (Binarizer1, Binarizer2, ENUtils, LSQLayer,
                              NoiseScaleBinarizer1, NoiseSoftSparser1,
                              ScaleBinarizer1, Sparser1, XLinear)

"""
Args:
- learnable_and_bias: If True, bias in AND layer is learnt otherwise fixed to -counts[1]+eps
    and threshold*height is added later but before activation
- learnable_or_bias: Bias in the OR layer is same for all neurons by default but if this is set to True,
    the bias is learnt otherwise it is set to 0 (for all neurons).

Notes:
    In the AND layer for every neuron this is what happens:
    - When learnable_and_bias=False
        - ReLU(w.x - counts[1] + eps + threshold*height)
        - w is also fixed to {-1, 0, 1}
        - -counts[1] + eps is the fixed bias in XLinear, threshold*height is added in forward()
    - When learnable_and_bias=True
        - ReLU(w.x + b + height*threshold)
        - w is fixed, b is learnt
"""
class DGT(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        height: int,
        learnable_and_bias: bool,
        learnable_or_bias: bool, ##### CHECK CHECK
        sigquant: Any,
        softquant: Any,
        and_act_fn: Any,
        batch_norm: bool,
        over_param: list,
        fp_type: torch.dtype
    ):
        super().__init__()

        self._learnable_and_bias = learnable_and_bias
        assert learnable_and_bias
        self._height = height
        self._sigquant = sigquant
        self._softquant = softquant
        self._and_act_fn = and_act_fn
        self._batch_norm = batch_norm
        self._fp_type = fp_type

        self._over_param = over_param

        int_nodes = 2 ** height - 1
        leaf_nodes = 2 ** height

        ########### L1 ##########


        if len(self._over_param)==0:
            self._predicate_l = nn.Linear(in_dim, int_nodes)
            if ZERO_INIT_PRED_BIAS:
                with torch.no_grad(): nn.init.zeros_(self._predicate_l.bias)
            self._predicate_l = nn.Sequential(self._predicate_l)
        else:
            self._predicate_l = []
            mid_nodes = [int(x*int_nodes) for x in self._over_param]
            self._predicate_l = [nn.Linear(a,b) for a,b in zip([in_dim]+mid_nodes, mid_nodes+[int_nodes])]
            if ZERO_INIT_PRED_BIAS:
                with torch.no_grad():
                    [nn.init.zeros_(x.bias) for x in self._predicate_l if isinstance(x, nn.Linear)]
            self._predicate_l = nn.Sequential(*self._predicate_l)

        if batch_norm:
            self._predicate_bn = nn.BatchNorm1d(int_nodes)

        if sigquant == 'LSQ':
            self._predicate_lsq = LSQLayer()

        ########### L2 ##########
        weight, fixed_bias = DGT._get_and_layer_params(height, self._fp_type)
        # fixed_bias is now 0 not -h
        self._and_l = XLinear(int_nodes, leaf_nodes, weight=weight, bias=None if learnable_and_bias else fixed_bias, same=False)

        #if self._and_layer == 'multiply':
        #   self.weight = nn.Parameter(weight[None], requires_grad=False)
        #   self.ONE_TENSOR = nn.Parameter(torch.Tensor([1.])[0].to(self._fp_type), requires_grad=False)

        ########### L3 ##########
        self._or_l = XLinear(leaf_nodes, out_dim, bias=None if learnable_or_bias else torch.zeros((out_dim,)), same=True)

    def get_parameters_set(self, set_idx: int) -> Iterator[nn.Parameter]:
        if set_idx == 1:
            vals = [self._predicate_l.parameters()]

            # vals.append(self._and_l.parameters())
            if self._sigquant == 'LSQ':
                vals.append(self._predicate_lsq.parameters())
            if self._batch_norm:
                vals.append(self._predicate_bn.parameters())

            return itertools.chain(*vals)

        elif set_idx == 2:
            return self._or_l.parameters()

        else:
            raise ValueError(f'{set_idx} must be in [1, 2]')

    def forward(
        self,
        x: torch.Tensor,
        sigslope: torch.Tensor,
        softslope : torch.Tensor,
        addl_bias: torch.Tensor,
        sigrandbinprob=None,
        softrandbinprob=None,
        log_f: Optional[Callable]=None,
        epoch=0,
        batch=0
    ) -> torch.Tensor: # type: ignore

        assert sigslope.shape == (2 ** self._height - 1,), '{} != {}'.format(sigslope.shape, (2 ** self._height - 1,))
        assert addl_bias.shape == (2 ** self._height,), '{} != {}'.format(addl_bias.shape, (2 ** self._height,))

        ########### L1 ##########
        pred_z = self._predicate_l(x)


        if self._batch_norm:
            pred_z = self._predicate_bn(pred_z)

        if self._sigquant is None:
            pred_a = torch.sigmoid(sigslope * pred_z) ; fac = 1
        else:
            if self._sigquant == 'NoiseScaleBinarizer1':
                pred_a, fac = NoiseScaleBinarizer1.apply(pred_z, sigrandbinprob) ; fac = fac.detach()
            elif self._sigquant == 'ScaleBinarizer1':
                pred_a, fac = ScaleBinarizer1.apply(pred_z) ; fac = fac.detach()
            elif self._sigquant == 'LSQ':
                pred_a = self._predicate_lsq(pred_z)
                fac = self._predicate_lsq.stepsize.detach()
            elif self._sigquant == 'LogisticBinarizer1':
                pred_a = torch.sigmoid(sigslope * pred_z)
                pred_a = 2*pred_a - 1
                pred_a = Binarizer1.apply(pred_a)
                fac = 1
            elif self._sigquant == 'TanhBinarizer1':
                pred_a = torch.tanh(sigslope * pred_z)
                pred_a = Binarizer1.apply(pred_a)
                fac = 1
            else:
                pred_a = globals()[self._sigquant].apply(pred_z) ; fac = 1
        pred_a = 2*pred_a - fac

        ########### L2 ##########
        and_z_a = self._and_l(pred_a)

        #if self._and_layer == 'multiply':
        #    pred_a = pred_a[:,None].expand(pred_a.shape[0], self.weight.shape[1], pred_a.shape[1])
        #    weight = self.weight.expand_as(pred_a)
        #    leave_signs = (pred_a*weight >= 0).all(-1)
        #    and_z_a = torch.exp(torch.log(torch.where(weight==0, self.ONE_TENSOR, torch.abs(pred_a))).sum(-1)) * leave_signs

        if isinstance(self._and_act_fn, nn.ReLU):
            and_z = and_z_a + addl_bias
            and_a = self._and_act_fn(and_z) / fac
        elif isinstance(self._and_act_fn, nn.Softmax):
            if self._softquant is None:
                and_a = self._and_act_fn(and_z_a * softslope)
            else:
                if self._softquant == 'SoftSparser1':
                    and_a = self._and_act_fn(and_z_a)
                    and_a = Sparser1.apply(and_a)
                elif self._softquant == 'NoiseSoftSparser1':
                    and_a = self._and_act_fn(and_z_a)
                    and_a = NoiseSoftSparser1.apply(and_a, softrandbinprob)
                else:
                    and_a = globals()[self._softquant].apply(and_z_a)

        ########### L3 ##########
        or_z = self._or_l(and_a) # and expect CrossEntropyLoss (== log(softmax(x)))

        # Log stats
        if log_f is not None:
            torch.set_printoptions(threshold=0, precision=8, edgeitems=10)
            log_f('\n|\n|\nX: \n{}\n|\n|\n'.format(x), stdout=False)

            log_f('PRED z ({}, {}): \n{}\n|\n|\n'.format(epoch, batch, pred_z), stdout=False)
            log_f('PRED a ({}, {}): \n{}\n|\n|\n'.format(epoch, batch, pred_a), stdout=False)

            log_f('AND addl_bias ({}, {}): \n{}\n|\n|\n'.format(epoch, batch, addl_bias), stdout=False)
            log_f('AND a ({}, {}): \n{}\n|\n|\n'.format(epoch, batch, and_a), stdout=False)
            log_f('AND a ({}, {}) (nonzero vals): \n{}\n|\n|\n'.format(epoch, batch, and_a[and_a > 0]), stdout=False)

            torch.set_printoptions(threshold=1025, precision=8)
            msg = 'For each input sample (from the batch), how many leaf nodes output>0.\nTowards end of training, should be 1 everywhere.\nThrough training, should ideally decrease gradually everywhere.'
            log_f('AND a ({}, {})\n{}: \n{}\n|\n|\n'.format(epoch, batch, msg, (and_a > 0).sum(dim=1)), stdout=False)
            msg = 'For each leaf node, how many input samples (from the batch) output>0.\nSays how many samples reached each leaf.\nDuring early stages, same sample could reach multiple leaves.\nHelps detect skewness in predictions.'
            log_f('AND a ({}, {})\n{}: \n{}\n|\n|\n'.format(epoch, batch, msg, (and_a > 0).sum(dim=0)), stdout=False)

            torch.set_printoptions(threshold=0, precision=8, edgeitems=10)
            log_f('OR z ({}, {}): \n{}\n'.format(epoch, batch, or_z), stdout=False)
            torch.set_printoptions()

        return or_z

    @staticmethod
    def _get_and_layer_params(height: int, fp_type) -> Tuple[torch.Tensor, torch.Tensor]:
        int_nodes = 2 ** height - 1
        leaf_nodes = 2 ** height

        weight = np.zeros((leaf_nodes, int_nodes))

        # Fill in the weight matrix level by level
        # i represents the level of nodes which we are handling at a given iteration
        for i in range(height):
            # Number of nodes in this level
            num_nodes = 2 ** i
            # Start number of node in this level
            start = 2 ** i - 1

            # Iterate through all nodes at this level
            for j in range(start, start + num_nodes):
                row_begin = (leaf_nodes // num_nodes) * (j - start)
                row_mid = row_begin + (leaf_nodes // (2 * num_nodes))
                row_end = row_begin + (leaf_nodes // num_nodes)

                weight[row_begin: row_mid, j] = 1
                weight[row_mid: row_end, j] = -1

        fixed_bias = torch.zeros(size=(2 ** height,), dtype=fp_type)

        return torch.from_numpy(weight).to(fp_type), fixed_bias

class DGTPredictor(DTExtractablePredictor, LearnablePredictor[torch.Tensor]):
    """
    Args:
        device_ids: None implies CPU
        min_sigslope_mult: Applicable only when sigslopes is None
        max_sigslope_mult: Applicable only when sigslopes is None
        criterion:
            If classification:
                Accepts (ypred=torch.tensor(n_samples, n_classes), ytarg=torch.tensor(n_samples,))
                Returns torch.tensor(n_samples,)
            If regression:
                Accepts (ypred=torch.tensor(n_samples,), ytarg=torch.tensor(n_samples,))
                Returns torch.tensor(n_samples,)
        br: Valid only when black_box is True
    """
    def __init__(
        self,
        n_features: int=2,
        n_classes: int=2,
        height: int=4,

        # Number of epochs for which AND bias must be learnt
        learn_and_bias_for: int=0,

        # Usage in one of these forms:
        # shape (epochs,): sigslope for each epoch (sigslope for every node in an epoch will be the same)
        # shape (epochs, int_nodes): sigslope for every node in each epoch specified
        sigslopes: Optional[Union[List[float], List[List[float]], np.ndarray]]=None,
        softslopes: Optional[Union[List[float], List[List[float]], np.ndarray]]=None,
        thresholds: List[float]=[0.5],

        eps: float=0.5,
        and_act_fn: Any=nn.ReLU,
        device_ids: Optional[List[int]]=None,

        # The total number of epochs, including those where AND bias is learnt
        epochs: int=100,
        vepochs: int=1,

        # Two entries for during labf and post labf
        l1_lambda: Tuple[float, float]=(0, 0),
        l2_lambda: Tuple[float, float]=(0, 0),

        lr1: float=0.01,
        lr2: float=0.01,
        batch_size: int=100,

        criterion: Callable[[NTArray, NTArray], Union[torch.Tensor]]=nn.CrossEntropyLoss(reduction='none'),
        optimizer: Type[optim.Optimizer]=optim.SGD, # type: ignore
        optimizer_kwargs: Dict[str, Any]={},
        lr_scheduler: Type[Any]=optim.lr_scheduler.StepLR,
        lr_scheduler_kwargs: Dict[str, Any]={'step_size': 1, 'gamma': 1},

        black_box: Optional[str]=None,
        br: float=0.5,
        gamma: float=0.05,
        use_last_model: bool=True, # TODO: Implement for non-bareEN case
        sigquant = None,
        softquant = None,
        sigrandbin = None,
        softrandbin = None,
        batch_norm: bool=False,
        grad_clip: float=None,
        pred_reg_only: bool=False,
        over_param: float = 1.,
    ):
        super().__init__()

        self._n_features = n_features
        self._n_classes = n_classes
        self._height = height
        self._learn_and_bias_for = learn_and_bias_for
        self._sigslopes = sigslopes
        self._softslopes = softslopes
        self._thresholds = thresholds
        self._eps = eps
        self._and_act_fn = and_act_fn
        self._device_ids = device_ids

        self._epochs = epochs
        self._vepochs = vepochs
        self._l1_lambda = l1_lambda
        self._l2_lambda = l2_lambda
        self._xl1_lambda = l1_lambda[0]
        self._xl2_lambda = l2_lambda[0]
        self._lr1 = lr1
        self._lr2 = lr2
        self._batch_size = batch_size
        self._pred_reg_only = pred_reg_only

        self._criterion: Callable[[NTArray, NTArray], Union[torch.Tensor]] = criterion
        self._optimizer = optimizer
        self._optimizer_kwargs = optimizer_kwargs
        self._lr_scheduler = lr_scheduler
        self._lr_scheduler_kwargs = lr_scheduler_kwargs

        self._black_box = black_box
        self.one_point_black_box = 'smart1p' in str(self._black_box)  or 'class' in str(self._black_box)
        self._br = br
        self._gamma = gamma

        self._sigquant = sigquant
        self._sigrandbinprob = sigrandbin

        self._softquant = softquant
        self._softrandbinprob = softrandbin

        self._is_pure_dt = (self._sigquant is not None) and (self._softquant is not None)

        self._over_param = over_param
        self._batch_norm = batch_norm
        self._grad_clip = grad_clip

        self._validate_args()

        # Prep for train
        if self._device_ids is not None:
            self._device0 = torch.device('cuda:{}'.format(self._device_ids[0]))
        self._is_classification = n_classes > 0
        self._fp_type = FPTYPE
        self._l2_reg_avail = self._is_l2_reg_avail()
        self._use_last_model = use_last_model
        self._best_val_loss = None

        self._model: DGT = self._create_model().to(self._fp_type)

        if self._learn_and_bias_for == 0:
            nn.init.zeros_(self._model._and_l.bias)
        self._train_invoked: bool = False
        if APPLY_NORM_TO_PREDPOOL:
            self._apply_norm_to_pred()

        # Initialize logging stores
        if DEBUG_PREDICATES:
            len_preds = len(self._model._predicate_l._modules.values())
            colnames = ['epoch'] + sum([[f'l{i}_mean',f'l{i}_std',f'l{i}_median',f'l{i}_max',f'l{i}_min'] for i in range(len_preds)], [])
            self._weight_info_df = pd.DataFrame(columns=(colnames))

        self._dense_progress_df = pd.DataFrame(columns=([
            'epoch',
            'train_acc', 'validn_acc',
            'dt_train_acc','dt_validn_acc',
            'cdt_train_acc','cdt_validn_acc',
            'train_loss_wreg','train_loss',
            'validn_loss_wreg','validn_loss',
            'lr','xsigslope','xthreshold']
            + (['stepsize'] if self._sigquant == 'LSQ' else [])),
            dtype=np.float64)
        self._sat_info = self._init_sat_info()

        # Setup sigslopes
        sigslopes = np.array(self._sigslopes)
        softslopes = np.array(self._softslopes)
        assert len(sigslopes.shape) in [1, 2]
        if len(sigslopes.shape) == 1:
            # sigslope given for each epoch
            assert sigslopes.shape == (self._epochs,)
            # given sigslope for each epoch, duplicate along pred nodes dimension
            self._xsigslopes = np.repeat(sigslopes.reshape((-1, 1)), 2 ** self._height - 1, 1)
            self._xsoftslopes = np.repeat(softslopes.reshape((-1, 1)), 2 ** self._height, 1)
        else:
            # sigslope given for every node for each epoch
            assert sigslopes.shape == (self._epochs, 2 ** self._height - 1)
            self._xsigslopes = sigslopes
            self._xsoftslopes = softslopes
        self._xsigslopes = torch.from_numpy(self._xsigslopes).to(dtype=self._fp_type, device=self._device0)
        self._xsoftslopes = torch.from_numpy(self._xsoftslopes).to(dtype=self._fp_type, device=self._device0)

        # Setup thresholds
        # xthreshold contains both threshold and eps. It is the bias (in addition to -h) for the AND layer
        # Important for this to be 0
        self._xthresholds = torch.zeros((self._epochs, 2 ** self._height), dtype=self._fp_type, device=self._device0)

        self._curr_xsigslope: torch.Tensor
        self._curr_xthreshold: torch.Tensor

        if PROFILE_TIME:
            self._exec_times: Dict[str, List[float]] = OrderedDict(
                _update_weights=[],
                _log_weights=[],
                _upd_dense_progress=[],
                _get_best_labels=[],
                _get_best_labelsin=[]
            )

    def get_hyperparams(self) -> Dict[str, Any]:
        return OrderedDict(
            n_features=self._n_features,
            n_classes=self._n_classes,
            height=self._height,
            learn_and_bias_for=self._learn_and_bias_for,
            sigslopes=[self._xsigslopes[0][0].item(), self._xsigslopes[-1][0].item()],
            softslopes=[self._xsoftslopes[0][0].item(), self._xsoftslopes[-1][0].item()],
            sigquant=self._sigquant,
            softquant=self._softquant,
            sigrandbinprob=[self._sigrandbinprob[0],self._sigrandbinprob[-1]],
            softrandbinprob=[self._softrandbinprob[0],self._softrandbinprob[-1]],
            thresholds=self._thresholds,
            eps=self._eps,
            and_act_fn=self._and_act_fn,
            batch_norm=self._batch_norm,
            device_ids=self._device_ids,

            epochs=self._epochs,
            vepochs=self._vepochs,
            lr1=self._lr1,
            lr2=self._lr2,
            l1_lambda=self._l1_lambda,
            l2_lambda=self._l2_lambda,
            pred_reg_only=self._pred_reg_only,
            batch_size=self._batch_size,
            grad_clip=self._grad_clip,

            criterion=self._criterion,
            optimizer=self._optimizer.__name__, # type: ignore
            optimizer_kwargs=self._optimizer_kwargs,
            lr_scheduler=self._lr_scheduler.__name__,
            lr_scheduler_kwargs=self._lr_scheduler_kwargs,
            black_box=self._black_box,
            br=self._br,
            gamma=self._gamma,
            use_last_model=self._use_last_model,
            over_param=','.join([str(x) for x in self._over_param]),
        )

    """
    Args:
        train_data: train_data['y'] is expected to have shape (n_samples,)
        validn_data: validn_data['y'] is expected to have shape (n_samples,)
    """
    def train(self, train_data: Dataset[torch.Tensor], validn_data: Dataset[torch.Tensor], test_data: Dataset[torch.Tensor]):
        assert not self._train_invoked, 'train() already invoked once'
        shuffle_seed = np.random.randint(10)
        if INIT_MODEL_WITH_CART:
            self._init_model_with_cart(train_data, validn_data, test_data, random_seed=shuffle_seed)

        if self._batch_norm:
            self._model.train()

        self._train_invoked = True
        train_data = train_data.to_type(self._fp_type, 'x' if self._is_classification else 'xy')
        validn_data = validn_data.to_type(self._fp_type, 'x' if self._is_classification else 'xy')
        test_data = test_data.to_type(self._fp_type, 'x' if self._is_classification else 'xy')

        # Move the model and data to GPU if needed
        self._move_model()
        train_data = self._move_data(train_data)
        validn_data = self._move_data(validn_data)
        test_data = self._move_data(test_data)

        self._set_sigslope_and_threshold(0)

        # Plot initial decision surface
        if SAVE_PLOTS:
            try:
                save_path = '{}/model-decisions-pre-train.png'.format(self.plots_dir)
                self.visualize_decisions(cast(torch.Tensor, train_data['x']), save_path=save_path)
            except AssertionError as e:
                self.log_f(str(e))

        if LOG_INIT_FIN_WTS:
            self.log_f('\n' + Utils.get_padded_text('Weights after model creation: Begin', left_padding='>', right_padding='<') + '\n', stdout=False)
            self._log_weights(-1, -1, grad=True, msg='before training')
            self.log_f('\n' + Utils.get_padded_text('Weights after model creation: End', left_padding='>', right_padding='<') + '\n', stdout=False)

        # Initialize optimizer
        optimizer = self._optimizer(
            [{'params': self._model.get_parameters_set(set_idx=1), 'lr':self._lr1,},
            {'params': self._model.get_parameters_set(set_idx=2), 'lr':self._lr1 if self._lr2 is None else self._lr2,}],
            **self._optimizer_kwargs)
        lr_scheduler = self._lr_scheduler(optimizer, **self._lr_scheduler_kwargs)

        if self._lr_scheduler==torch.optim.lr_scheduler.CosineAnnealingWarmRestarts:
            lrsched_during = True
        else:
            lrsched_during = False


        itemized_criterion = lambda y_pred, y_targ: float(self._criterion(y_pred, y_targ).mean())
        if DENSE_PROGRESS_LOG_EPOCH_EVERY != 0:
            if not self.one_point_black_box:
                self._upd_dense_progress(0, *self._get_lr(optimizer), train_data, validn_data, itemized_criterion)


        self.log_f('Using shuffle seed: {}\n'.format(shuffle_seed), stdout=False)
        batcher = Batcher(self._batch_size, train_data['x'], train_data['y'], shuffle=True, shuffle_seed=shuffle_seed)
        steps_per_epoch = batcher._steps_per_epoch
        total_steps = steps_per_epoch * self._epochs
        DENSE_LOG_EVERY_BB1P = max(total_steps // DENSE_LOG_BB1P_COUNT,1)

        one_point_dump : list = []

        for i in range(self._epochs):
            if PROFILE_TIME: starttime = time.time()

            if i == self._learn_and_bias_for:
                self._model._and_l.bias.requires_grad_(False)

                learnt_bias = self._model._and_l.bias.detach().clone().cpu().numpy()

                xthreshold_begin = np.zeros(2 ** self._height) if i > 0 else -learnt_bias + np.repeat(self._thresholds[0], 2 ** self._height)
                xthreshold_end = -learnt_bias + np.repeat(self._thresholds[-1], 2 ** self._height) + self._eps

                self._xthresholds[self._learn_and_bias_for:] = torch.from_numpy(np.linspace(
                        xthreshold_begin, xthreshold_end, self._epochs - self._learn_and_bias_for
                    )).to(dtype=self._fp_type, device=self._device0)

                self._set_sigslope_and_threshold(i) # no-op for labf > 0

                self._xl1_lambda = self._l1_lambda[1]
                self._xl2_lambda = self._l2_lambda[1]

                if DEBUG_LOG_NET_EPOCH_EVERY != 0:
                    torch.set_printoptions(threshold=0, precision=8, edgeitems=10)
                    self.log_f('\nafter labf: xsigslopes [begin]:\n{}\nafter labf: xsigslopes [end]\n'.format(self._xsigslopes), stdout=False)
                    self.log_f('after labf: xthresholds [begin]:\n{}\nafter labf: xthresholds [end]\n\n'.format(self._xthresholds), stdout=False)
                    torch.set_printoptions()

            vepochs = self._vepochs if i >= self._learn_and_bias_for else 1

            if SAT_INFO_EPOCH_EVERY != 0 and (i % SAT_INFO_EPOCH_EVERY == 0):
                self._upd_sat_info(i, train_data)

            for _ in range(vepochs):
                curr_batch = -1
                while True:
                    curr_batch += 1

                    x, y = cast(Tuple[torch.Tensor, torch.Tensor], batcher.next())
                    self._update_weights(x=x, y=y, optimizer=optimizer, epoch=i, batch=curr_batch, extract_cdt_predictor_data=train_data, one_point_dump=one_point_dump)

                    if lrsched_during and (i*steps_per_epoch + curr_batch)%LR_SCHED_STEPS==0:
                        lr_scheduler.step()

                    if self.one_point_black_box and (i*steps_per_epoch + curr_batch)%DENSE_LOG_EVERY_BB1P==0:
                        self._upd_dense_progress(i*steps_per_epoch + curr_batch, *self._get_lr(optimizer), train_data, validn_data, itemized_criterion)

                    if batcher.is_new_cycle():
                        break

                if DEBUG_PREDICATES:
                    self._upd_weight_info(i+1)
                # Log
                if (DENSE_PROGRESS_LOG_EPOCH_EVERY != 0) and (i % DENSE_PROGRESS_LOG_EPOCH_EVERY == 0):
                    if not self.one_point_black_box:
                        self._upd_dense_progress(i + 1, *self._get_lr(optimizer), train_data, validn_data, itemized_criterion)
            # TODO: Should some of what comes below be inside vepochs loop?
            if COMPARE_LABEL_DIST_EPOCH_EVERY != 0 and i % COMPARE_LABEL_DIST_EPOCH_EVERY == 0:
                self.log_f('Label dist comparison on train data (epoch={}):\n{}\n'.format(i, self._get_label_dist_comp(train_data)), stdout=False)

            if (not self._use_last_model) or (self._use_lforb):
                val_loss = self.loss(validn_data or train_data, loss_func=itemized_criterion) # TODO: should include this? self._get_reg_loss(itemized=True)
                if (self._best_val_loss is None) or (val_loss < self._best_val_loss):
                    self.log_f('INFO: Saving model at epoch={} because {} > {}\n'.format(i, self._best_val_loss, val_loss), stdout=False)
                    self._best_val_loss = val_loss
                    torch.save(self._model.state_dict(), '{}/best_val_model.pt'.format(self.logs_dir))

            # Update LR
            if not lrsched_during:
                lr_scheduler.step()

            # Update sigslope and threshold
            self._set_sigslope_and_threshold(i + 1)

            if PROFILE_TIME:
                aa = np.array(self._update_weights_times)
                self._update_weights_times = []
                bb = np.array(self._log_weights_times)
                self._log_weights_times = []
                cc = np.array(self._upd_dense_progress_times)
                self._upd_dense_progress_times = []
                dd = np.array(self._get_best_labels_times)
                self._get_best_labels_times = []
                ee = np.array(self._get_best_labelsin_times)
                self._get_best_labelsin_times = []
                print('_update_weights: ({}, {}, {}, {})\n _log_weights: ({}, {}, {}, {})\n _upd_dense_progress: ({}, {}, {}, {})\n _get_best_labels: ({}, {}, {}, {})\n get_best_labelsin: ({}, {}, {}, {})'.format(
                    td(seconds=aa.sum()), td(seconds=np.nan_to_num(aa.mean())), td(seconds=np.nan_to_num(aa.std())), len(aa),
                    td(seconds=bb.sum()), td(seconds=np.nan_to_num(bb.mean())), td(seconds=np.nan_to_num(bb.std())), len(bb),
                    td(seconds=cc.sum()), td(seconds=np.nan_to_num(cc.mean())), td(seconds=np.nan_to_num(cc.std())), len(cc),
                    td(seconds=dd.sum()), td(seconds=np.nan_to_num(dd.mean())), td(seconds=np.nan_to_num(dd.std())), len(dd),
                    td(seconds=ee.sum()), td(seconds=np.nan_to_num(ee.mean())), td(seconds=np.nan_to_num(ee.std())), len(ee)
                ))
                print('epoch time: {}\n\n'.format(td(seconds=time.time() - starttime)))

        if (DENSE_PROGRESS_LOG_EPOCH_EVERY != 0) and ((self._epochs - 1) % DENSE_PROGRESS_LOG_EPOCH_EVERY != 0):
            self._upd_dense_progress(self._epochs, *self._get_lr(optimizer), train_data, validn_data, itemized_criterion)

        model_changed = False
        if not self._use_last_model:
            # since torch.load loads the model back into the device it was on (which is what self._model
            # originally was and is in now), we don't need to move anything
            self.load_best_model(self.logs_dir)
            model_changed = True

        if DENSE_PROGRESS_LOG_EPOCH_EVERY != 0:
            if model_changed:
                self._upd_dense_progress(self._epochs + 1, *self._get_lr(optimizer), train_data, validn_data, itemized_criterion)
            if self.one_point_black_box:
                dense_progress_path = '{}/dense-progress.csv'.format(self.plots_dir)
            else:
                dense_progress_path = '{}/dense-progress.csv'.format(self.logs_dir)
            self._dense_progress_df.to_csv(dense_progress_path, index=False)
            if PROGRESS_PLOTS:
                ENUtils.plot_dense_progress(dense_progress_path, '{}/dense-progress.png'.format(self.plots_dir))

        if DEBUG_PREDICATES:
            weight_progress_path = '{}/weight-progress.csv'.format(self.logs_dir)
            self._weight_info_df.to_csv(weight_progress_path, index=False)
            ENUtils.plot_weight_progress(self._weight_info_df, self.plots_dir)



        if SAT_INFO_EPOCH_EVERY != 0:
            sat_info_path = '{}/sat-info.json'.format(self.logs_dir)
            Utils.write_json(self._sat_info, sat_info_path)
            if PROGRESS_PLOTS:
                ENUtils.plot_sat_info(sat_info_path, '{}/sat-info.png'.format(self.plots_dir))

        self.log_f('\n' + Utils.get_padded_text('Weights after training: Begin', left_padding='>', right_padding='<') + '\n', stdout=False)
        if LOG_INIT_FIN_WTS:
            self._log_weights(-1, -1, grad=True, msg='after training')
        self.log_f('\n' + Utils.get_padded_text('Weights after training: End', left_padding='>', right_padding='<') + '\n', stdout=False)

        if self._black_box == BB_ENUM.BB_SMART_1P or self._black_box == BB_ENUM.BB_CLASS:
            np.save(f'{self.plots_dir}/rewards.npy', np.array(one_point_dump))

    def inference(self, x: torch.Tensor) -> torch.Tensor:
        assert self._model is not None, 'Model doesn\'t exist but inference() called'

        if self._batch_norm:
            self._model.eval()

        with torch.no_grad():
            x = x.to(dtype=self._fp_type)
            if self._device0 is not None:
                x = x.to(self._device0)
            ret = self._model(x, self._curr_xsigslope, self._curr_xsoftslope, self._curr_xthreshold,\
                              self._curr_sigrandbinprob, self._curr_softrandbinprob,)
            ret = ret.argmax(1) if self._is_classification else ret.squeeze(1)
            assert len(ret.shape) == 1

        if self._batch_norm:
            self._model.train()

        return ret

    def raw_inference(self, x: torch.Tensor) -> torch.Tensor:
        assert self._model is not None, 'Model doesn\'t exist but inference() called'

        if self._batch_norm:
            self._model.eval()

        with torch.no_grad():
            x = x.to(dtype=self._fp_type)
            if self._device0 is not None:
                x = x.to(self._device0)
            ret = self._model(x, self._curr_xsigslope, self._curr_xsoftslope, self._curr_xthreshold,  \
                            self._curr_sigrandbinprob, self._curr_softrandbinprob,)
            ret = ret if self._is_classification else ret.squeeze(1)

        if self._batch_norm:
            self._model.train()

        return ret

    def auc_inference(self, x: torch.Tensor) -> torch.Tensor:
        assert self._is_classification
        return F.softmax(self.raw_inference(x), dim=1)[:, 1]

    def load_best_model(self, logs_dir: str):
        self._use_last_model = False
        self.log_f('Loading best-model.pt ...', stdout=False)
        self._model.load_state_dict(torch.load('{}/best_val_model.pt'.format(logs_dir)))
        self.log_f('Done\n', stdout=False)

    @torch.no_grad()
    def extract_dt_predictor(self) -> DTPredictor:
        assert self._model is not None, 'Model doesn\'t exist but extract_dt_predictor() called'

        weight, bias = self._extract_pred_weights_for_dt()

        labels = self._extract_labels()
        label_scores = self._extract_label_scores()

        return DTPredictor(weight, bias, labels, label_scores=label_scores, _is_classification=self._is_classification)

    @torch.no_grad()
    def extract_cdt_predictor(self, data: Dataset[np.ndarray]) -> DTPredictor:
        assert self._model is not None, 'Model doesn\'t exist but extract_cdt_predictor() called'

        weight, bias = self._extract_pred_weights_for_dt()

        if PROFILE_TIME: starttime = time.time()

        labels, tt = DTPredictor.get_best_leaf_labels(weight, bias, data.to_device(device=self._device0), self._is_classification)

        if PROFILE_TIME: self._exec_times['get_best_leaf_labels'].append(time.time() - starttime)
        if PROFILE_TIME: self._exec_times['get_best_leaf_labelsin'].append(tt)

        return DTPredictor(weight, bias, labels, label_scores=None, _is_classification=self._is_classification)

    """
    Note: When a new sig quantizer is added whose flip threshold is not 0,
    this function should handle those cases explicitly, like done for LSQ
    """
    @torch.no_grad()
    def _extract_pred_weights_for_dt(self) -> Tuple[torch.Tensor, torch.Tensor]:
        #weight1 = self._model._predicate_l.weight
        #bias1 = self._model._predicate_l.bias
        #out_ x in_
        #out_
        f = lambda m : (m.weight.clone(), m.bias.clone())
        combined = [f(m) for m in self._model._predicate_l._modules.values()]

        #reducefn = lambda a,b: torch.cat([torch.matmul(a[:,:-1],b), a[:,-1:]],1)
        #combined = functools.reduce(reducefn, combined)
        #weight,bias = combined[:,:-1], combined[:,-1]

        weight,bias = combined[0]
        for w,b in combined[1:]:
            #print(weight.shape, bias.shape, w.shape, b.shape)
            weight = w.matmul(weight)
            bias = w.matmul(bias) + b


        flipthresh = 0 # the value at which decision flips
        if self._sigquant == 'LSQ':
            flipthresh = self._model._predicate_lsq.stepsize / 2

        if self._batch_norm:
            bn = self._model._predicate_bn
            bias_offset = ((flipthresh - bn.bias) * ((bn.running_var + bn.eps).sqrt() / bn.weight)) + bn.running_mean
        else:
            bias_offset = flipthresh
        bias = bias - bias_offset

        return weight, bias

    @torch.no_grad()
    def _extract_labels(self) -> np.ndarray:
        assert self._model is not None, 'Model doesn\'t exist but _extract_labels() called'
        assert isinstance(self._and_act_fn, nn.Sigmoid) or isinstance(self._and_act_fn, nn.ReLU) or isinstance(self._and_act_fn, nn.LeakyReLU) or isinstance(self._and_act_fn, nn.Softplus) or isinstance(self._and_act_fn, nn.Softmax)

        identity_mat = torch.eye(2 ** self._height, dtype=self._fp_type, device=self._device0)
        if not self._is_classification:
            if isinstance(self._and_act_fn, nn.ReLU) or isinstance(self._and_act_fn, nn.LeakyReLU) or isinstance(self._and_act_fn, nn.Softplus):
                assert self._model._and_l.bias.shape == (2 ** self._height, )
                assert self._xthresholds[-1].shape == (2 ** self._height, )
                identity_mat *= (self._height + self._model._and_l.bias + self._xthresholds[-1])

        if self._device0 is None:
            labels = self._model._or_l(identity_mat)
        else:
            labels = self._model._or_l(identity_mat.to(self._device0))
        labels = labels.argmax(1) if self._is_classification else labels.squeeze(1)

        return labels

    @torch.no_grad()
    def _extract_label_scores(self) -> Optional[np.ndarray]:
        if (not self._is_classification) or self._n_classes != 2:
            return None

        assert self._model is not None, 'Model doesn\'t exist but _extract_label_scores() called'
        assert isinstance(self._and_act_fn, nn.Sigmoid) or isinstance(self._and_act_fn, nn.ReLU) or isinstance(self._and_act_fn, nn.LeakyReLU) or isinstance(self._and_act_fn, nn.Softplus) or isinstance(self._and_act_fn, nn.Softmax)

        identity_mat = torch.eye(2 ** self._height, dtype=self._fp_type, device=self._device0)
        if isinstance(self._and_act_fn, nn.ReLU) or isinstance(self._and_act_fn, nn.LeakyReLU) or isinstance(self._and_act_fn, nn.Softplus):
            identity_mat *= (self._height * Utils.safe_subscript(self._xthresholds, self._epochs) + self._eps)

        if self._device0 is None:
            scores = F.softmax(self._model._or_l(identity_mat), dim=1).detach().clone().numpy().max(axis=1)
        else:
            scores = F.softmax(self._model._or_l(identity_mat.to(self._device0)), dim=1).detach().clone().to('cpu').numpy().max(axis=1)

        return scores

    def _create_model(self) -> DGT:
        return DGT(
            in_dim=self._n_features, out_dim=self._n_classes if self._is_classification else 1,
            height=self._height, learnable_and_bias=True, learnable_or_bias=self._is_classification,
            and_act_fn=self._and_act_fn, sigquant=self._sigquant, softquant=self._softquant, batch_norm=self._batch_norm, over_param=self._over_param, fp_type=self._fp_type)

    def _move_model(self):
        if self._device0 is not None:
            self._device_ids = cast(List[int], self._device_ids)
            if len(self._device_ids) > 1:
                raise NotImplementedError
                self._model = nn.DataParallel(self._model, device_ids=self._device_ids)
            # print("MOVING MODEL", self._device0, torch.cuda.memory_allocated(self._device0), sum([x.numel() for x in self._model._predicate_l.parameters()]), self._model._over_param)
            self._model.to(self._device0)
            # print("MOVED MODEL", self._device0, torch.cuda.memory_allocated(self._device0))

    def _move_data(self, data) -> Dataset[torch.Tensor]:
        if self._device0 is not None:
            # This actually needs to be done only when len(self._device_ids) == 1
            # however doing it for len > 1 is fine and may help in perf as GPU-GPU
            # tensor copying might be faster than CPU-GPU tensor copying
            data = data.to_device(self._device0)
        return data

    def _validate_args(self):
        assert self._sigslopes is not None, 'Removed support for automatic sigslopes computation'

        assert len(self._thresholds) in [1, 2]
        if len(self._thresholds) == 1:
            assert self._learn_and_bias_for > 0, 'When len(thresholds) == 1, labf is expected to be > 0'
        else:
            assert self._learn_and_bias_for == 0, 'When len(thresholds) == 2, labf is expected to be 0'

    """
    Apply normalization to the model's predicate layer vectors
    """
    def _apply_norm_to_pred(self):
        with torch.no_grad(): # TODO: this is required right?
            w = self._model._predicate_l.weight
            b = self._model._predicate_l.bias

            norm = torch.norm(w, dim=1)
            w.div_(norm.view((-1, 1)))
            b.div_(norm)

    """
    History-independent (it is set not updated)
    """
    def _set_sigslope_and_threshold(self, epoch: int):
        self._curr_xsigslope = Utils.safe_subscript(self._xsigslopes, epoch)
        self._curr_xsoftslope = Utils.safe_subscript(self._xsoftslopes, epoch)
        self._curr_xthreshold = Utils.safe_subscript(self._xthresholds, epoch)
        self._curr_sigrandbinprob = Utils.safe_subscript(self._sigrandbinprob, epoch)
        self._curr_softrandbinprob = Utils.safe_subscript(self._softrandbinprob, epoch)

    def _update_weights(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        optimizer,
        epoch: int,
        batch: int,
        extract_cdt_predictor_data: Optional[Dataset[torch.Tensor]]=None,
        one_point_dump: list= [],
    ):
        if PROFILE_TIME: starttime = time.time()

        debug_log_net: bool = self._should_debug(epoch, batch)

        optimizer.zero_grad()

        if self._black_box == BB_ENUM.BB_DIRECT:
            with torch.no_grad():
                w = Utils.model_to_vector(self._model)
                u = torch.empty(len(w)).normal_(mean=0, std=1).to(w.device).to(dtype=self._fp_type)
                u.div_(torch.norm(u))

                Utils.vector_to_model(w + self._br * u, self._model)
                y_pred = self._model(x, self._curr_xsigslope, self._curr_xsoftslope, self._curr_xthreshold, \
                                    self._curr_sigrandbinprob, self._curr_softrandbinprob,
                                    log_f=self.log_f if debug_log_net and DEBUG_INPUT_DATA else None,
                                    epoch=epoch, batch=batch)
                if not self._is_classification:
                    y_pred = y_pred.squeeze(1)

                lplus = self._criterion(y_pred, y).mean()

                Utils.vector_to_model(w - self._br * u, self._model)
                y_pred = self._model(x, self._curr_xsigslope, self._curr_xsoftslope, self._curr_xthreshold,
                                    self._curr_sigrandbinprob, self._curr_softrandbinprob,
                                    log_f=self.log_f if debug_log_net and DEBUG_INPUT_DATA else None,
                                    epoch=epoch, batch=batch)
                if not self._is_classification:
                    y_pred = y_pred.squeeze(1)
                lminus = self._criterion(y_pred, y).mean()

                Utils.vector_to_model(w, self._model)
                grad = (lplus - lminus) * u
                Utils.vector_to_model(grad, self._model, replace_grad=True)
                self._get_reg_loss().backward()

        else:
            # Forward pass
            if debug_log_net and DEBUG_INPUT_DATA:
                self.log_f('\n{}\n'.format(Utils.get_padded_text('Forward pass: Begin', left_padding='-', right_padding='-')), stdout=False)

            y_pred = self._model(
                x,
                self._curr_xsigslope,
                self._curr_xsoftslope,
                self._curr_xthreshold,
                self._curr_sigrandbinprob,
                self._curr_softrandbinprob,
                log_f=self.log_f if debug_log_net and DEBUG_INPUT_DATA else None,
                epoch=epoch,
                batch=batch
            )
            if not self._is_classification:
                y_pred = y_pred.squeeze(1)

            if debug_log_net and DEBUG_INPUT_DATA:
                if DEBUG_Y:
                    self.log_f('\n|\n|\nY_TARGET: \n{}\n|\n|\n'.format(y), stdout=False)
                    self.log_f('Y_PRED: \n{}\n|\n|\n'.format(y_pred), stdout=False)
                    if self._is_classification:
                        self.log_f('(e^o_t)/sigma(e^o_i): \n{}\n|\n|\n'.format(F.softmax(y_pred, dim=1)[range(len(y_pred)), y]), stdout=False)
                    else:
                        self.log_f('Y_PRED - Y_TARGET: \n{}\n|\n|\n'.format(y_pred - y), stdout=False)
                self.log_f('\n{}\n'.format(Utils.get_padded_text('Forward pass: End', left_padding='-', right_padding='-')), stdout=False)

            # Compute loss and gradients
            if self._black_box is not None:
                if self._black_box == BB_ENUM.BB_CLASS:
                    arange = torch.arange(y.shape[0])
                    y_hat = y_pred.detach().argmax(-1)
                    P = (1-self._gamma)*F.one_hot(y_hat, self._n_classes) + self._gamma/self._n_classes
                    y_hat_twilda = torch.distributions.categorical.Categorical(P).sample()
                    logits = y_pred[arange, y_hat_twilda]
                    sigmoids = torch.sigmoid(logits)
                    reward = (y_hat_twilda==y).to(self._fp_type)
                    one_point_dump.append(reward.detach().cpu().numpy())
                    loss = F.mse_loss(sigmoids, reward, reduction='none')
                    loss = loss/P[arange, y_hat_twilda]
                    loss = loss.mean()
                    loss.backward()

                else:
                    with torch.no_grad():
                        if self._black_box == BB_ENUM.BB_SMART_1P:
                            u = 2 * np.random.random_integers(0, 1) - 1
                            negreward = self._criterion(y_pred + self._br * u, y)
                            lp = (1 / self._br) * negreward * u
                            one_point_dump.append(negreward[0].detach().cpu().numpy())
                        elif self._black_box == BB_ENUM.BB_SMART_2P:
                            u = 2 * np.random.random_integers(0, 1) - 1
                            lp0 = (1 / (2 * self._br)) * (self._criterion(y_pred + self._br * u, y) - self._criterion(y_pred - self._br * u, y)) * u
                            # lp1 = self._criterion(y_pred + self._br, y) - self._criterion(y_pred - self._br, y)
                            # lp2 = 4 * self._br * (y_pred - y)
                            lp = lp0

                        elif self._black_box == BB_ENUM.BB_QUANT_DT:
                            y_pred_dt = self.extract_dt_predictor().inference(x)
                            lp = 2 * (y_pred_dt - y)

                        else:
                            # BB_QUANT_CDT
                            y_pred_cdt = self.extract_cdt_predictor(extract_cdt_predictor_data).inference(x)
                            lp = 2 * (y_pred_cdt - y)

                        lp_fact = (1 / y_pred.shape[0]) * lp

                        if debug_log_net and DEBUG_INPUT_DATA:
                            torch.set_printoptions(threshold=0, precision=8, edgeitems=50)
                            self.log_f('lp (epoch={}, batch={}):\n{}\n|\n|\n'.format(epoch, batch, lp), stdout=False)
                            self.log_f('y (epoch={}, batch={}):\n{}\n|\n|\n'.format(epoch, batch, y), stdout=False)
                            self.log_f('y_pred (epoch={}, batch={}):\n{}\n|\n|\n'.format(epoch, batch, y_pred), stdout=False)
                            if self._black_box == BB_ENUM.BB_QUANT_DT:
                                self.log_f('y_pred_dt (epoch={}, batch={}):\n{}\n|\n|\n'.format(epoch, batch, y_pred_dt), stdout=False)
                            elif self._black_box == BB_ENUM.BB_QUANT_CDT:
                                self.log_f('y_pred_cdt (epoch={}, batch={}):\n{}\n|\n|\n'.format(epoch, batch, y_pred_cdt), stdout=False)
                            torch.set_printoptions()

                    y_pred.backward(lp_fact)
                    self._get_reg_loss().backward()

            else:
                loss = self._criterion(y_pred, y).mean() + self._get_reg_loss()
                loss.backward()

        if debug_log_net:
            self.log_f('\n{}\n'.format(Utils.get_padded_text('Before optimizer step(): Begin', left_padding='-', right_padding='-')), stdout=False)
            self._log_weights(epoch=epoch, batch=batch, grad=True, msg='before optimizer.step()')
            self.log_f('{}\n'.format(Utils.get_padded_text('Before optimizer step(): End', left_padding='-', right_padding='-')), stdout=False)

        # Update weights
        if self._grad_clip is not None:
            nn.utils.clip_grad_norm_(self._model.parameters(), self._grad_clip)
        optimizer.step()

        if debug_log_net:
            self.log_f('\n{}\n'.format(Utils.get_padded_text('After optimizer step(): Begin', left_padding='-', right_padding='-')), stdout=False)
            self._log_weights(epoch=epoch, batch=batch, grad=True, msg='after optimizer.step()')
            self.log_f('{}\n'.format(Utils.get_padded_text('After optimizer step(): End', left_padding='-', right_padding='-')), stdout=False)

        if APPLY_NORM_TO_PREDPOOL:
            self._apply_norm_to_pred()

            if debug_log_net:
                self.log_f('\n{}\n'.format(Utils.get_padded_text('After projection: Begin', left_padding='-', right_padding='-')), stdout=False)
                self._log_weights(epoch=epoch, batch=batch, grad=True, msg='after projection')
                self.log_f('{}\n'.format(Utils.get_padded_text('After projection: End', left_padding='-', right_padding='-')), stdout=False)

        if PROFILE_TIME: self._exec_times['_update_weights'].append(time.time() - starttime)

    def _should_debug(self, epoch: int, batch: int) -> bool:
        return ((DEBUG_LOG_NET_EPOCH_EVERY != 0 and epoch % DEBUG_LOG_NET_EPOCH_EVERY == 0) and
            ((DEBUG_LOG_NET_BATCH_EVERY != 0 and batch % DEBUG_LOG_NET_BATCH_EVERY == 0) or
                (batch < DEBUG_LOG_NET_BATCH_FIRST)))

    """
    Get regularized loss
    """
    def _get_reg_loss(self, itemized: bool=False) -> Union[float, torch.Tensor]:
        if (self._xl1_lambda != 0) or (self._xl2_lambda != 0):
            with torch.set_grad_enabled(not itemized):
                l1_norm = 0
                l2_norm_sq: Union[int, torch.Tensor] = 0
                if self._pred_reg_only:
                    allparams = self._model._predicate_l.parameters()
                else:
                    allparams = self._model.parameters()
                for param in allparams:
                    if param.requires_grad:
                        l1_norm += torch.norm(param, 1)
                        if self._xl2_lambda != 0:
                            l2_norm_sq += torch.pow(torch.norm(param, 2), 2)

                reg_loss = cast(torch.Tensor, self._xl1_lambda * l1_norm)
                if self._xl2_lambda != 0:
                    reg_loss = reg_loss + self._xl2_lambda * (1 / 2) * l2_norm_sq
        else:
            reg_loss = torch.tensor([0.], requires_grad=True, dtype=self._fp_type, device=self._device0)

        if itemized:
            reg_loss = reg_loss.item()
        return reg_loss

    def _log_weights(self, epoch: int, batch: int, grad: bool=True, msg: str=''):
        if PROFILE_TIME: starttime = time.time()
        torch.set_printoptions(threshold=1000000, precision=8)
        log_str = ''

        # Predicate Layer
        # log_str += 'PRED weight (epoch={}, batch={}) [{}]: \n{}\n|\n|\n'.format(epoch, batch, msg, self._model._predicate_l.weight[:, :50])
        # if grad:
        #     log_str += 'PRED weight grad (epoch={}, batch={}) [{}]: \n{}\n|\n|\n'.format(epoch, batch, msg, self._model._predicate_l.weight.grad[:, :50] if self._model._predicate_l.weight.grad is not None else None)
        # log_str += 'PRED bias (epoch={}, batch={}) [{}]: \n{}\n|\n|\n'.format(epoch, batch, msg, self._model._predicate_l.bias)
        # if grad:
        #     log_str += 'PRED bias grad (epoch={}, batch={}) [{}]: \n{}\n|\n|\n'.format(epoch, batch, msg, self._model._predicate_l.bias.grad)


        # AND Layer
        log_str += 'AND weight (epoch={}, batch={}) [{}]: \n{}\n|\n|\n'.format(epoch, batch, msg, self._model._and_l.weight[:5, :5])
        if grad:
            log_str += 'AND weight grad (epoch={}, batch={}) [{}]: \n{}\n|\n|\n'.format(epoch, batch, msg, self._model._and_l.weight.grad[:5, :5] if self._model._and_l.weight.grad is not None else None)
        log_str += 'AND bias (epoch={}, batch={}) [{}]: \n{}\n|\n|\n'.format(epoch, batch, msg, self._model._and_l.bias)
        if grad:
            log_str += 'AND bias grad (epoch={}, batch={}) [{}]: \n{}\n|\n|\n'.format(epoch, batch, msg, self._model._and_l.bias.grad)

        # OR Layer
        log_str += 'OR weight (epoch={}, batch={}) [{}]: \n{}\n|\n|\n'.format(epoch, batch, msg, self._model._or_l.weight)
        if grad:
            log_str += 'OR weight grad (epoch={}, batch={}) [{}]: \n{}\n|\n|\n'.format(epoch, batch, msg, self._model._or_l.weight.grad)
        log_str += 'OR bias (epoch={}, batch={}) [{}]: \n{}\n|\n|\n'.format(epoch, batch, msg, self._model._or_l.bias)
        if grad:
            log_str += 'OR bias grad (epoch={}, batch={}) [{}]: \n{}\n|\n|\n'.format(epoch, batch, msg, self._model._or_l.bias.grad)

        self.log_f(log_str, stdout=False)
        torch.set_printoptions()

        if PROFILE_TIME: self._exec_times['_log_weights'].append(time.time() - starttime)

    """
    Exact CART perf will be obtained only when labels also work, which
    will happen only if AND layer outputs 1-hot vector.

    Note: When a new sig quantizer is added whose flip threshold is not 0,
    this function should handle those cases explicitly, like done for LSQ
    """
    def _init_model_with_cart(self, train_data: Dataset[torch.Tensor], validn_data: Dataset[torch.Tensor], test_data: Dataset[torch.Tensor], random_seed: int):
        assert not self._is_classification
        assert self._sigquant is not None, 'Initializing with CART weights won\'t help unless predicate layer is quantized'

        train_data = train_data.to_ndarray()
        x = train_data['x']
        y = train_data['y']

        cart_model = DecisionTreeRegressor(max_depth=self._height, random_state=random_seed)
        cart_model.fit(x, y)

        weights, biases, labels, _ = Utils.get_cart_wts(cart_model, self._height, train_data.n_features)
        if self._sigquant == 'LSQ':
            biases = biases + LSQLayer.init_when_cart / 2

        cart_train_acc = sklearn.metrics.mean_squared_error(train_data['y'], cart_model.predict(train_data['x']))
        cart_validn_acc = sklearn.metrics.mean_squared_error(validn_data['y'], cart_model.predict(validn_data['x']))
        cart_test_acc = sklearn.metrics.mean_squared_error(test_data['y'], cart_model.predict(test_data['x']))
        self.log_f(f'\nCART train_acc: {Utils.denormalize_acc(cart_train_acc, AccFuncType.mse, train_data.mirror_y_params)}\n', stdout=False)
        self.log_f(f'CART validn_acc: {Utils.denormalize_acc(cart_validn_acc, AccFuncType.mse, train_data.mirror_y_params)}\n', stdout=False)
        self.log_f(f'CART test_acc: {Utils.denormalize_acc(cart_test_acc, AccFuncType.mse, train_data.mirror_y_params)}\n\n', stdout=False)

        self._model._predicate_l.weight = nn.Parameter(torch.from_numpy(weights).to(dtype=FPTYPE), requires_grad=True)
        self._model._predicate_l.bias = nn.Parameter(torch.from_numpy(biases).to(dtype=FPTYPE), requires_grad=True)

        self._model._or_l._l.weight = nn.Parameter(torch.from_numpy(labels.reshape((1, -1))).to(dtype=FPTYPE), requires_grad=True)

    def _upd_weight_info(self, epoch):
        layers = self._model._predicate_l._modules.values()
        len_preds = len(layers)
        colnames = ['epoch'] + sum([[f'l{i}_mean',f'l{i}_std',f'l{i}_median',f'l{i}_max',f'l{i}_min'] for i in range(len_preds)], [])
        values = [epoch] + sum([[x.detach().cpu().numpy() for x in [w.mean(), w.std(), w.median(), w.max(), w.abs().min(),]] for layer in layers for w in [layer.weight]], [])
        dct = dict(zip(colnames, values))
        self._weight_info_df = self._weight_info_df.append(dct, ignore_index=True)
        return

    def _upd_dense_progress(
        self, epoch: int, lr1: float, lr2: float,
        train_data: Dataset[torch.Tensor], validn_data: Optional[Dataset[torch.Tensor]],
        itemized_criterion: Callable[[np.ndarray, np.ndarray], float]
    ):
        if PROFILE_TIME: starttime = time.time()

        reg_loss = cast(float, self._get_reg_loss(itemized=True))
        train_loss = self.loss(train_data, loss_func=itemized_criterion)
        train_loss_wreg = train_loss + reg_loss

        train_acc = self.acc(train_data, denormalize=DENORMALIZE_RESULTS)

        validn_loss = None
        validn_acc = None
        validn_loss_wreg = None
        if validn_data:
            validn_loss = self.loss(validn_data, loss_func=itemized_criterion)
            validn_loss_wreg = validn_loss + reg_loss
            validn_acc = self.acc(validn_data, denormalize=DENORMALIZE_RESULTS)

        if self._is_pure_dt:
            dt_train_acc = None
            dt_validn_acc = None
            cdt_train_acc = None
            cdt_validn_acc = None
        else:
            dt = self.extract_dt_predictor()
            dt.acc_func = self.acc_func
            dt.acc_func_type = self.acc_func_type
            dt_train_acc = dt.acc(train_data, denormalize=DENORMALIZE_RESULTS)
            dt_validn_acc = None
            dt_validn_acc = dt.acc(validn_data, denormalize=DENORMALIZE_RESULTS)

            cdt = self.extract_cdt_predictor(train_data)
            cdt.acc_func = self.acc_func
            cdt.acc_func_type = self.acc_func_type
            cdt_train_acc = cdt.acc(train_data, denormalize=DENORMALIZE_RESULTS)
            cdt_validn_acc = None
            cdt_validn_acc = cdt.acc(validn_data, denormalize=DENORMALIZE_RESULTS)


        dct = {
            'epoch': epoch,
            'train_loss_wreg': train_loss_wreg,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'validn_loss_wreg': validn_loss_wreg,
            'validn_loss': validn_loss,
            'validn_acc': validn_acc,
            'dt_train_acc': dt_train_acc,
            'dt_validn_acc': dt_validn_acc,
            'cdt_train_acc': cdt_train_acc,
            'cdt_validn_acc': cdt_validn_acc,
            'lr_pred': '{:e}'.format(lr1),
            'lr_or': '{:e}'.format(lr2),
            'xsigslope': self._curr_xsigslope[0].item(),
            'xsoftslope': self._curr_xsoftslope[0].item(),
            'xthreshold': self._curr_xthreshold[0].item(),
            'sigrandbinprob' :  self._curr_sigrandbinprob,
            'softrandbinprob' :  self._curr_softrandbinprob,
        }
        if self._sigquant == 'LSQ':
            dct['stepsize'] = self._model._predicate_lsq.stepsize.detach().item()
        self._dense_progress_df = self._dense_progress_df.append(dct, ignore_index=True)

        if PROFILE_TIME: self._exec_times['_upd_dense_progress'].append(time.time() - starttime)

    def _init_sat_info(self) -> Dict[str, Union[int, List[int], List[List[int]]]]:
        n_intervals = len(SAT_INFO_DIST_DISC) + 1
        d: Dict[str, List[Union[int, List[int]]]] = {'height': self._height, 'epoch': []}

        for i in range(n_intervals):
            d[self.get_interval_desc(i)] = []
        return d

    """
    Get textual description of an interval in the saturation info discretization
    """
    def get_interval_desc(self, idx: int) -> str:
        disc = SAT_INFO_DIST_DISC
        if idx == 0:
            return '(,{:.2f}]'.format(disc[0])
        elif idx == len(disc):
            return '[{:.2f},)'.format(disc[-1])
        else:
            return '({:.2f},{:.2f}]'.format(disc[idx - 1], disc[idx])


    """
    Returns a list whose each element corresponds to one interval in the
    sat info discretization and contains the fraction of outputs belonging to that
    interval. If number of internal nodes is <= SAT_INFO_NODE_MAX then
    a list of fractions each corresponding to one internal node is given otherwise
    a singleton list is present that denotes the aggregate over all nodes.

    List[List]: for each interval[for each node[exists a fraction]]
    """
    def get_sat_info(self, data: Dataset[torch.Tensor], skip_data_logistics: bool=False) -> List[List[float]]:
        if not skip_data_logistics:
            data = data.to_type(self._fp_type, 'x' if self._is_classification else 'xy')
            data = self._move_data(data)

        with torch.no_grad():
            z = self._model._predicate_l(data['x'])
            a = torch.sigmoid(self._curr_xsigslope * z)

        diff = (a - 0.5).abs()
        int_nodes = (2 ** self._height - 1)
        node_safe = int_nodes <= SAT_INFO_NODE_MAX
        disc = SAT_INFO_DIST_DISC

        n_intervals = len(disc) + 1
        res = [None] * n_intervals

        n = data.n_examples * int_nodes
        cnt = (diff <= disc[0]).float()
        res[0] = torch.div(cnt.sum(dim=0), n).cpu().numpy().tolist() if node_safe else [torch.div(cnt.sum(), n).cpu().numpy().tolist()]

        for i in range(1, len(disc)):
            cnt = ((diff > disc[i - 1]) & (diff <= disc[i])).float()
            res[i] = torch.div(cnt.sum(dim=0), n).cpu().numpy().tolist() if node_safe else [torch.div(cnt.sum(), n).cpu().numpy().tolist()]

        cnt = (diff >= disc[-1]).float()
        res[-1] = torch.div(cnt.sum(dim=0), n).cpu().numpy().tolist() if node_safe else [torch.div(cnt.sum(), n).cpu().numpy().tolist()]

        return res

    def _upd_sat_info(self, epoch: int, train_data: Dataset[torch.Tensor]):
        n_intervals = len(SAT_INFO_DIST_DISC) + 1
        sat_info = self.get_sat_info(train_data, skip_data_logistics=True)

        self._sat_info['epoch'].append(epoch)
        for i in range(n_intervals):
            self._sat_info[self.get_interval_desc(i)].append(sat_info[i])

    def _get_lr(
        self,
        optimizer: optim.Optimizer # type: ignore
    ) -> List[float]:

        lrs = [param_group['lr'] for param_group in optimizer.param_groups]
        #assert len(lrs) == 1
        return lrs#[0]

    def _is_l2_reg_avail(self) -> bool:
        return False

    def _get_label_dist_comp(self, data) -> str:
        y_pred = self.inference(data['x']).detach().clone().to('cpu').numpy()
        y_targ = data['y'].detach().clone().to('cpu').numpy()

        if LOG_PRED_TARG_DIFF:
            y_diff = np.abs(y_pred - y_targ)
            els = list(zip(y_diff, y_pred, y_targ))
            els = sorted(els, key=lambda x: -x[0])
            self.log_f('(diff, pred, targ)\n{}\n'.format(els), stdout=False)

        if self._is_classification:
            pred_cnts = dict(zip(*np.unique(y_pred, return_counts=True)))
            targ_cnts = dict(zip(*np.unique(y_targ, return_counts=True)))

            for i in range(self._n_classes):
                if i not in pred_cnts:
                    pred_cnts[i] = 0
                if i not in targ_cnts:
                    targ_cnts[i] = 0

            return '\n'.join([
                '{}: pred, targ:  {}, {}: diff: {}'.format(
                    i, pred_cnts[i], targ_cnts[i], pred_cnts[i] - targ_cnts[i]) for i in range(self._n_classes)
            ])

        else:
            lo = min(y_pred.min(), y_targ.min())
            hi = max(y_pred.max(), y_targ.max())

            bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
            if lo < bins[0]:
                bins = [lo] + bins
            if hi > bins[-1]:
                bins.append(hi)

            return '\n'.join(
                ['{:.4f}: pred, targ:  {}, {}'.format(edge, pred_cnt, targ_cnt) for pred_cnt, edge, targ_cnt, edge in zip(
                    *np.histogram(y_pred, bins=bins, density=False),
                    *np.histogram(y_targ, bins=bins, density=False))]
            )
