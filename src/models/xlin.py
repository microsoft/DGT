from collections import OrderedDict
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union, cast

from xconstants import REMOTE_SESS
if REMOTE_SESS:
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from xcommon import Batcher, Dataset, LearnablePredictor
from xconstants import (BB_ENUM, DENSE_PROGRESS_LOG_EPOCH_EVERY, DENSE_LOG_BB1P_COUNT)

class LinearModel(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self._w = nn.Parameter(torch.tensor(np.random.normal(loc=0, scale=1, size=(n_features,)), dtype=torch.float32), requires_grad=True)
        self._b = nn.Parameter(torch.tensor([0.], dtype=torch.float32), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor: # type: ignore
        return torch.matmul(x, self._w.view((-1, 1))).view(-1) + self._b

class LinPredictor(LearnablePredictor[np.ndarray]):
    def __init__(
        self,
        n_features: int=2,
        n_classes: int=-1,

        epochs: int=100,
        l1_lambda: float=0.1,
        l2_lambda: float=0,
        lr: float=0.01,
        batch_size: int=100,

        criterion: Callable[[np.ndarray, np.ndarray], Union[torch.Tensor]]=nn.CrossEntropyLoss(reduction='none'),
        optimizer: Type[optim.Optimizer]=optim.SGD, # type: ignore
        optimizer_kwargs: Dict[str, Any]={},
        lr_scheduler: Type[Any]=optim.lr_scheduler.StepLR,
        lr_scheduler_kwargs: Dict[str, Any]={'step_size': 1, 'gamma': 1},

        black_box: Optional[str]=None,
        br: float=0.5,

        device_ids=None,
    ):
        self._n_features = n_features
        self._n_classes = n_classes
        self._device_ids = device_ids
        if self._device_ids is not None:
            self._device0 = torch.device('cuda:{}'.format(self._device_ids[0]))
        self._epochs = epochs
        self._l1_lambda = l1_lambda
        self._l2_lambda = l2_lambda
        self._lr = lr
        self._batch_size = batch_size

        self._criterion = criterion
        self._optimizer = optimizer
        self._optimizer_kwargs = optimizer_kwargs
        self._lr_scheduler = lr_scheduler
        self._lr_scheduler_kwargs = lr_scheduler_kwargs
        self._black_box = black_box
        self._br = br

        self._model = LinearModel(n_features)
        self._train_invoked = False

        self._is_classification = n_classes > 0
        assert not self._is_classification

        self._dense_progress_df = pd.DataFrame(dtype=np.float64)

    def get_hyperparams(self) -> Dict[str, Any]:
        return OrderedDict(
            n_features=self._n_features,
            n_classes=self._n_classes,

            epochs=self._epochs,
            l1_lambda=self._l1_lambda,
            l2_lambda=self._l2_lambda,
            lr=self._lr,
            batch_size=self._batch_size,

            criterion=self._criterion,
            optimizer=self._optimizer,
            optimizer_kwargs=self._optimizer_kwargs,
            lr_scheduler=self._lr_scheduler,
            lr_scheduler_kwargs=self._lr_scheduler_kwargs,

            black_box=self._black_box,
            br=self._br,
        )

    def _move_model(self):
        self._model = self._model.to(self._device0)

    def _move_data(self, data) -> Dataset[torch.Tensor]:
        if self._device0 is not None:
            # This actually needs to be done only when len(self._device_ids) == 1
            # however doing it for len > 1 is fine and may help in perf as GPU-GPU
            # tensor copying might be faster than CPU-GPU tensor copying
            data = data.to_tensor().to_device(self._device0)
        return data

    def train(self, train_data: Dataset[np.ndarray], validn_data: Dataset[np.ndarray], test_data: Dataset[np.ndarray]):
        assert not self._train_invoked
        self._train_invoked = True
        self._move_model()
        train_data = self._move_data(train_data)
        validn_data = self._move_data(validn_data)
        test_data = self._move_data(test_data)
        # Use custom l2 reg instead of weight_decay parameter because we don't want to include bias
        # in regularization
        optimizer = self._optimizer(self._model.parameters(), lr=self._lr, **self._optimizer_kwargs)
        lr_scheduler = self._lr_scheduler(optimizer, **self._lr_scheduler_kwargs)

        #itemized_criterion = lambda y_pred, y_targ: float(self._criterion(y_pred, y_targ).mean())
        def itemized_criterion(y_pred, y_targ):
            #print("PRED", y_pred.shape, type(y_pred))
            #print(y_targ.shape)
            return float(self._criterion(y_pred, y_targ).mean())
        
        if DENSE_PROGRESS_LOG_EPOCH_EVERY != 0:
            if 'smart1p' not in str(self._black_box):
                self._log_dense_progress(0, self._get_lr(optimizer), train_data, validn_data, itemized_criterion)

        shuffle_seed = np.random.randint(10)
        batcher = Batcher(self._batch_size, train_data['x'], train_data['y'], shuffle=True, shuffle_seed=shuffle_seed)
        steps_per_epoch = batcher._steps_per_epoch
        total_steps = steps_per_epoch * self._epochs
        DENSE_LOG_EVERY_BB1P = total_steps // DENSE_LOG_BB1P_COUNT
        one_point_dump : list = []

        for i in range(self._epochs):
            batch = 0
            curr_batch = -1
            while True:
                batch += 1
                curr_batch += 1
                optimizer.zero_grad()
                x, y = cast(Tuple[torch.Tensor, torch.Tensor], batcher.next())

                if self._black_box in [BB_ENUM.BB_SMART_1P, BB_ENUM.BB_SMART_2P]:
                    y_pred = self._model(x)

                    with torch.no_grad():
                        if self._black_box == BB_ENUM.BB_SMART_1P:
                            u = 2 * np.random.random_integers(0, 1) - 1
                            negreward = self._criterion(y_pred + self._br * u, y)
                            lp = (1 / self._br) * negreward * u
                            one_point_dump.append(negreward[0].detach().cpu().numpy())
                        else:
                            u = 2 * np.random.random_integers(0, 1) - 1
                            lp0 = (1 / (2 * self._br)) * (self._criterion(y_pred + self._br * u, y) - self._criterion(y_pred - self._br * u, y)) * u
                            lp1 = self._criterion(y_pred + self._br, y) - self._criterion(y_pred - self._br, y)
                            lp2 = 4 * self._br * (y_pred - y)
                            lp = lp0

                        lp_fact = (1 / y_pred.shape[0]) * lp

                    y_pred.backward(lp_fact)
                    self._get_reg_loss().backward()

                elif self._black_box == BB_DIRECT:
                    raise ValueError
                    # with torch.no_grad():
                    #     u = torch.empty(len(self._w) + 1).normal_(mean=0, std=1).to(self._w.device).to(dtype=self._fp_type)
                    #     u.div_(torch.norm(u))

                    #     Utils.vector_to_model(w + self._br * u, self._model)
                    #     y_pred = self._model(x, self._curr_sigslope, self._curr_threshold, log_f=self.log_f if debug_log_net and DEBUG_INPUT_DATA else None, epoch=i, batch=curr_batch)
                    #     if not self._is_classification:
                    #         y_pred = y_pred.squeeze(1)
                    #     lplus = self._criterion(y_pred, y).mean()

                    #     Utils.vector_to_model(w - self._br * u, self._model)
                    #     y_pred = self._model(x, self._curr_sigslope, self._curr_threshold, log_f=self.log_f if debug_log_net and DEBUG_INPUT_DATA else None, epoch=i, batch=curr_batch)
                    #     if not self._is_classification:
                    #         y_pred = y_pred.squeeze(1)
                    #     lminus = self._criterion(y_pred, y).mean()

                    #     Utils.vector_to_model(w, self._model)
                    #     grad = (lplus - lminus) * u
                    #     Utils.vector_to_model(grad, self._model, replace_grad=True)
                    #     self._get_reg_loss().backward()

                else:
                    raise ValueError('black_box (={}) must be one of [{}, {}]'.format(self._black_box, BB_SMART, BB_DIRECT))

                # if batch > 2:
                #     exit()
                # print('w:', self._model._w)
                # print('w_grad:', self._model._w.grad)
                # print('=======================')
                # print('b:', self._model._b)
                # print('b_grad:', self._model._b.grad)
                # print('=======================')

                if 'smart1p' in str(self._black_box) and (i*steps_per_epoch + curr_batch)%DENSE_LOG_EVERY_BB1P==0:
                    self._log_dense_progress(i*steps_per_epoch + curr_batch, self._get_lr(optimizer), train_data, validn_data, itemized_criterion)

                optimizer.step()
                if batcher.is_new_cycle():
                    break

            if (DENSE_PROGRESS_LOG_EPOCH_EVERY != 0) and (i % DENSE_PROGRESS_LOG_EPOCH_EVERY == 0):
                if 'smart1p' not in str(self._black_box):
                    self._log_dense_progress(i + 1, self._get_lr(optimizer), train_data, validn_data, itemized_criterion)

            lr_scheduler.step()

        if (DENSE_PROGRESS_LOG_EPOCH_EVERY != 0) and ((self._epochs - 1) % DENSE_PROGRESS_LOG_EPOCH_EVERY != 0):
            self._log_dense_progress(self._epochs, self._get_lr(optimizer), train_data, validn_data, itemized_criterion)

        if DENSE_PROGRESS_LOG_EPOCH_EVERY != 0:
            self._plot_dense_progress()

        if self._black_box == BB_ENUM.BB_SMART_1P:
            np.save(f'{self.plots_dir}/rewards.npy', np.array(one_point_dump))


    def inference(self, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            #x = x.to(dtype=self._fp_type)
            if self._device0 is not None:
                x = x.to(self._device0)
            return self._model(x)

    def raw_inference(self, x: np.ndarray) -> np.ndarray:
        return self.inference(x)

    def auc_inference(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _log_dense_progress(
        self, curr_epoch: int, lr: float,
        train_data: Dataset[torch.Tensor], validn_data: Optional[Dataset[torch.Tensor]],
        itemized_criterion: Callable[[np.ndarray, np.ndarray], float]
    ):
        reg_loss = cast(float, self._get_reg_loss(itemized=True))
        train_loss = self.loss(train_data, loss_func=itemized_criterion) + reg_loss
        train_acc = self.acc(train_data)

        validn_loss = None
        validn_acc = None
        if validn_data:
            validn_loss = self.loss(validn_data, loss_func=itemized_criterion) + reg_loss
            validn_acc = self.acc(validn_data)

        self._dense_progress_df = self._dense_progress_df.append({
            'epoch': curr_epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'validn_loss': validn_loss,
            'validn_acc': validn_acc,
            'lr': lr}, ignore_index=True)

        if 'smart1p' in str(self._black_box):
            dense_progress_path = '{}/dense-progress.csv'.format(self.plots_dir)
        else:
            dense_progress_path = '{}/dense-progress.csv'.format(self.logs_dir)

        self._dense_progress_df.to_csv(dense_progress_path, index=False)

    def _plot_dense_progress(self):
        _, ax = plt.subplots(nrows=2, ncols=2, figsize=(18, 12))
        df = self._dense_progress_df
        msize = 10

        # train_acc, validn_acc
        ax[0][0].set_xlabel('Epochs')
        ax[0][0].plot(df['epoch'], df['train_acc'], label='train_acc', marker='.', markersize=msize)
        ax[0][0].plot(df['epoch'], df['validn_acc'], label='validn_acc', marker='.', markersize=msize)
        ax[0][0].grid()
        ax[0][0].legend()

        # train_loss, validn_loss
        ax[0][1].set_xlabel('Epochs')
        ax[0][1].plot(df['epoch'], df['train_loss'], label='train_loss', marker='.', markersize=msize)
        ax[0][1].plot(df['epoch'], df['validn_loss'], label='validn_loss', marker='.', markersize=msize)
        ax[0][1].grid()
        ax[0][1].legend()

        # lr
        ax[1][0].set_xlabel('Epochs')
        ax[1][0].plot(df['epoch'], df['lr'], label='lr', marker='.', markersize=msize)
        ax[1][0].grid()
        ax[1][0].legend()

        plt.savefig('{}/dense-progress.png'.format(self.plots_dir))
        plt.close()

    def _get_lr(
        self,
        optimizer: optim.Optimizer # type: ignore
    ) -> float:

        lrs = [param_group['lr'] for param_group in optimizer.param_groups]
        assert len(lrs) == 1
        return lrs[0]

    """
    Get regularized loss
    """
    def _get_reg_loss(self, itemized: bool=False) -> Union[float, torch.Tensor]:
        if (self._l1_lambda != 0) or (self._l2_lambda != 0):
            with torch.set_grad_enabled(not itemized):
                l1_norm = 0
                l2_norm_sq: Union[int, torch.Tensor] = 0

                for param in [self._model._w]:
                    if param.requires_grad:
                        l1_norm += torch.norm(param, 1)
                        if self._l2_lambda != 0:
                            l2_norm_sq += torch.pow(torch.norm(param, 2), 2)

                reg_loss = cast(torch.Tensor, self._l1_lambda * l1_norm)
                if self._l2_lambda != 0:
                    reg_loss = reg_loss + self._l2_lambda * l2_norm_sq
        else:
            reg_loss = torch.tensor([0.], requires_grad=True, dtype=torch.float32, device=self._device0)

        if itemized:
            reg_loss = reg_loss.item()
        return reg_loss