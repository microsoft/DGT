from collections import OrderedDict
from typing import Any, Dict, Tuple, List
import json

import numpy as np
from vowpalwabbit import pyvw
from xconstants import REMOTE_SESS
if REMOTE_SESS:
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from xcommon import Dataset, LearnablePredictor


class VWPredictor(LearnablePredictor[np.ndarray]):
    def __init__(
        self,
        n_features: int,
        n_classes: int,
        epochs: int,
        lr: float,
        epsilon: float,
        l1_lambda: float,
        l2_lambda: float,
        num_tlogs: int
    ):
        super().__init__()
        self._n_features = n_features
        self._n_classes = n_classes
        self._epochs = epochs
        self._lr = lr
        self._epsilon = epsilon
        self._l1_lambda = l1_lambda
        self._l2_lambda = l2_lambda
        self._num_tlogs = num_tlogs

        self._is_classification = n_classes > 0
        assert self._is_classification
        self._train_invoked = False

        self._vw = pyvw.vw(f' --cb_explore {n_classes} --epsilon {epsilon} --learning_rate {lr} --l1 {l1_lambda} --l2 {l2_lambda}')

    def get_hyperparams(self) -> Dict[str, Any]:
        return OrderedDict(
            n_features=self._n_features,
            n_classes=self._n_classes,
            epochs=self._epochs,
            lr=self._lr,
            epsilon=self._epsilon,
            l1_lambda=self._l1_lambda,
            l2_lambda=self._l2_lambda,
            num_tlogs=self._num_tlogs
        )

    def train(self, train_data: Dataset[np.ndarray], validn_data: Dataset[np.ndarray], test_data: Dataset[np.ndarray]):
        assert not self._train_invoked, 'train() already invoked once'
        self._train_invoked = True

        self._validate_data(train_data, validn_data, test_data)

        trainx, trainy = VWPredictor.stringify_x(train_data['x']), train_data['y'].tolist()

        reward_progress = []
        test_acc_progress = []

        total_iters = self._epochs * train_data.n_examples

        for i in range(self._epochs):
            for j in range(train_data.n_examples):
                ex_str = f' | {trainx[j]}'
                ex = self._vw.parse(ex_str, labelType=self._vw.lContextualBandit)
                pred = np.array(self._vw.predict(ex))
                self._vw.finish_example(ex)
                assert len(pred.shape) == 1

                pred = pred / pred.sum()
                action = np.random.choice(self._n_classes, p=pred) # 0-indexed action
                prob = pred[action]

                reward = 1 if action == trainy[j] else 0
                cost = -reward

                ex_str = f'{action + 1}:{cost}:{prob} | {trainx[j]}' # 1-indexed action required here
                ex = self._vw.parse(ex_str)
                self._vw.learn(ex)
                self._vw.finish_example(ex)

                reward_progress.append(reward)
                curr_iters = i * train_data.n_examples + j
                every = (total_iters // self._num_tlogs)
                if every == 0:
                    every = 1
                if curr_iters % every == 0:
                    test_acc_progress.append((curr_iters, self.acc(test_data)))

        with open(f'{self.logs_dir}/reward_progress.json', 'w') as f:
            json.dump(reward_progress, f)
        with open(f'{self.logs_dir}/test_acc_progress.json', 'w') as f:
            json.dump(test_acc_progress, f)

        self._plot(reward_progress, test_acc_progress)

    def inference(self, x: np.ndarray) -> np.ndarray:
        assert x.shape[1] == self._n_features

        res = np.zeros(x.shape[0])

        x = VWPredictor.stringify_x(x)
        for i in range(len(x)):
            ex_str = f' | {x[i]}'
            ex = self._vw.parse(ex_str, labelType=self._vw.lContextualBandit)
            pred = np.array(self._vw.predict(ex))
            self._vw.finish_example(ex)
            assert len(pred.shape) == 1
            res[i] = pred.argmax()

        return res

    def raw_inference(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def auc_inference(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def stringify_x(x: np.ndarray) -> List[str]:
        res = [None] * x.shape[0]

        for i, xi in enumerate(x.tolist()):
            res[i] = ' '.join(f'f{eli}:{el}' for eli, el in enumerate(xi))

        return res

    def _validate_data(self, train_data: Dataset[np.ndarray], validn_data: Dataset[np.ndarray], test_data: Dataset[np.ndarray]):
        assert train_data.n_features == self._n_features
        assert validn_data.n_features == self._n_features
        assert test_data.n_features == self._n_features

        assert train_data['y'].dtype in [np.int32, np.int64]
        assert validn_data['y'].dtype in [np.int32, np.int64]
        assert test_data['y'].dtype in [np.int32, np.int64]

        assert train_data['y'].min() == 0
        assert train_data['y'].max() == self._n_classes - 1
        assert validn_data['y'].min() == 0
        assert validn_data['y'].max() == self._n_classes - 1
        assert test_data['y'].min() == 0
        assert test_data['y'].max() == self._n_classes - 1

    def _plot(self, reward_progress: List[float], test_acc_progress: List[Tuple[int, float]]):
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))
        ax = axs[0]
        ax.plot(pd.Series(reward_progress).rolling(500).mean(),  marker='.', markersize=1, linewidth=0.5)
        ax.set_xlabel('#Queries to reward')
        ax.set_ylabel('Rolling mean of reward')
        ax.grid()

        ax = axs[1]
        x, y = tuple(zip(*test_acc_progress))
        ax.plot(x, y, marker='.', markersize=2, linewidth=1)
        ax.set_xlabel('#Queries to reward')
        ax.set_ylabel('Test Accuracy')
        ax.grid()
        plt.tight_layout()
        plt.savefig(f'{self.plots_dir}/progress.png')
        plt.close(fig)