from collections import OrderedDict
from typing import Any, Dict

import numpy as np
from sklearn.linear_model import Ridge

from xcommon import Dataset, LearnablePredictor


class SkLinPredictor(LearnablePredictor[np.ndarray]):
    def __init__(
        self,
        n_features: int,
        n_classes: int,
        **kwargs
    ):
        super().__init__()
        self._kwargs = kwargs
        if 'l2_lambda' not in self._kwargs:
            self._kwargs['l2_lambda'] = 0
        assert 'alpha' not in self._kwargs, 'use l2_lambda instead'

        self._is_classification = n_classes > 0
        assert not self._is_classification

        self._train_invoked = False
        assert 'random_state' not in self._kwargs
        self._kwargs['random_state'] = np.random.randint(low=0, high=10000000)

        self._model = None

    def get_hyperparams(self) -> Dict[str, Any]:
        return OrderedDict(
            [(k, v) for k, v in self._kwargs.items()]
        )

    def train(self, train_data: Dataset[np.ndarray], validn_data: Dataset[np.ndarray], test_data: Dataset[np.ndarray]):
        assert not self._train_invoked, 'train() already invoked once'
        self._train_invoked = True

        self._model = Ridge(alpha=self._kwargs['l2_lambda'] * train_data.n_examples, fit_intercept=True)
        self._model.fit(train_data['x'], train_data['y'])

    def inference(self, x: np.ndarray) -> np.ndarray:
        assert self._model is not None, 'Model doesn\'t exist but inference() called'
        return self._model.predict(x)

    def raw_inference(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def auc_inference(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError
