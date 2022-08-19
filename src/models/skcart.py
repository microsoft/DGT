from collections import OrderedDict
from typing import Any, Dict

import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from xcommon import Dataset, LearnablePredictor


class SkCARTPredictor(LearnablePredictor[np.ndarray]):
    def __init__(
        self,
        n_features: int,
        n_classes: int,
        **kwargs
    ):
        super().__init__()
        self._kwargs = kwargs
        self._actual_height = -1

        self._is_classification = n_classes > 0

        self._train_invoked = False

        assert 'random_state' not in kwargs
        kwargs['random_state'] = np.random.randint(low=0, high=10000000)

        if self._is_classification:
            self._model = DecisionTreeClassifier(**kwargs)
        else:
            self._model = DecisionTreeRegressor(**kwargs)

    def get_hyperparams(self) -> Dict[str, Any]:
        return OrderedDict(
            [('actual_height', self._actual_height)] +
            [(k, v) if k != 'max_depth' else ('height', v) for k, v in self._kwargs.items()]
        )

    def train(self, train_data: Dataset[np.ndarray], validn_data: Dataset[np.ndarray], test_data: Dataset[np.ndarray]):
        assert not self._train_invoked, 'train() already invoked once'
        self._train_invoked = True
        self._model.fit(train_data['x'], train_data['y'])
        self._actual_height = self._model.get_depth()

        try:
            if self._actual_height != self._kwargs['max_depth']:
                self.log_f('WARN: actual depth != max_depth: {} != {}\n'.format(self._actual_height, self._kwargs['max_depth']), stdout=False)
        except KeyError:
            pass

    def inference(self, x: np.ndarray) -> np.ndarray:
        assert self._model is not None, 'Model doesn\'t exist but inference() called'
        return self._model.predict(x)

    def raw_inference(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def auc_inference(self, x: np.ndarray) -> np.ndarray:
        #assert self._model.n_classes_ == [0, 1]
        return self._model.predict_proba(x)[:, 1]