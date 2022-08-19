from typing import Any, Dict, Union

import numpy as np
import torch

from xcommon import Dataset, LearnablePredictor, NTArray
from xconstants import FPTYPE


class FixedConstantPredictor(LearnablePredictor[NTArray]):
    def __init__(
        self,
        n_features: int,
        n_classes: int,
        val: Union[int, float]
    ):
        super().__init__()
        self._is_classification = n_classes > 0
        self._val = val

    def get_hyperparams(self) -> Dict[str, Any]:
        raise NotImplementedError

    def train(self, train_data: Dataset[NTArray], validn_data: Dataset[NTArray], test_data: Dataset[NTArray]):
        raise NotImplementedError

    def inference(self, x: NTArray) -> NTArray:
        if isinstance(x, torch.Tensor):
            self._device0 = x.device
            return torch.full((x.shape[0],), self._val, device=x.device, dtype=torch.int64 if self._is_classification else FPTYPE)
        else:
            return np.full((x.shape[0],), self._val, dtype=np.int64 if self._is_classification else np.float32)

    def raw_inference(self, x: NTArray) -> NTArray:
        raise NotImplementedError

    def auc_inference(self, x: NTArray) -> NTArray:
        raise NotImplementedError
