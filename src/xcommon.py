import functools
import time
import itertools
import json
import math
import multiprocessing
import os
import queue
import copy
import uuid
from abc import ABC, abstractmethod
from typing import (Any, Callable, Dict, Generic, List, Optional,
                    Sequence, Tuple, TypeVar, Union, cast)

from xconstants import REMOTE_SESS
if REMOTE_SESS:
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from xconstants import (CMAP, COMPUTE_BALANCED_ACC, MARKER_SIZE, FPTYPE, PROFILE_TIME,
                        DETERMINISTIC, AccFuncType, REG_ACC, INT_NODE_REPR_MAX_FEATURES)


NTArray = TypeVar('NTArray', np.ndarray, torch.Tensor)

class Logger():
    r"""
    Args:
        random_append: Whether to append a new UUID to file_name
    """
    def __init__(self, file_name: str, random_append=False):
        if random_append:
            append_str = '-{}'.format(uuid.uuid4())
            if '.' in file_name:
                parts = file_name.split('.')
                parts[-2] += append_str
                file_name = '.'.join(parts)
            else:
                file_name += append_str
        self._f = open(file_name, 'a', buffering=1)

    def log(self, value: str, stdout=True):
        self._f.write(value)
        if stdout:
            print(value, sep='', end='', flush=True)

    def close(self):
        self._f.close()

class Utils():
    cmlog: Callable[[str, bool], None] = lambda s, stdout=True: print(s) if stdout else None

    r"""
    Args:
        matrix: shape=(nrows, ncols)
    Returns:
        *: np.shape=(matrix.shape[0],), containing number of occurences
            of num in each row
    """
    @staticmethod
    def count(matrix: np.ndarray, num: Union[int, float]) -> np.ndarray:
        assert len(matrix.shape) == 2
        return np.apply_along_axis(lambda row: np.count_nonzero(row == num), 1, matrix)

    @staticmethod
    def sign(num: Union[int, float]) -> str:
        return '+' if num >= 0 else '-'

    @staticmethod
    def ceil(a: int, b: int) -> int:
        return a // b if a % b == 0 else (a // b) + 1

    @staticmethod
    def get_pts(axes_range: Tuple[int, int]) -> np.ndarray:
        axes_range = (-3, 3)
        num_pts = 1e5
        num_pts_per_axis = int(math.sqrt(num_pts))
        pts = np.array(list(itertools.product(
            np.linspace(axes_range[0], axes_range[1], num_pts_per_axis),
            np.linspace(axes_range[0], axes_range[1], num_pts_per_axis)))
        )
        return pts

    @staticmethod
    def get_padded_text(phrase: str, line_len: int=120, left_padding: str='=', right_padding: str='=') -> str:
        rem_len = max(0, line_len - len(phrase) - 2)
        left_padding = left_padding * (rem_len // 2)
        right_padding = right_padding * (rem_len // 2)
        return '{} {} {}'.format(left_padding, phrase, right_padding)

    @staticmethod
    def get_initialized_bias(
        in_features: int, out_features: int, initialization_mean: Optional[torch.Tensor]=None
    ) -> nn.Parameter:

        if initialization_mean is None:
            initialization_mean = torch.zeros((out_features,)).float()
        assert initialization_mean.shape == (out_features,)

        k = 1 / math.sqrt(in_features)
        lb = initialization_mean - k
        ub = initialization_mean + k
        init_val = torch.distributions.uniform.Uniform(lb, ub).sample().float() # type: ignore

        return nn.Parameter(init_val, requires_grad=True) # type: ignore

    """
    interval: Used when category=='minmax'. (lower bound, upper bound)
    mirror_params: Used when category=='mirror'. what to (subtract by, divide by, multiply by, add by)
    """
    @staticmethod
    def normalize(
        x: np.ndarray,
        category='zscore',
        interval: Optional[Tuple[float, float]]=None,
        mirror_params: Optional[Tuple[Union[np.ndarray, int, float], Union[np.ndarray, int, float], Union[np.ndarray, int, float], Union[np.ndarray, int, float]]]=None
    ) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
        assert len(x.shape) <= 2
        if category == 'minmax' and len(x.shape) == 1:
            Utils.cmlog('\nmax: {}, min: {}, max-min: {}\n'.format(x.max(), x.min(), x.max() - x.min()))

        if category == 'zscore':
            mean = x.mean(axis=0)
            std = x.std(axis=0)
            if isinstance(std, np.ndarray):
                std[std == 0] = 1
            x = (x - mean) / std
            return x, (mean, std, 1, 0)

        elif category == 'minmax':
            assert interval is not None
            mn, mx = x.min(axis=0), x.max(axis=0)
            div = mx - mn
            if isinstance(div, np.ndarray):
                div[div == 0] = 1
            x = (x - mn) / div
            x *= (interval[1] - interval[0])
            x += interval[0]
            return x, (mn, div, interval[1] - interval[0], interval[0])

        elif category == 'mirror':
            assert mirror_params is not None
            sub, div, mul, add = mirror_params
            x = (x - sub) / div
            x *= mul
            x += add
            return x, mirror_params

        else:
            raise ValueError('category must be in ["zscore", "minmax", "mirror"]')

    @staticmethod
    def denormalize_acc(
        acc: float,
        acc_func_type: AccFuncType,
        mirror_y_params: Optional[Tuple[float, float, float, float]]
    ) -> Tuple[float, str]:

        if mirror_y_params is None:
            return acc, 'No denormalization done: no normalization is applied on target'
        else:
            sub, div, mul, add = mirror_y_params
            fact = (div / mul)

            if acc_func_type is AccFuncType.acc:
                return acc, 'No denormalization done: accuracy doesn\'t require it'
            if acc_func_type is AccFuncType.mse:
                return acc * (fact * fact), 'Multiplying MSE by {}'.format(fact * fact)
            elif acc_func_type is AccFuncType.rmse:
                return acc * fact, 'Multiplying RMSE by {}'.format(fact)
            else:
                return acc, 'No denormalization done: structure of criterion is unknown'

    @staticmethod
    def gen_dist(sigma: float, size: int, orig_size: int=100000):
        y = np.random.normal(0, sigma, size=orig_size)
        y = y[(y > 0) & (y <= 0.5)]
        np.random.shuffle(y)
        y = y[:size]
        return y

    @staticmethod
    def safe_subscript(arr: Sequence, idx: int):
        return arr[max(0, min(idx, len(arr) - 1))]

    @staticmethod
    def sigmoid(x: float) -> float:
        return 1 / (1 + math.exp(-x))

    @staticmethod
    def unit_dec(x: float, l: float) -> float:
        return 2 * (1 - Utils.sigmoid(l * x))

    @staticmethod
    def best_fit_slope(y: np.ndarray, x: Optional[np.ndarray]=None) -> float:
        if x is None:
            x = np.arange(0, len(y))
        return np.polynomial.polynomial.Polynomial.fit(x, y, 1).convert().coef[1]

    @staticmethod
    def shorten_mid(s: str, begin_keep: int, end_keep: int) -> str:
        if begin_keep + end_keep >= len(s):
            return s
        return s[:begin_keep] + s[-end_keep:]

    @staticmethod
    def inv_sigmoid(x: float) -> float:
        return math.log(x / (1 - x))

    @staticmethod
    def is_binary_labels(y: np.ndarray) -> bool:
        assert len(y.shape) == 1

        return ((y.dtype in [np.int16, np.int32, np.int64]) and
            (np.array_equal(np.unique(y), np.array([0, 1])) or
            np.array_equal(np.unique(y), np.array([0])) or
            np.array_equal(np.unique(y), np.array([1]))))

    @staticmethod
    def is_classes(y: Union[torch.Tensor, np.ndarray]) -> bool:
        assert len(y.shape) == 1

        try:
            return y.dtype in [np.int16, np.int32, np.int64, torch.int16, torch.int32, torch.int64]
        except TypeError:
            return False

    @staticmethod
    def safe_log(s: str, path: str, l: 'Optional[multiprocessing.synchronize.Lock]'=None, stdout: bool=True):
        if l is not None:
            l.acquire()

        with open(path, 'a', buffering=1) as f:
            f.write(s)
        if stdout:
            print(s, sep='', end='', flush=True)

        if l is not None:
            l.release()

    @staticmethod
    def model_to_vector(model: nn.Module):
        vec = []
        for param in model.parameters():
            if param.requires_grad:
                vec.append(param.view(-1))
        return torch.cat(vec)

    @staticmethod
    def vector_to_model(vec: torch.Tensor, model: nn.Module, replace_grad: bool=False):
        pointer = 0
        for param in model.parameters():
            if param.requires_grad:
                # The length of the parameter
                num_param = param.numel()

                if replace_grad:
                    # Slice the vector, reshape it, and replace the old data of the parameter
                    param.grad = vec[pointer:pointer + num_param].view_as(param).data
                else:
                    # Slice the vector, reshape it, and replace the old data of the parameter
                    param.data = vec[pointer:pointer + num_param].view_as(param).data

                # Increment the pointer
                pointer += num_param

    @staticmethod
    def huber_loss(y_pred: Union[torch.Tensor, np.ndarray], y_targ: Union[torch.Tensor, np.ndarray], delta: float=0.2) -> Union[torch.Tensor, np.ndarray]:
        if isinstance(y_pred, np.ndarray):
            assert len(y_pred.shape) == 1
            assert len(y_targ.shape) == 1

            l1 = 0.5 * ((y_pred - y_targ) ** 2)
            l2 = delta * np.abs(y_pred - y_targ) - 0.5 * (delta ** 2)
            sel = (np.abs(y_pred - y_targ) <= delta)
            return sel.astype(np.int) * l1 + (~sel).astype(np.int) * l2

        else:
            assert len(y_pred.shape) == 1
            assert len(y_targ.shape) == 1

            l1 = 0.5 * ((y_pred - y_targ) ** 2)
            l2 = delta * (y_pred - y_targ).abs() - 0.5 * (delta ** 2)
            sel = (y_pred - y_targ).abs() <= delta
            return sel.to(dtype=y_pred.dtype) * l1 + sel.bitwise_not().to(dtype=y_pred.dtype) * l2

    @staticmethod
    def vec_check(a, b, epoch, batch):
        assert torch.equal(a, b), 'epoch={}, batch={}\na={}\nb={}\nd={}\n'.format(epoch, batch, a, b, torch.eq(a, b))

    @staticmethod
    def write_json(obj: Dict, path: str):
        with open(path, 'w') as f:
            if 'black_box' in obj:
                cobj = copy.deepcopy(obj)
                cobj['black_box'] = str(cobj['black_box'])
                json.dump(cobj, f)
            else:
                json.dump(obj, f)

    @staticmethod
    def read_json(path: str) -> Dict:
        with open(path, 'r') as f:
            return json.load(f)

    @staticmethod
    def get_exp_dir(config_dir: str) -> str:
        return os.path.split(os.path.split(config_dir)[0])[0]

    @staticmethod
    def get_dict_combinations(d: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        keys = list(d.keys())
        result: List[Dict[str, Any]] = []
        for combination in itertools.product(*d.values()): # type: ignore
            result.append({})
            for i, val in enumerate(combination):
                result[-1][keys[i]] = val
        return result

    @staticmethod
    def get_list_dict_combinations(ld: List[Dict[str, List[Any]]]) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        for d in ld:
            result.extend(Utils.get_dict_combinations(d))
        return result

    @staticmethod
    def get_subtree_idcs(height: int) -> Tuple[List[int], List[int]]:
        left_idcs = []
        right_idcs = []
        for h in range(1, height + 1):
            idcs = list(range(2 ** h - 1, 2 ** (h + 1) - 1))
            left_idcs.extend(idcs[: 2 ** (h - 1)])
            right_idcs.extend(idcs[2 ** (h - 1): ])
        return left_idcs, right_idcs

    """
        start: threshold at epoch=0
        segments: (epoch, threshold)
    """
    @staticmethod
    def get_curve(start: float, segments: List[Tuple[int, float]]) -> List[float]:
        segments = [(0, start)] + segments
        curve: List[float] = []
        for i in range(1, len(segments)):
            curve.extend(np.linspace(segments[i - 1][1], segments[i][1], segments[i][0] - segments[i - 1][0]).tolist())
        return curve

    """
    arr: (n, y)
    val: (n,)
    """
    @staticmethod
    def append_col(arr: np.ndarray, val: Union[int, float, np.ndarray]) -> np.ndarray:
        new_arr = np.empty((arr.shape[0], arr.shape[1] + 1)) if isinstance(arr, torch.Tensor) else np.empty((arr.shape[0], arr.shape[1] + 1))
        new_arr[:, :-1] = arr
        new_arr[:, -1] = val
        return new_arr

    @staticmethod
    def get_device_num(arr: torch.Tensor) -> int:
        try:
            num = arr.get_device()
        except RuntimeError as e:
            if 'get_device is not implemented' in str(e):
                num = -1 # cpu
            else:
                raise e
        return num

    """
    Get the definition of acc() - what calling model.acc() should do
    """
    @staticmethod
    def get_acc_def(is_classification_dataset: bool, criterion: Optional[Callable]=None) -> Tuple[Callable, AccFuncType]:
        if is_classification_dataset:
            acc_func = cast(Callable[[NTArray, NTArray], float], functools.partial(Predictor.zo_accuracy_p, itemize=True))
            acc_func_type = AccFuncType.acc
        else:
            if REG_ACC is AccFuncType.mse:
                acc_func = cast(Callable[[NTArray, NTArray], float], functools.partial(Predictor.mse, itemize=True))
            elif REG_ACC is AccFuncType.rmse:
                acc_func = cast(Callable[[NTArray, NTArray], float], functools.partial(Predictor.rmse, itemize=True))
            elif REG_ACC is AccFuncType.criterion:
                acc_func = lambda y_pred, y_targ: float(criterion(y_pred, y_targ).mean())
            else:
                raise ValueError('REG_ACC must be in ["mse", "rmse", "criterion"]')
            acc_func_type = REG_ACC

        return acc_func, acc_func_type

    @staticmethod
    def get_cart_wts(model: Union[DecisionTreeRegressor, DecisionTreeClassifier], max_depth: int, n_features: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, int]]:
        is_classifier = isinstance(model, DecisionTreeClassifier)
        weights = np.zeros((2 ** max_depth - 1, n_features), dtype=np.float32)
        biases = np.zeros((2 ** max_depth - 1,), dtype=np.float32)
        labels = np.zeros(2 ** max_depth, dtype=np.int32 if is_classifier else np.float32)
        leaf_map = {} # map leaf index in full tree (0-indexed) to leaf index (not 0-indexed) in CART tree

        cl = model.tree_.children_left
        cr = model.tree_.children_right
        f = model.tree_.feature
        t = model.tree_.threshold
        v = model.tree_.value

        newv = []
        if is_classifier:
            for vidx in range(v.shape[0]):
                newv.append(model.classes_[np.argmax(v[vidx][0])])
        else:
            for vidx in range(v.shape[0]):
                newv.append(v[vidx][0][0])
        v = newv

        assert np.array_equal(cl == -1, cr == -1)

        q = queue.Queue()
        q.put({
            'midx': 0, # index of node in model array
            'pidx': 0, # index of node in predicate array
            'depth': 0, # depth of node
            'lidx': -1, # for fake filler nodes (valid when midx == -1): the leaf that they replace
        })
        while not q.empty():
            node = q.get()
            midx, pidx, depth, lidx = node['midx'], node['pidx'], node['depth'], node['lidx']

            if depth == max_depth:
                # we are at leaf, no need to explore further, save labels
                vidx = lidx if midx == -1 else midx
                labels[pidx - (2 ** max_depth - 1)] = v[vidx]
                leaf_map[pidx - (2 ** max_depth - 1)] = vidx
            else:
                # need to explore further
                if midx != -1:
                    weights[pidx][f[midx]] = -1
                    biases[pidx] = t[midx]

                if midx == -1:
                    q.put({'midx': -1, 'pidx': 2 * pidx + 1, 'depth': depth + 1, 'lidx': lidx})
                    q.put({'midx': -1, 'pidx': 2 * pidx + 2, 'depth': depth + 1, 'lidx': lidx})
                else:
                    # any children that are leaf, need fake filler nodes
                    if (cl[midx] == -1) or (cr[midx] == -1):
                        assert cl[midx] == cr[midx]
                    if cl[midx] == -1:
                        q.put({'midx': -1, 'pidx': 2 * pidx + 1, 'depth': depth + 1, 'lidx': midx})
                    else:
                        q.put({'midx': cl[midx], 'pidx': 2 * pidx + 1, 'depth': depth + 1, 'lidx': -1})

                    if cr[midx] == -1:
                        q.put({'midx': -1, 'pidx': 2 * pidx + 2, 'depth': depth + 1, 'lidx': midx})
                    else:
                        q.put({'midx': cr[midx], 'pidx': 2 * pidx + 2, 'depth': depth + 1, 'lidx': -1})

        labels = np.array(labels)
        return weights, biases, labels, leaf_map

    @staticmethod
    def append_linedict_to_csv(d: Dict[str, Any], path: str):
        if os.path.isfile(path):
            df = pd.read_csv(path)
            df = df.append(d, ignore_index=True)
        else:
            df = pd.DataFrame([d], columns=d.keys())
        cast(pd.DataFrame, df).to_csv(path, index=False)

    @staticmethod
    def join_datasets(d1: 'Dataset[np.ndarray]', d2: 'Dataset[np.ndarray]') -> 'Dataset[np.ndarray]':
        assert isinstance(d1['x'], np.ndarray)
        assert d1.name == d2.name
        assert d1.shuffle_seed == d2.shuffle_seed
        assert d1.n_features == d2.n_features

        # assert d1.normalize_x_kwargs == d2.normalize_x_kwargs, (d1.normalize_x_kwargs, d2.normalize_x_kwargs)
        # assert d1.normalize_y_kwargs == d2.normalize_y_kwargs
        assert d1.mirror_x_params == d2.mirror_x_params
        assert d1.mirror_y_params == d2.mirror_y_params

        return Dataset(
            x=np.concatenate((d1['x'], d2['x'])),
            y=np.concatenate((d1['y'], d2['y'])),
            name=d1.name,
            copy=True,
            autoshrink_y=False,
            normalize_x_kwargs=d1.normalize_x_kwargs,
            normalize_y_kwargs=d1.normalize_y_kwargs,
            mirror_x_params=d1.mirror_x_params,
            mirror_y_params=d1.mirror_y_params,
            shuffle_seed=d1.shuffle_seed
        )

    @staticmethod
    def tostr(arr: np.ndarray) -> str:
        if len(arr.shape) == 1:
            ret = str(arr.tolist())

        else:
            assert len(arr.shape) == 2
            nrows = arr.shape[0]
            arr = arr.tolist()

            ret = '[\n'
            for i in range(nrows):
                ret += f' {arr[i]}\n'
            ret += ']'

        return ret

class Batcher(Generic[NTArray]):
    r"""
    Args:
        batch_size: Size of each batch
        arrs: Arrays to apply batching on
        shuffle: Whether to shuffle at the beginning of each cycle
        shuffle_seed: Seed to use to shuffle
    """
    def __init__(self, batch_size: int, *arrs: NTArray, shuffle: bool=True, shuffle_seed: Optional[int]=None):
        # Sanity checks
        assert len(arrs) > 0
        n = arrs[0].shape[0]
        for arr in arrs:
            assert arr.shape[0] == n
        if batch_size > n:
            Utils.cmlog('WARN: batch_size ({}) > n ({}). Setting batch_size to n ({})\n'.format(batch_size, n, n), stdout=False)
            batch_size = n
        self._n = n

        self._batch_size = batch_size
        self._steps_per_epoch: int = n // batch_size + (0 if n%batch_size==0 else 1)
        self._arrs: List[NTArray] = list(arrs)
        self._shuffle = shuffle
        self._curr = 0
        self._rs = np.random.RandomState(seed=shuffle_seed)
        if self._shuffle:
            self._shuffle_arrs()

    r"""
    Returns true if the next batch is part of a new cycle
    """
    def is_new_cycle(self) -> bool:
        return self._curr == 0

    def next(self) -> Union[Tuple[NTArray, ...], NTArray]:
        ret = tuple([arr[self._curr: self._curr + self._batch_size] for arr in self._arrs])

        self._curr = self._curr + self._batch_size
        if self._curr >= self._n:
            self._curr = 0
            if self._shuffle:
                self._shuffle_arrs()

        if len(ret) == 1:
            return ret[0]
        return ret

    def _shuffle_arrs(self):
        new_idx = self._rs.permutation(self._n)
        for i, _ in enumerate(self._arrs):
            self._arrs[i] = self._arrs[i][new_idx]

class Dataset(Generic[NTArray]):
    r"""
    Args:
        x: shape=(n_examples, n_features)
        y: shape=(n_examples,) or (n_examples, n_labels), if n_labels=1 converts ndim to 1 when autoshrink_y=True
        name: Name of the dataset

    Properties:
        n_examples: int
        n_features: int
        n_labels: int
        name: str

    Notes:
        Dataset will reside on same device as tensor even when copy=True
    """
    def __init__(
        self, x: NTArray,
        y: NTArray,
        name: str='',
        copy: bool=True,
        autoshrink_y=False,

        # metadata reg. how this dataset was normalized
        normalize_x_kwargs: Optional[Dict[str, Any]]=None,
        normalize_y_kwargs: Optional[Dict[str, Any]]=None,
        mirror_x_params: Optional[Tuple[float, float, float, float]]=None,
        mirror_y_params: Optional[Tuple[float, float, float, float]]=None,

        shuffle_seed: int=-100
    ):
        assert len(x.shape) == 2 # type: ignore
        assert len(y.shape) == 1 or len(y.shape) == 2 # type: ignore
        assert x.shape[0] == y.shape[0]

        self.name = name
        self.shuffle_seed = shuffle_seed
        self._x: NTArray
        self._y: NTArray
        if copy:
            if isinstance(x, torch.Tensor):
                self._x = x.detach().clone()
                self._y = y.detach().clone()
            else:
                self._x = cast(np.ndarray, x).copy()
                self._y = cast(np.ndarray, y).copy()
        else:
            self._x = x
            self._y = y

        self.n_examples, self.n_features = self._x.shape
        if len(self._y.shape) == 1: # type: ignore
            self.n_labels = 1
        else:
            self.n_labels = self._y.shape[1]
            if autoshrink_y and self.n_labels == 1:
                self._y = self._y.squeeze(1)

        # metadata
        self.normalize_x_kwargs = normalize_x_kwargs
        self.normalize_y_kwargs = normalize_y_kwargs
        self.mirror_x_params = mirror_x_params
        self.mirror_y_params = mirror_y_params

    def __repr__(self) -> str:
        return 'x: {}\ny: {}'.format(self._x.__repr__(), self._y.__repr__())

    def __str__(self) -> str:
        return 'x: {}\ny: {}'.format(self._x.__str__(), self._y.__str__())

    def __getitem__(self, key: Union[str, int, slice]) -> Union[NTArray, 'Dataset[NTArray]']:
        if isinstance(key, tuple):
            raise ValueError('Multidimensional indexing not allowed')

        if isinstance(key, str):
            assert key in ['x', 'y']
            return self._x if key == 'x' else self._y
        elif isinstance(key, int):
            return Dataset(self._x[key: key + 1], self._y[key: key + 1], name=self.name, copy=False, shuffle_seed=self.shuffle_seed, normalize_x_kwargs=self.normalize_x_kwargs, normalize_y_kwargs=self.normalize_y_kwargs, mirror_x_params=self.mirror_x_params, mirror_y_params=self.mirror_y_params)
        elif isinstance(key, slice):
            return Dataset(self._x[key], self._y[key], name=self.name, copy=False, shuffle_seed=self.shuffle_seed, normalize_x_kwargs=self.normalize_x_kwargs, normalize_y_kwargs=self.normalize_y_kwargs, mirror_x_params=self.mirror_x_params, mirror_y_params=self.mirror_y_params)
        else:
            raise ValueError('Indexing should be done either using a str, int or slice')

    def shuffle(self, seed: Optional[int]=None) -> 'Dataset[NTArray]':
        if seed is None:
            idx = np.random.permutation(self.n_examples)
        else:
            idx = np.random.RandomState(seed=seed).permutation(self.n_examples)
        return Dataset(self._x[idx], self._y[idx], name=self.name, copy=False, shuffle_seed=self.shuffle_seed, normalize_x_kwargs=self.normalize_x_kwargs, normalize_y_kwargs=self.normalize_y_kwargs, mirror_x_params=self.mirror_x_params, mirror_y_params=self.mirror_y_params)

    def to_device(self, device: torch.device) -> 'Dataset[torch.Tensor]':
        assert isinstance(self._x, torch.Tensor)
        return Dataset(self._x.to(device), self._y.to(device), name=self.name, copy=False, shuffle_seed=self.shuffle_seed, normalize_x_kwargs=self.normalize_x_kwargs, normalize_y_kwargs=self.normalize_y_kwargs, mirror_x_params=self.mirror_x_params, mirror_y_params=self.mirror_y_params)

    def copy(self) -> 'Dataset[NTArray]':
        return Dataset(self._x, self._y, name=self.name, copy=True, shuffle_seed=self.shuffle_seed, normalize_x_kwargs=self.normalize_x_kwargs, normalize_y_kwargs=self.normalize_y_kwargs, mirror_x_params=self.mirror_x_params, mirror_y_params=self.mirror_y_params)

    def to_type(self, dtype, dim: str) -> 'Dataset[NTArray]':
        assert dim in ['x', 'y', 'xy']
        if isinstance(self._x, torch.Tensor):
            x = self._x.to(dtype) if dim == 'x' or dim == 'xy' else self._x
            y = self._y.to(dtype) if dim == 'y' or dim == 'xy' else self._y
        else:
            x = self._x.astype(dtype) if dim == 'x' or dim == 'xy' else self._x
            y = self._y.astype(dtype) if dim == 'y' or dim == 'xy' else self._y
        return Dataset(x, y, name=self.name, copy=False, shuffle_seed=self.shuffle_seed, normalize_x_kwargs=self.normalize_x_kwargs, normalize_y_kwargs=self.normalize_y_kwargs, mirror_x_params=self.mirror_x_params, mirror_y_params=self.mirror_y_params)

    def to_tensor(self) -> 'Dataset[torch.Tensor]':
        if isinstance(self._x, np.ndarray):
            return Dataset(torch.from_numpy(self._x), torch.from_numpy(self._y), name=self.name, copy=False, shuffle_seed=self.shuffle_seed, normalize_x_kwargs=self.normalize_x_kwargs, normalize_y_kwargs=self.normalize_y_kwargs, mirror_x_params=self.mirror_x_params, mirror_y_params=self.mirror_y_params)
        else:
            return self

    def to_ndarray(self) -> 'Dataset[np.ndarray]':
        if isinstance(self._x, torch.Tensor):
            return Dataset(
                self._x.detach().clone().to('cpu').numpy(),
                self._y.detach().clone().to('cpu').numpy(),
                name=self.name, copy=False, shuffle_seed=self.shuffle_seed, normalize_x_kwargs=self.normalize_x_kwargs, normalize_y_kwargs=self.normalize_y_kwargs, mirror_x_params=self.mirror_x_params, mirror_y_params=self.mirror_y_params)
        else:
            return self

    def split(
        self, ratio1: float, ratio2: float=0
    ) -> (
        Union[Tuple['Dataset[NTArray]', 'Dataset[NTArray]'],
        Tuple['Dataset[NTArray]', 'Dataset[NTArray]', 'Dataset[NTArray]']]
    ):
        assert ratio1 <= 1
        assert ratio2 <= 1
        assert ratio1 + ratio2 <= 1

        n1 = int(ratio1 * self.n_examples)
        n2 = int(ratio2 * self.n_examples)

        if ratio2 == 0:
            return cast(Dataset[NTArray], self[:n1]), cast(Dataset[NTArray], self[n1:])
        else:
            return (
                cast(Dataset[NTArray], self[:n1]),
                cast(Dataset[NTArray], self[n1: n1 + n2]),
                cast(Dataset[NTArray], self[n1 + n2:])
            )

    def normalize_x(self, **normalize_kwargs) -> 'Dataset[NTArray]':
        x, mirror_params = Utils.normalize(self._x, **normalize_kwargs)
        d = Dataset(x, self._y, name=self.name, copy=False, shuffle_seed=self.shuffle_seed, normalize_x_kwargs=self.normalize_x_kwargs, normalize_y_kwargs=self.normalize_y_kwargs, mirror_x_params=self.mirror_x_params, mirror_y_params=self.mirror_y_params)
        d.normalize_x_kwargs = normalize_kwargs
        d.mirror_x_params = mirror_params
        return d

    def normalize_y(self, **normalize_kwargs) -> 'Dataset[NTArray]':
        y, mirror_params = Utils.normalize(self._y, **normalize_kwargs)
        d = Dataset(self._x, y, name=self.name, copy=False, shuffle_seed=self.shuffle_seed, normalize_x_kwargs=self.normalize_x_kwargs, normalize_y_kwargs=self.normalize_y_kwargs, mirror_x_params=self.mirror_x_params, mirror_y_params=self.mirror_y_params)
        d.normalize_y_kwargs = normalize_kwargs
        d.mirror_y_params = mirror_params
        return d

    r"""
    Precondition: n_features == 2 and n_labels == 1
    """
    def visualize(self, title='', save_path=''):
        if self.n_features != 2:
            Utils.cmlog('WARN: Ignoring Dataset.visualize(): self.n_features != 2\n', stdout=False)
            return
        if self.n_labels != 1:
            Utils.cmlog('WARN: Ignoring Dataset.visualize(): self.n_labels != 1\n', stdout=False)
            return

        _, ax = plt.subplots()
        cmap = plt.cm.get_cmap(CMAP)
        ax.set_title(title)
        if len(self._y.shape) == 1: # type: ignore
            sc = ax.scatter(self._x[:, 0], self._x[:, 1], c=self._y, cmap=cmap, s=MARKER_SIZE)
        else:
            sc = ax.scatter(self._x[:, 0], self._x[:, 1], c=self._y.squeeze(1), cmap=cmap, s=MARKER_SIZE)
        plt.colorbar(sc)

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def get_stats(self, title='') -> str:
        stats = f' {self.name}-{self.shuffle_seed}: {title}\n'
        sep = '-' * max(len(stats) + 2, 25)

        stats = sep + '\n' + stats + sep + '\n'
        stats += ' n_examples: {}, '.format(self.n_examples)
        stats += ' n_features: {}, '.format(self.n_features)
        stats += ' n_labels  : {}\n'.format(self.n_labels)

        if Utils.is_classes(self._y):
            label_dist = ', '.join(['{:>2}: {}'.format(val, cnt) for val, cnt in zip(*np.unique(self._y, return_counts=True))])
        else:
            label_dist = ', '.join(['{:.4f}: {}'.format(edge, cnt) for cnt, edge in zip(*np.histogram(self._y, range=(0, 1), bins=10, density=False))])
        stats += ' Label distribution: {}\n'.format(label_dist)

        stats += sep + '\n'
        return stats

class Predictor(ABC, Generic[NTArray]):
    def __init__(self):
        super().__init__()
        self._is_classification: bool = True
        self._device0: Optional[torch.device] = None
        self._is_pure_dt: bool = False
        self._use_lforb: bool = False

        self.acc_func: Optional[Callable[[NTArray, NTArray], float]] = None
        self.acc_func_type: Optional[AccFuncType] = None
        self.log_f = lambda s, stdout=True: print(s) if stdout else None # type: ignore
        self.logs_dir: str = '.'
        self.plots_dir: str = '.'

    r"""
    Args:
        x: shape=(n_samples, n_features)

    Returns:
        *: shape= (n_samples,)
        For classification: each number is [0..num_classes - 1]
        For regression: each number is in Real space

    Notes:
        Shouldn't record/propagate gradients (i.e. use torch.no_grad)
    """
    @abstractmethod
    def inference(self, x: NTArray) -> NTArray:
        raise NotImplementedError

    r"""
    Args:
        x: shape=(n_samples, n_features)

    Returns:
        *: shape= (n_samples, *)

    Notes:
        Typically for regression this is the same as inference
        Shouldn't record/propagate gradients (i.e. use torch.no_grad)
    """
    @abstractmethod
    def raw_inference(self, x: NTArray) -> NTArray:
        raise NotImplementedError

    r"""
    Args:
        x: shape=(n_samples, n_features)

    Returns:
        *: shape= (n_samples,)
            Score of the class with the greater label

    Notes:
        Shouldn't record/propagate gradients (i.e. use torch.no_grad)
        Will be called only when valid
    """
    @abstractmethod
    def auc_inference(self, x: NTArray) -> NTArray:
        raise NotImplementedError

    r"""
    Notes:
        Typically used for classification

        itemize allows this function to be used not just at the level of user (along with loss, acc etc.) but also
        at a lower level, for instance inside train() where gradients are required to be present. In this case,
        y_pred would have tracked gradients.
    """
    @staticmethod
    def zo_accuracy(y_pred: NTArray, y_targ: NTArray, itemize: bool=True) -> Union[torch.Tensor, float]:
        assert type(y_pred) == type(y_targ)
        assert len(y_pred.shape) == 1
        assert len(y_targ.shape) == 1

        if COMPUTE_BALANCED_ACC:
            assert len(y_pred.shape) == 1
            assert len(y_targ.shape) == 1

            if isinstance(y_pred, torch.Tensor):
                pos = len(y_targ.nonzero().flatten())
                neg = len(y_targ) - pos # type: ignore
                tp = len((y_targ * y_pred).nonzero().flatten())
                tn = len(((1 - y_targ) * (1 - y_pred)).nonzero().flatten()) # type: ignore
            else:
                pos = len(y_targ.nonzero()[0])
                neg = len(y_targ) - pos
                tp = len((y_targ * y_pred).nonzero()[0])
                tn = len(((1 - y_targ) * (1 - y_pred)).nonzero()[0])

            if pos == 0:
                return tn / neg
            elif neg == 0:
                return tp / pos
            else:
                return (tp / pos + tn / neg) / 2

        else:
            if isinstance(y_pred, torch.Tensor):
                ret = torch.mean((y_pred == y_targ).float())
                if itemize:
                    return cast(float, ret.item()) # mean will be a float
                else:
                    return ret
            else:
                return np.mean(y_pred == y_targ)

    r"""
    Notes:
        Typically used for classification

        itemize allows this function to be used not just at the level of user (along with loss, acc etc.) but also
        at a lower level, for instance inside train() where gradients are required to be present. In this case,
        y_pred would have tracked gradients.
    """
    @staticmethod
    def zo_accuracy_p(y_pred: NTArray, y_targ: NTArray, itemize: bool=True) -> Union[torch.Tensor, float]:
        return Predictor.zo_accuracy(y_pred, y_targ, itemize) * 100

    r"""
    Notes:
        Typically used for regression

        itemize allows this function to be used not just at the level of user (along with loss, acc etc.) but also
        at a lower level, for instance inside train() where gradients are required to be present. In this case,
        y_pred would have tracked gradients.
    """
    @staticmethod
    def mse(y_pred: NTArray, y_targ: NTArray, itemize: bool=True) -> Union[torch.Tensor, float]:
        assert type(y_pred) == type(y_targ), '{} != {}'.format(type(y_pred), type(y_targ))
        assert len(y_pred.shape) == 1
        assert len(y_targ.shape) == 1

        if isinstance(y_pred, torch.Tensor):
            ret = torch.mean(((y_pred - y_targ) ** 2))
            if itemize:
                return cast(float, ret.item()) # mean will be a float
            return ret
        else:
            return np.mean((y_pred - y_targ) ** 2)

    r"""
    Notes:
        Typically used for regression

        itemize allows this function to be used not just at the level of user (along with loss, acc etc.) but also
        at a lower level, for instance inside train() where gradients are required to be present. In this case,
        y_pred would have tracked gradients.
    """
    @staticmethod
    def rmse(y_pred: NTArray, y_targ: NTArray, itemize: bool=True) -> Union[torch.Tensor, float]:
        ret = Predictor.mse(y_pred, y_targ, itemize)
        if isinstance(ret, torch.Tensor):
            return torch.sqrt(ret)
        else:
            return math.sqrt(ret)

    @torch.no_grad()
    def _test(
        self, data: Dataset[NTArray], use_raw_inference: bool,
        loss_func: Callable[[NTArray, NTArray], float]
    ) -> float:
        assert len(data['y'].shape) == 1

        batcher = Batcher(8192, data['x'], data['y'], shuffle=False)
        y_pred = []
        while True:
            x,y = batcher.next()
            if use_raw_inference:
                y_pred.append(self.raw_inference(cast(NTArray, x)))
            else:
                y_pred.append(self.inference(cast(NTArray, x)))
            if batcher.is_new_cycle():
                break


        if self._device0 is not None:
            y_targ = cast(torch.Tensor, data['y']).to(self._device0)
            y_pred = torch.cat(y_pred, 0)
        else:
            y_targ = cast(NTArray, data['y'])
            y_pred = np.concatenate(y_pred, 0)

        if hasattr(self, '_fp_type') and (not self._is_classification):
            y_targ = y_targ.to(dtype=self._fp_type)

        return loss_func(y_pred, y_targ)

    r"""
    Uses raw_inference() while acc() uses inference()

    Args:
        data: Test data
            (each number in data['y'] is [0..num_classes - 1] if classification or R if regression)
        loss_func: Returns loss given (raw_inference() output, actual labels data['y'])
    Returns:
        *: Average (over n_examples) loss
    """
    def loss(self, data: Dataset[NTArray], loss_func: Callable[[NTArray, NTArray], float]) -> float:
        return self._test(data, use_raw_inference=True, loss_func=loss_func)

    r"""
    Uses inference() while loss() uses raw_inference()
    By default uses info about
        - how data is normalized
        - what acc_func is
    to try to denormalize the value computed

    Args:
        data: Test data
            (each number in data['y'] is [0..num_classes - 1] if classification or R if regression)
    Returns:
        *: Average (over n_examples) loss
    """
    def acc(self, data: Dataset[NTArray], denormalize: bool=True) -> float:
        acc_func = self.acc_func
        acc_func_type = self.acc_func_type

        if acc_func is None:
            if self._is_classification:
                acc_func = cast(Callable[[NTArray, NTArray], float], functools.partial(self.zo_accuracy_p, itemize=True))
                acc_func_type = AccFuncType.acc
            else:
                acc_func = cast(Callable[[NTArray, NTArray], float], functools.partial(self.rmse, itemize=True))
                acc_func_type = AccFuncType.rmse

        acc = self._test(data, use_raw_inference=False, loss_func=acc_func)
        if denormalize:
            acc, _ = Utils.denormalize_acc(acc, acc_func_type, data.mirror_y_params)
        return acc

    r"""
    Args:
        data: Test data
            (each number in data['y'] is [0..num_classes - 1])
    Returns:
        *: ROC AUC score
    """
    def auc(self, data: Dataset[NTArray]) -> float:
        assert self._is_classification, 'ROC AUC valid only for classification'
        assert data.n_labels == 1

        y_targ = data['y']
        if isinstance(y_targ, torch.Tensor):
            y_targ = y_targ.to('cpu').numpy()
        assert set(cast(np.ndarray, y_targ)) == set((0, 1)), 'ROC AUC supported only for binary classification'

        y_pred = self.auc_inference(cast(NTArray, data['x']))
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.to('cpu').numpy()

        try:
            auc_val = roc_auc_score(y_targ, y_pred)
        except ValueError as e:
            auc_val = -1
        return auc_val

    r"""
    Args:
        x: shape=(n_samples, n_features)
    """
    def visualize_decisions(self, x: NTArray, title: str='', save_path: str=''):
        if len(x.shape) != 2:
            Utils.cmlog('WARN: Ignoring Predictor.visualize_decisions(): len(x.shape) != 2\n', stdout=False) # type: ignore
            return
        if x.shape[1] != 2:
            Utils.cmlog('WARN: Ignoring Predictor.visualize_decisions(): x.shape[1] != 2\n', stdout=False) # type: ignore
            return

        _, ax = plt.subplots()
        ax.set_title(title)
        y = self.inference(x)
        if len(y.shape) == 2 and y.shape[1] == 1: # type: ignore
            y = y.squeeze(1)

        if len(y.shape) != 1:
            Utils.cmlog('WARN: Ignoring Predictor.visualize_decisions(): len(y.shape) != 1\n', stdout=False) # type: ignore
            return

        cmap = plt.get_cmap(CMAP)

        if isinstance(x, torch.Tensor):
            x = x.to('cpu').numpy()
            y = y.to('cpu').numpy()
        sc = ax.scatter(cast(np.ndarray, x)[:, 0], cast(np.ndarray, x)[:, 1], c=y, cmap=cmap, s=MARKER_SIZE)
        plt.colorbar(sc)

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

class LearnablePredictor(Predictor[NTArray], Generic[NTArray]):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_hyperparams(self) -> Dict[str, Any]:
        raise NotImplementedError

    r"""
    Args:
        train_data: Dataset to train on
            (each number in data['y'] is [0..num_classes - 1] if classification or R if regression)
        get_dataset_kwargs: The kwargs passed to get_dataset to obtain train_data and validn_data
        mirror_params: Details of normalization done by get_dataset_kwargs
            The above two are useful when denormalizing acc recorded during training
    """
    @abstractmethod
    def train(
        self,
        train_data: Dataset[NTArray],
        validn_data: Dataset[NTArray],
        test_data: Dataset[NTArray]
    ):
        raise NotImplementedError

class ChildrenInfo():
    def __init__(
        self, left_idx: Optional[int]=None, right_idx: Optional[int]=None,
        left_is_leaf: bool=False, right_is_leaf: bool=False
    ):
        self.left_idx = left_idx
        self.right_idx = right_idx
        self.left_is_leaf = left_is_leaf
        self.right_is_leaf = right_is_leaf

class DTPredictor(Predictor[torch.Tensor]):
    r"""
    Methods in this are decorated with no_grad() because the weights, biases and labels might be pointing
    to weights used in another model

    Args:
        weights: shape=(n_int_nodes, *)
        biases: shape=(n_int_nodes,)

        labels: shape=(n_leaf_nodes,) if classification
                shape=(n_leaf_nodes,) if simple regression
                shape=(n_leaf_nodes, n_featuresx) if regression with linear model in leaf

        labels_biases: None                  if classification
                       shape=(n_leaf_nodes,) if regression

        label_score: value used only when Predictor.auc_inference() (or Predictor.auc()) is called

        Precondition: n_int_nodes = n_leaf_nodes - 1 and n_leaf_nodes = 2 ** h
    """
    @torch.no_grad()
    def __init__(
        self,
        weights: torch.Tensor,
        biases: torch.Tensor,
        labels: torch.Tensor,
        labels_biases: Optional[torch.Tensor]=None,
        label_scores: Optional[torch.Tensor]=None,
        _is_classification: bool=True
    ):
        super().__init__()

        n_int_nodes = weights.shape[0]
        n_leaf_nodes = labels.shape[0]

        assert n_int_nodes == biases.shape[0]
        assert n_int_nodes == n_leaf_nodes - 1
        assert n_leaf_nodes != 0 and (n_leaf_nodes & (n_leaf_nodes - 1) == 0)
        assert len(biases.shape) == 1

        self._is_classification = _is_classification
        self._device0 = weights.device
        self._fp_type = FPTYPE

        self._leaf_layer_start = n_int_nodes
        self._weights = weights
        self._biases = biases
        self._labels = labels
        self._labels_biases = labels_biases

        self._leaf_ancs, self._leaf_ancs_dir = DTPredictor._get_leaf_ancs(height=int(math.log2(n_leaf_nodes)), device=Utils.get_device_num(weights))

        self._linear_leaf = (self._labels_biases is not None)
        assert not self._linear_leaf, 'linear model in leaves not fully supported (only the inference part i think)'

        if self._linear_leaf:
            self._leaf_node_repr = lambda label, label_bias: DTPredictor._linear_leaf_node_repr(label, label_bias)
        else:
            self._leaf_node_repr = lambda label, label_bias: DTPredictor._constant_leaf_node_repr(label)


    r"""
    Args:
        x: shape=(n_examples, n_features)

    Returns:
        *: shape=(n_examples,) or (n_examples, .); contains labels
    """
    @torch.no_grad()
    def inference(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device=self._device0, dtype=self._fp_type)
        return DTPredictor._fast_inference(
            x,
            self._weights,
            self._biases,
            self._leaf_ancs,
            self._leaf_ancs_dir,
            self._labels)

    @torch.no_grad()
    def raw_inference(self, x: torch.Tensor) -> torch.Tensor:
        return self.inference(x)

    @torch.no_grad()
    def auc_inference(self, x: torch.Tensor) -> torch.Tensor:
        assert Utils.is_binary_labels(self._labels), 'auc_inference only works for binary classification'

        raise NotImplementedError

    @torch.no_grad()
    def visualize_tree(self, save_path: str='', data: Optional[Dataset[torch.Tensor]]=None):
        import ete3

        if data is not None:
            data = data.to_tensor().to_type(FPTYPE, 'x' if self._is_classification else 'xy')
            data = data.to_device(self._device0)
        t = ete3.Tree() # type: ignore

        tree_style = ete3.TreeStyle() # type: ignore
        tree_style.rotation = -90
        tree_style.orientation = 1
        tree_style.show_scale = False
        tree_style.show_leaf_name = False

        idcs = None if data is None else list(range(data.n_examples))
        self._build_ete_tree(ete3, t, 0, self._weights.shape[0] == 0, data, idcs)

        if save_path:
            os.environ['QT_QPA_PLATFORM'] = 'offscreen'
            t.render(save_path, tree_style=tree_style)
        else:
            t.show(tree_style=tree_style)

    """
    Currently used only in _build_ete_tree and not anywhere else (not even in inference)
    no_grad() not required since _build_ete_tree already has it
    """
    @staticmethod
    def _should_go_left(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> bool:
        return (torch.dot(x, w) + b >= 0).item()


    r"""
    Build a tree rooted at either weights[node_idx] or labels[node_idx] (depending on is_leaf) and attach it to node
    """
    @torch.no_grad()
    def _build_ete_tree(
        self,
        ete3,
        node: 'ete3.TreeNode', # type: ignore
        node_idx: int,
        is_leaf: bool,
        data: Optional[Dataset[torch.Tensor]]=None,
        idcs: Optional[List[int]]=None
    ):
        if is_leaf:
            if self._linear_leaf:
                text = DTPredictor._linear_leaf_node_repr(self._labels[node_idx], self._labels_biases[node_idx])
            else:
                text = DTPredictor._constant_leaf_node_repr(self._labels[node_idx])

        else:
            text = DTPredictor._int_node_repr(self._weights[node_idx], self._biases[node_idx])

        recurse: bool = True
        if data is not None:
            text += '\n  (n={})  \n'.format(len(idcs))
            if len(idcs) > 0:
                if self._is_classification:
                    text = text[:-4] + ', mode={:.3f} \n'.format(data['y'][idcs].mode())
                else:
                    text = text[:-4] + ', mean={:.3f}, std={:.3f}, \n min={:.3f}, max={:.3f})  \n'.format(
                        data['y'][idcs].mean(), data['y'][idcs].std(unbiased=False), data['y'][idcs].min(), data['y'][idcs].max())
            else:
                text = '(n = 0)'
                recurse = False

        face = ete3.TextFace(text) # type: ignore
        face.border.width = 1
        face.rotation = -90
        node.add_face(face, 1, position='branch-right')

        if not recurse:
            return

        if not is_leaf:
            left_idcs = None
            right_idcs = None
            if data is not None:
                left_idcs = [i for i in idcs if DTPredictor._should_go_left(data['x'][i], self._weights[node_idx], self._biases[node_idx])]
                right_idcs = list(set(idcs) - set(left_idcs))

            left = node.add_child()
            node_style = ete3.NodeStyle() # type: ignore
            node_style['size'] = 0
            node_style['hz_line_color'] = 'GREEN'
            node_style['hz_line_width'] = 5
            left.set_style(node_style)

            left_idx = 2 * node_idx + 1
            is_leaf = left_idx >= self._leaf_layer_start
            if is_leaf:
                left_idx -= self._leaf_layer_start

            self._build_ete_tree(ete3, left, left_idx, is_leaf, data, left_idcs)

            right = node.add_child()
            node_style = ete3.NodeStyle() # type: ignore
            node_style['size'] = 0
            node_style['hz_line_color'] = 'RED'
            node_style['hz_line_width'] = 5
            right.set_style(node_style)

            right_idx = 2 * node_idx + 2
            is_leaf = right_idx >= self._leaf_layer_start
            if is_leaf:
                right_idx -= self._leaf_layer_start

            self._build_ete_tree(ete3, right, right_idx, is_leaf, data, right_idcs)

    """
    w: (n_features,)
    b: (1,)
    """
    @staticmethod
    @torch.no_grad()
    def _int_node_repr(w: torch.Tensor, b: torch.Tensor) -> str:
        sel = (w != 0)
        if sel.sum() == 0 and b.item() == 0:
            mult = 1
        else:
            if (w != 0).sum() > 0:
                mn = torch.abs(w[w != 0]).min().item()
                mult = 10 ** (int(abs(math.log10(mn))) + 1)
            else:
                mult = 1
        mult = 1

        text = '\n '
        for i in range(min(w.shape[0], INT_NODE_REPR_MAX_FEATURES)):
            text += ' {} {:.2f} x{} '.format(Utils.sign(w[i].item()), abs(w[i].item()) * mult, i)
            if i % 3 == 2:
                text += ' \n'
        text += ' {} {:.2f} >= 0 \n'.format(Utils.sign(b.item()), abs(b.item()) * mult)
        return text

    """
    label: (1,)
    """
    @staticmethod
    @torch.no_grad()
    def _constant_leaf_node_repr(label: torch.Tensor) -> str:
        return '\n  {:.4f}  \n'.format(label.item())

    """
    label: (n_featuresx,)
    label_bias: (1,)
    """
    @staticmethod
    @torch.no_grad()
    def _linear_leaf_node_repr(label: torch.Tensor, label_bias: torch.Tensor) -> str:
        return (
            ' ' +
            ''.join([' {} {:.2f} x{}'.format(Utils.sign(label[i].item()), abs(label[i].item()), i + 1) for i in range(label.shape[0])]) +
            ' {} {:.2f}'.format(Utils.sign(label_bias.item()), abs(label_bias.item())) +
            ' '
        )

    r"""
    Returns leaf labels of a tree with node parameters given as weights, biases, such that
    the tree fits to the data as perfectly as possible.
    In the case of classification, the label for a leaf will be the mode of target (which are classes)
    of all data points that fall in that leaf, whereas in the case of regression, the label for a leaf will
    be the mean of targets of all data points that fall in that leaf.
    """
    @staticmethod
    @torch.no_grad()
    def get_best_leaf_labels(
        weights: torch.Tensor,
        biases: torch.Tensor,
        data: Dataset[torch.Tensor],
        is_classification: bool=True
    ) -> Tuple[torch.Tensor, float]:

        height = int(math.log2(weights.shape[0] + 1))
        n_leaf_nodes = 2 ** height

        leaf_ancs, leaf_ancs_dir = DTPredictor._get_leaf_ancs(height, device=Utils.get_device_num(weights))

        if PROFILE_TIME: starttime = time.time()
        leaf_idcs = DTPredictor._fast_inference(
            data['x'],
            weights,
            biases,
            leaf_ancs,
            leaf_ancs_dir
        )
        tt = time.time() - starttime if PROFILE_TIME else -1

        if is_classification:
            labels = torch.empty((n_leaf_nodes,), device=weights.device, dtype=data['y'].dtype)
            mode = data['y'].mode()[0].item()
            for i in range(n_leaf_nodes):
                sel = (leaf_idcs == i)
                labels[i] = data['y'][sel].mode()[0].item() if sel.sum().item() > 0 else mode

        else:
            if DETERMINISTIC:
                labels = torch.empty((n_leaf_nodes,), device=weights.device, dtype=data['y'].dtype)
                mean = data['y'].mean().item()
                for i in range(n_leaf_nodes):
                    sel = (leaf_idcs == i)
                    labels[i] = data['y'][sel].mean().item() if sel.sum().item() > 0 else mean

            else:
                labels = torch.zeros_like(data['y']).scatter_add(0, leaf_idcs, data['y'])
                vals, cnts = leaf_idcs.unique(return_counts=True)
                denom = torch.zeros_like(leaf_idcs)
                denom[vals] = cnts
                labels[denom == 0] = data['y'].mean().item()
                denom[denom == 0] = 1
                labels = (labels / denom)[: n_leaf_nodes]

        return labels, tt

    """
    device is an int to help enable lru_cache
    Returns
        - a matrix of shape (n_leaf_nodes, height) which gives indices of all ancestors of every leaf node
        - a matrix of shape (n_leaf_nodes, height) which gives direction taken at every leaf node (1 for left or -1 for right)
    """
    @staticmethod
    @functools.lru_cache()
    def _get_leaf_ancs(height: int, device: int) -> Tuple[torch.Tensor, torch.Tensor]:
        leaf_ancs = np.array([[0], [0]], dtype=np.int64)
        leaf_ancs_dir = np.array([[1], [-1]], dtype=np.int64)

        for i in range(2, height + 1):
            # compute leaf_ancs, leaf_ancs_dir for height i
            new_leaf_ancs = np.append(leaf_ancs, 2 * leaf_ancs[:, -1:] + 1, axis=1)
            new_leaf_ancs[1::2, -1] += 1
            new_leaf_ancs = np.repeat(new_leaf_ancs, 2, axis=0)
            leaf_ancs = new_leaf_ancs

            new_leaf_ancs_dir = np.append(leaf_ancs_dir, np.ones((len(leaf_ancs_dir), 1), dtype=np.int64), axis=1)
            new_leaf_ancs_dir = np.repeat(new_leaf_ancs_dir, 2, axis=0)
            new_leaf_ancs_dir[1::2, -1] = -1
            leaf_ancs_dir = new_leaf_ancs_dir

        leaf_ancs = torch.from_numpy(leaf_ancs).to(dtype=torch.int64)
        leaf_ancs_dir = torch.from_numpy(leaf_ancs_dir).to(dtype=torch.int64)
        if device >= 0:
            leaf_ancs = leaf_ancs.to(device=device)
            leaf_ancs_dir = leaf_ancs_dir.to(device=device)

        return leaf_ancs, leaf_ancs_dir

    @staticmethod
    @torch.no_grad()
    def _fast_inference(
        x: torch.Tensor, # (n_samples, n_features)
        weights: torch.Tensor, # (n_int_nodes, n_features)
        biases: torch.Tensor, # (n_int_nodes,)
        leaf_ancs: torch.Tensor, # (n_leaf_nodes, height)
        leaf_ancs_dir: torch.Tensor, # (n_leaf_nodes, height)
        leaf_labels: Optional[torch.Tensor]=None
    ) -> torch.Tensor:

        height = int(math.log2(weights.shape[0] + 1))

        mult = torch.matmul(x, weights.T) + biases
        xpath = 2 * (mult >= 0).to(dtype=torch.int64) - 1 # shape=(n_samples, n_int_nodes)
        tmp = (xpath[:, leaf_ancs] * leaf_ancs_dir).sum(axis=2) # shape=(n_samples, n_leaf_nodes)
        sel = (tmp == height)
        leaf_idcs = torch.nonzero(sel, as_tuple=True)[1]

        return leaf_idcs if leaf_labels is None else leaf_labels[leaf_idcs]

class DTExtractablePredictor(Predictor[torch.Tensor]):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def extract_dt_predictor(self) -> DTPredictor:
        raise NotImplementedError
