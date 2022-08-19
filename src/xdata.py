from typing import Any, Dict, Optional, Tuple, cast

import numpy as np
import pandas as pd
from scipy import stats
import requests
import os
import bz2
import gzip
import pickle
import tarfile
import io
import zipfile

from models.xfixedconst import FixedConstantPredictor
from xcommon import Dataset, Utils
from xconstants import DATASETS_ROOT

TVT = Tuple[Dataset[np.ndarray], Dataset[np.ndarray], Dataset[np.ndarray]]
TV = Tuple[Dataset[np.ndarray], Dataset[np.ndarray]]

SEED_DEF: int = 1

class DataLoader():
    stats: Dict[str, Dict[str, int]] = Utils.read_json(f'{DATASETS_ROOT}/stats.json')

    @staticmethod
    def get_dataset(
        dataset_name: str, **kwargs
    ) -> TVT:
        assert 'shuffle_seed' in kwargs

        try:
            train, validn, test = getattr(DataLoader, 'prep_{}_all'.format(dataset_name.replace('-', '_')))(**kwargs)
        except AttributeError:
            train, validn, test = DataLoader._prep_generic_all(dataset_name=dataset_name, **kwargs)

        train.shuffle_seed = kwargs['shuffle_seed']
        validn.shuffle_seed = kwargs['shuffle_seed']
        test.shuffle_seed = kwargs['shuffle_seed']
        return train, validn, test

    @staticmethod
    def _prep_generic(dataset_name: str, category: Optional[str]) -> Dataset[np.ndarray]:
        if category in ['train', 'val', 'test']:
            x = np.load(f'{DATASETS_ROOT}/{dataset_name}/{dataset_name}-{category}-x.npy').astype(np.float32)
            y = np.load(f'{DATASETS_ROOT}/{dataset_name}/{dataset_name}-{category}-y.npy').astype(np.int64 if DataLoader.is_classification(dataset_name) else np.float32)

            return Dataset(x, y, name=dataset_name, copy=False, autoshrink_y=True)

        else:
            raise ValueError('category must be in ["train", "test", "val"]')

    @staticmethod
    def _prep_generic_all(
        dataset_name: str, normalize_x_kwargs: Optional[Dict[str, Any]]=None, normalize_y_kwargs: Optional[Dict[str, Any]]=None, shuffle_seed: int=SEED_DEF
    ) -> TVT:

        train = DataLoader._prep_generic(dataset_name, category='train')
        validn = DataLoader._prep_generic(dataset_name, category='val')
        test = DataLoader._prep_generic(dataset_name, category='test')
        return DataLoader.normalize_all_datasets(train, validn, test, normalize_x_kwargs, normalize_y_kwargs)

    @staticmethod
    def prep_pdbbind_all(
        normalize_x_kwargs: Optional[Dict[str, Any]]=None, normalize_y_kwargs: Optional[Dict[str, Any]]=None, shuffle_seed: int=SEED_DEF
    ) -> TVT:
        url = r'https://raw.githubusercontent.com/guanghelee/iclr20-lcn/master/data/PDBbind.pkl.gz'
        path = f'{DATASETS_ROOT}/pdbbind/PDBbind.pkl.gz'
        DataLoader.download(url, path)

        with gzip.open(path, 'rb') as f:
            xtrain, ytrain, xval, yval, xtest, ytest = pickle.load(f)

        train = Dataset(xtrain.astype(np.float32), ytrain.astype(np.float32), name='pdbbind', copy=False, autoshrink_y=True)
        validn = Dataset(xval.astype(np.float32), yval.astype(np.float32), name='pdbbind', copy=False, autoshrink_y=True)
        test = Dataset(xtest.astype(np.float32), ytest.astype(np.float32), name='pdbbind', copy=False, autoshrink_y=True)

        return DataLoader.normalize_all_datasets(train, validn, test, normalize_x_kwargs, normalize_y_kwargs)

    @staticmethod
    def prep_abalone(category, seed) -> Dataset[np.ndarray]:
        npz = np.load(open('{}/abalone/abalone{}.npz'.format(DATASETS_ROOT, seed-1), 'rb'))
        x,y = npz[f'X_{category}'].astype(np.float32),npz[f'y_{category}'].astype(np.float32)
        return Dataset(x, y, name='abalone', copy=False, autoshrink_y=True)

    @staticmethod
    def prep_abalone_all(
        normalize_x_kwargs: Optional[Dict[str, Any]]=None, normalize_y_kwargs: Optional[Dict[str, Any]]=None, shuffle_seed: int=SEED_DEF
    ) -> TVT:
        from xconstants import TAO

        train, validn = cast(TV, DataLoader.prep_abalone('train', shuffle_seed).shuffle(seed=shuffle_seed).split(0.8 if not TAO else 0.9))
        test = DataLoader.prep_abalone('test', shuffle_seed).shuffle(seed=shuffle_seed)

        return DataLoader.normalize_all_datasets(train, validn, test, normalize_x_kwargs, normalize_y_kwargs)

    @staticmethod
    def prep_cpuactiv(category, seed) -> Dataset[np.ndarray]:
        with open('{}/cpuactiv/cpu_act{}.npz'.format(DATASETS_ROOT, seed-1), 'rb') as f:
            npz = np.load(f)
            x,y = npz[f'X_{category}'].astype(np.float32),npz[f'y_{category}'].astype(np.float32)
        return Dataset(x, y, name='cpuactiv', copy=False, autoshrink_y=True)

    @staticmethod
    def prep_cpuactiv_all(
        normalize_x_kwargs: Optional[Dict[str, Any]]=None, normalize_y_kwargs: Optional[Dict[str, Any]]=None, shuffle_seed: int=SEED_DEF
    ) -> TVT:
        from xconstants import TAO

        train, validn = cast(TV, DataLoader.prep_cpuactiv('train', shuffle_seed).shuffle(seed=shuffle_seed).split(0.8 if not TAO else 0.9))
        test = DataLoader.prep_cpuactiv('test', shuffle_seed).shuffle(seed=shuffle_seed)

        return DataLoader.normalize_all_datasets(train, validn, test, normalize_x_kwargs, normalize_y_kwargs)

    @staticmethod
    def prep_ailerons(category: str) -> Dataset[np.ndarray]:
        df = pd.read_csv(f'{DATASETS_ROOT}/ailerons/ailerons.{category}', sep=',', header=None)

        x = np.array(df.iloc[:, :-1], dtype=np.float32)
        y = np.array(df.iloc[:, -1], dtype=np.float32) * 1e4
        return Dataset(x, y, name='ailerons', copy=False, autoshrink_y=True)

    @staticmethod
    def prep_ailerons_all(
        normalize_x_kwargs: Optional[Dict[str, Any]]=None, normalize_y_kwargs: Optional[Dict[str, Any]]=None, shuffle_seed: int=SEED_DEF
    ) -> TVT:
        url = r'https://www.dcc.fc.up.pt/~ltorgo/Regression/ailerons.tgz'
        DataLoader.download(url, f'{DATASETS_ROOT}/ailerons')

        from xconstants import TAO

        train, validn = cast(TV, DataLoader.prep_ailerons('data').shuffle(seed=shuffle_seed).split(0.8 if not TAO else 0.9))
        test = DataLoader.prep_ailerons('test').shuffle(seed=shuffle_seed)
        return DataLoader.normalize_all_datasets(train, validn, test, normalize_x_kwargs, normalize_y_kwargs)

    @staticmethod
    def prep_ctslice(category, seed) -> Dataset[np.ndarray]:
        raise NotImplementedError
        with open('{}/ctslice/ctslice{}.npz'.format(DATASETS_ROOT, seed-1), 'rb') as f:
            npz = np.load(f)
            x,y = npz[f'X_{category}'].astype(np.float32),npz[f'y_{category}'].astype(np.float32)
        return Dataset(x, y, name='ctslice', copy=False, autoshrink_y=True)

    @staticmethod
    def prep_ctslice_all(
        normalize_x_kwargs: Optional[Dict[str, Any]]=None, normalize_y_kwargs: Optional[Dict[str, Any]]=None, shuffle_seed: int=SEED_DEF
    ) -> TVT:
        from xconstants import TAO

        train, validn = cast(TV, DataLoader.prep_ctslice('train', shuffle_seed).shuffle(seed=shuffle_seed).split(0.8 if not TAO else 0.9))
        test = DataLoader.prep_ctslice('test', shuffle_seed).shuffle(seed=shuffle_seed)

        return DataLoader.normalize_all_datasets(train, validn, test, normalize_x_kwargs, normalize_y_kwargs)

    @staticmethod
    def prep_year(category: Optional[str]) -> Dataset[np.ndarray]:
        if category in ['train', 'test']:
            x = np.load(f'{DATASETS_ROOT}/year/year-{category}-x.npy').astype(np.float32)
            y = np.load(f'{DATASETS_ROOT}/year/year-{category}-y.npy').astype(np.float32)

            return Dataset(x, y, name='year', copy=False, autoshrink_y=True)
        else:
            raise ValueError('category must be in ["train", "test"]')

    @staticmethod
    def prep_year_all(
        normalize_x_kwargs: Optional[Dict[str, Any]]=None, normalize_y_kwargs: Optional[Dict[str, Any]]=None, shuffle_seed: int=SEED_DEF
    ) -> TVT:
        url = r'https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip'
        path = f'{DATASETS_ROOT}/year'
        if DataLoader.download(url, path):
            df = pd.read_csv(f'{path}/YearPredictionMSD.txt')
            split_at = 463715
            train, test = df.iloc[:split_at], df.iloc[split_at:]
            train.iloc[:, 0].to_numpy(np.int64).save(f'{path}/year-train-y.npy')
            train.iloc[:, 1:].to_numpy(np.float32).save(f'{path}/year-train-x.npy')
            test.iloc[:, 0].to_numpy(np.int64).save(f'{path}/year-test-y.npy')
            test.iloc[:, 1:].to_numpy(np.float32).save(f'{path}/year-test-x.npy')

        from xconstants import TAO

        train, validn = cast(TV, DataLoader.prep_year(category='train').shuffle(seed=shuffle_seed).split(0.8 if not TAO else 0.9))
        test = DataLoader.prep_year(category='test')
        return DataLoader.normalize_all_datasets(train, validn, test, normalize_x_kwargs, normalize_y_kwargs)

    @staticmethod
    def prep_mic(category: Optional[str]) -> Dataset[np.ndarray]:
        if category in ['train', 'test', 'val']:
            x = np.load('{}/mic/mic-{}-x.npy'.format(DATASETS_ROOT, category)).astype(np.float32)
            y = np.load('{}/mic/mic-{}-y.npy'.format(DATASETS_ROOT, category)).astype(np.float32)

            return Dataset(x, y, name='mic', copy=False, autoshrink_y=True)
        else:
            raise ValueError('category must be in ["train", "test", "val"]')

    @staticmethod
    def prep_mic_all(
        normalize_x_kwargs: Optional[Dict[str, Any]]=None, normalize_y_kwargs: Optional[Dict[str, Any]]=None, shuffle_seed: int=SEED_DEF
    ) -> TVT:

        # other papers dont use given val, instead split train, so we do the same
        train, validn = cast(TV, DataLoader.prep_mic(category='train').shuffle(seed=shuffle_seed).split(0.8))
        test = cast(Dataset[np.ndarray], DataLoader.prep_mic(category='test'))
        return DataLoader.normalize_all_datasets(train, validn, test, normalize_x_kwargs, normalize_y_kwargs)

    @staticmethod
    def prep_yah(category: Optional[str]) -> Dataset[np.ndarray]:
        if category in ['train', 'test', 'val']:
            x = np.load('{}/yah/yah-{}-x.npy'.format(DATASETS_ROOT, category)).astype(np.float32)
            y = np.load('{}/yah/yah-{}-y.npy'.format(DATASETS_ROOT, category)).astype(np.float32)

            return Dataset(x, y, name='yah', copy=False, autoshrink_y=True)
        else:
            raise ValueError('category must be in ["train", "test", "val"]')

    @staticmethod
    def prep_yah_all(
        normalize_x_kwargs: Optional[Dict[str, Any]]=None, normalize_y_kwargs: Optional[Dict[str, Any]]=None, shuffle_seed: int=SEED_DEF
    ) -> TVT:

        train = cast(Dataset[np.ndarray], DataLoader.prep_yah(category='train'))
        validn = cast(Dataset[np.ndarray], DataLoader.prep_yah(category='val'))
        test = cast(Dataset[np.ndarray], DataLoader.prep_yah(category='test'))
        return DataLoader.normalize_all_datasets(train, validn, test, normalize_x_kwargs, normalize_y_kwargs)

    @staticmethod
    def prep_connect_4(shuffle_seed: int) -> Dataset[np.ndarray]:
        url = r'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/connect-4'
        path = f'{DATASETS_ROOT}/connect-4/connect-4.txt'
        DataLoader.download(url, path)

        return DataLoader.read_libsvm_format(
            path,
            n_features=DataLoader.stats['connect-4']['n_features'],
            n_classes=DataLoader.stats['connect-4']['n_classes'],
            name='connect-4', shuffle_seed=shuffle_seed)

    @staticmethod
    def prep_connect_4_all(
       normalize_x_kwargs: Optional[Dict[str, Any]]=None, normalize_y_kwargs: Optional[Dict[str, Any]]=None, shuffle_seed: int=SEED_DEF
    ) -> TVT:

        train, validn, test = cast(TVT, DataLoader.prep_connect_4(shuffle_seed).split(0.64, 0.16))
        return DataLoader.normalize_all_datasets(train, validn, test, normalize_x_kwargs, normalize_y_kwargs)

    @staticmethod
    def prep_mnist(shuffle_seed: int, category: Optional[str]=None) -> Dataset[np.ndarray]:
        if category is None:
            url = r'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.bz2'
            path = f'{DATASETS_ROOT}/mnist/mnist.txt'
            DataLoader.download(url, path)

            return DataLoader.read_libsvm_format(
                path,
                n_features=DataLoader.stats['mnist']['n_features'],
                n_classes=DataLoader.stats['mnist']['n_classes'],
                name='mnist', shuffle_seed=shuffle_seed)

        elif category == 'test':
            url = r'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.t.bz2'
            path = f'{DATASETS_ROOT}/mnist/mnist-test.txt'
            DataLoader.download(url, path)

            return DataLoader.read_libsvm_format(
                path,
                n_features=DataLoader.stats['mnist']['n_features'],
                n_classes=DataLoader.stats['mnist']['n_classes'],
                name='mnist', shuffle_seed=1)

        else:
            raise ValueError('category must be in [None, "test"]')

    @staticmethod
    def prep_mnist_all(
        normalize_x_kwargs: Optional[Dict[str, Any]]=None, normalize_y_kwargs: Optional[Dict[str, Any]]=None, shuffle_seed: int=SEED_DEF
    ) -> TVT:

        train, validn = cast(
            Tuple[Dataset[np.ndarray], Dataset[np.ndarray]],
            DataLoader.prep_mnist(shuffle_seed, category=None).split(0.8))
        test = DataLoader.prep_mnist(shuffle_seed, category='test')
        return DataLoader.normalize_all_datasets(train, validn, test, normalize_x_kwargs, normalize_y_kwargs)

    @staticmethod
    def prep_protein(category: Optional[str]=None) -> Dataset[np.ndarray]:
        if category is None:
            url = r'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/protein.bz2'
            path = f'{DATASETS_ROOT}/protein/protein.txt'
            DataLoader.download(url, path)

            return DataLoader.read_libsvm_format(
                path,
                n_features=DataLoader.stats['protein']['n_features'],
                n_classes=DataLoader.stats['protein']['n_classes'],
                name='protein')

        elif category == 'train':
            url = r'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/protein.tr.bz2'
            path = f'{DATASETS_ROOT}/protein/protein-train.txt'
            DataLoader.download(url, path)

            return DataLoader.read_libsvm_format(
                path,
                n_features=DataLoader.stats['protein']['n_features'],
                n_classes=DataLoader.stats['protein']['n_classes'],
                name='protein')

        elif category == 'test':
            url = r'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/protein.t.bz2'
            path = f'{DATASETS_ROOT}/protein/protein-test.txt'
            DataLoader.download(url, path)

            return DataLoader.read_libsvm_format(
                path,
                n_features=DataLoader.stats['protein']['n_features'],
                n_classes=DataLoader.stats['protein']['n_classes'],
                name='protein')

        elif category == 'val':
            url = r'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/protein.val.bz2'
            path = f'{DATASETS_ROOT}/protein/protein-val.txt'
            DataLoader.download(url, path)

            return DataLoader.read_libsvm_format(
                path,
                n_features=DataLoader.stats['protein']['n_features'],
                n_classes=DataLoader.stats['protein']['n_classes'],
                name='protein')

        else:
            raise ValueError('category must be in [None, "train", "test", "val"]')

    @staticmethod
    def prep_protein_all(
        normalize_x_kwargs: Optional[Dict[str, Any]]=None, normalize_y_kwargs: Optional[Dict[str, Any]]=None, shuffle_seed: int=SEED_DEF
    ) -> TVT:

        train = DataLoader.prep_protein(category='train')
        validn = DataLoader.prep_protein(category='val')
        test = DataLoader.prep_protein(category='test')
        return DataLoader.normalize_all_datasets(train, validn, test, normalize_x_kwargs, normalize_y_kwargs)

    @staticmethod
    def prep_sensit_combined(shuffle_seed: int, category: Optional[str]=None) -> Dataset[np.ndarray]:
        if category is None:
            url = r'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/vehicle/combined.bz2'
            path = f'{DATASETS_ROOT}/sensit-combined/sensit-combined.txt'
            DataLoader.download(url, path)

            return DataLoader.read_libsvm_format(
                path,
                n_features=DataLoader.stats['sensit-combined']['n_features'],
                n_classes=DataLoader.stats['sensit-combined']['n_classes'],
                name='sensit-combined', shuffle_seed=shuffle_seed)

        elif category == 'test':
            url = r'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/vehicle/combined.t.bz2'
            path = f'{DATASETS_ROOT}/sensit-combined/sensit-combined-test.txt'
            DataLoader.download(url, path)

            return DataLoader.read_libsvm_format(
                path,
                n_features=DataLoader.stats['sensit-combined']['n_features'],
                n_classes=DataLoader.stats['sensit-combined']['n_classes'],
                name='sensit-combined', shuffle_seed=1)

        else:
            raise ValueError('category must be in [None, "test"]')

    @staticmethod
    def prep_sensit_combined_all(
        normalize_x_kwargs: Optional[Dict[str, Any]]=None, normalize_y_kwargs: Optional[Dict[str, Any]]=None, shuffle_seed: int=SEED_DEF
    ) -> TVT:

        train, validn = cast(
            Tuple[Dataset[np.ndarray], Dataset[np.ndarray]],
            DataLoader.prep_sensit_combined(shuffle_seed, category=None).split(0.8))
        test = DataLoader.prep_sensit_combined(shuffle_seed, category='test')
        return DataLoader.normalize_all_datasets(train, validn, test, normalize_x_kwargs, normalize_y_kwargs)

    @staticmethod
    def prep_letter(category: Optional[str]=None) -> Dataset[np.ndarray]:
        if category is None:
            url = r'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/letter.scale'
            path = f'{DATASETS_ROOT}/letter/letter.txt'
            DataLoader.download(url, path)

            return DataLoader.read_libsvm_format(
                path,
                n_features=DataLoader.stats['letter']['n_features'],
                n_classes=DataLoader.stats['letter']['n_classes'],
                name='letter')

        elif category == 'train':
            url = r'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/letter.scale.tr'
            path = f'{DATASETS_ROOT}/letter/letter-train.txt'
            DataLoader.download(url, path)

            return DataLoader.read_libsvm_format(
                path,
                n_features=DataLoader.stats['letter']['n_features'],
                n_classes=DataLoader.stats['letter']['n_classes'],
                name='letter')

        elif category == 'test':
            url = r'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/letter.scale.t'
            path = f'{DATASETS_ROOT}/letter/letter-test.txt'
            DataLoader.download(url, path)

            return DataLoader.read_libsvm_format(
                path,
                n_features=DataLoader.stats['letter']['n_features'],
                n_classes=DataLoader.stats['letter']['n_classes'],
                name='letter')

        elif category == 'val':
            url = r'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/letter.scale.val'
            path = f'{DATASETS_ROOT}/letter/letter-val.txt'
            DataLoader.download(url, path)

            return DataLoader.read_libsvm_format(
                path,
                n_features=DataLoader.stats['letter']['n_features'],
                n_classes=DataLoader.stats['letter']['n_classes'],
                name='letter')

        else:
            raise ValueError('category must be in [None, "train", "test", "val"]')

    @staticmethod
    def prep_letter_all(
        normalize_x_kwargs: Optional[Dict[str, Any]]=None, normalize_y_kwargs: Optional[Dict[str, Any]]=None, shuffle_seed: int=SEED_DEF
    ) -> TVT:
        train = DataLoader.prep_letter(category='train')
        validn = DataLoader.prep_letter(category='val')
        test = DataLoader.prep_letter(category='test')

        return DataLoader.normalize_all_datasets(train, validn, test, normalize_x_kwargs, normalize_y_kwargs)

    @staticmethod
    def prep_pendigits(shuffle_seed: int, category: Optional[str]=None) -> Dataset[np.ndarray]:
        if category is None:
            url = r'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/pendigits'
            path = f'{DATASETS_ROOT}/pendigits/pendigits.txt'
            DataLoader.download(url, path)

            return DataLoader.read_libsvm_format(
                path,
                n_features=DataLoader.stats['pendigits']['n_features'],
                n_classes=DataLoader.stats['pendigits']['n_classes'],
                name='pendigits', shuffle_seed=shuffle_seed)

        elif category == 'test':
            url = r'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/pendigits.t'
            path = f'{DATASETS_ROOT}/pendigits/pendigits-test.txt'
            DataLoader.download(url, path)

            return DataLoader.read_libsvm_format(
                path,
                n_features=DataLoader.stats['pendigits']['n_features'],
                n_classes=DataLoader.stats['pendigits']['n_classes'],
                name='pendigits', shuffle_seed=1)

        else:
            raise ValueError('category must be in [None, "test"]')

    @staticmethod
    def prep_pendigits_all(
        normalize_x_kwargs: Optional[Dict[str, Any]]=None, normalize_y_kwargs: Optional[Dict[str, Any]]=None, shuffle_seed: int=SEED_DEF
    ) -> TVT:

        train, validn = cast(
            Tuple[Dataset[np.ndarray], Dataset[np.ndarray]],
            DataLoader.prep_pendigits(shuffle_seed, category=None).split(0.8))
        test = DataLoader.prep_pendigits(shuffle_seed, category='test')
        return DataLoader.normalize_all_datasets(train, validn, test, normalize_x_kwargs, normalize_y_kwargs)

    @staticmethod
    def prep_satimage(category: Optional[str]=None) -> Dataset[np.ndarray]:
        if category is None:
            url = r'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/satimage.scale'
            path = f'{DATASETS_ROOT}/satimage/satimage.txt'
            DataLoader.download(url, path)

            return DataLoader.read_libsvm_format(
                path,
                n_features=DataLoader.stats['satimage']['n_features'],
                n_classes=DataLoader.stats['satimage']['n_classes'],
                name='satimage')

        elif category == 'train':
            url = r'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/satimage.scale.tr'
            path = f'{DATASETS_ROOT}/satimage/satimage-train.txt'
            DataLoader.download(url, path)

            return DataLoader.read_libsvm_format(
                path,
                n_features=DataLoader.stats['satimage']['n_features'],
                n_classes=DataLoader.stats['satimage']['n_classes'],
                name='satimage')

        elif category == 'test':
            url = r'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/satimage.scale.t'
            path = f'{DATASETS_ROOT}/satimage/satimage-test.txt'
            DataLoader.download(url, path)

            return DataLoader.read_libsvm_format(
                path,
                n_features=DataLoader.stats['satimage']['n_features'],
                n_classes=DataLoader.stats['satimage']['n_classes'],
                name='satimage')

        elif category == 'val':
            url = r'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/satimage.scale.val'
            path = f'{DATASETS_ROOT}/satimage/satimage-val.txt'
            DataLoader.download(url, path)

            return DataLoader.read_libsvm_format(
                path,
                n_features=DataLoader.stats['satimage']['n_features'],
                n_classes=DataLoader.stats['satimage']['n_classes'],
                name='satimage')

        else:
            raise ValueError('category must be in [None, "train", "test", "val"]')

    @staticmethod
    def prep_satimage_all(
        normalize_x_kwargs: Optional[Dict[str, Any]]=None, normalize_y_kwargs: Optional[Dict[str, Any]]=None, shuffle_seed: int=SEED_DEF
    ) -> TVT:

        train = DataLoader.prep_satimage(category='train')
        validn = DataLoader.prep_satimage(category='val')
        test = DataLoader.prep_satimage(category='test')
        return DataLoader.normalize_all_datasets(train, validn, test, normalize_x_kwargs, normalize_y_kwargs)

    @staticmethod
    def prep_segment(shuffle_seed: int) -> Dataset[np.ndarray]:
        url = r'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/segment.scale'
        path = f'{DATASETS_ROOT}/segment/segment.txt'
        DataLoader.download(url, path)

        return DataLoader.read_libsvm_format(
            path,
            n_features=DataLoader.stats['segment']['n_features'],
            n_classes=DataLoader.stats['segment']['n_classes'],
            name='segment', shuffle_seed=shuffle_seed)

    @staticmethod
    def prep_segment_all(
        normalize_x_kwargs: Optional[Dict[str, Any]]=None, normalize_y_kwargs: Optional[Dict[str, Any]]=None, shuffle_seed: int=SEED_DEF
    ) -> TVT:

        train, validn, test = cast(TVT, DataLoader.prep_segment(shuffle_seed).split(0.64, 0.16))
        return DataLoader.normalize_all_datasets(train, validn, test, normalize_x_kwargs, normalize_y_kwargs)

    @staticmethod
    def prep_bace(category: Optional[str]) -> Dataset[np.ndarray]:
        if category in ['train', 'val', 'test']:
            if category == 'val':
                category = 'valid'

            url = rf'https://raw.githubusercontent.com/guanghelee/iclr20-lcn/master/data/bace_split/{category}.fgp2048.csv'
            path = f'{DATASETS_ROOT}/bace/{category}.fgp2048.csv'
            DataLoader.download(url, path)

            df = pd.read_csv(path)
            x = np.array([[int(j) for j in i] for i in df['mol'].values]).astype(np.float32)
            y = df['Class'].values.astype(np.int64)

            return Dataset(x, y, name='bace', copy=False, autoshrink_y=True)
        else:
            raise ValueError('category must be in ["train", "test", "val"]')

    @staticmethod
    def prep_bace_all(
        normalize_x_kwargs: Optional[Dict[str, Any]]=None, normalize_y_kwargs: Optional[Dict[str, Any]]=None, shuffle_seed: int=SEED_DEF
    ) -> TVT:

        train = DataLoader.prep_bace(category='train')
        validn = DataLoader.prep_bace(category='val')
        test = DataLoader.prep_bace(category='test')
        return DataLoader.normalize_all_datasets(train, validn, test, normalize_x_kwargs, normalize_y_kwargs)

    @staticmethod
    def prep_hiv(category: Optional[str]) -> Dataset[np.ndarray]:
        if category in ['train', 'val', 'test']:
            if category == 'val':
                category = 'valid'

            url = rf'https://raw.githubusercontent.com/guanghelee/iclr20-lcn/master/data/HIV_split/{category}.fgp2048.csv'
            path = f'{DATASETS_ROOT}/hiv/{category}.fgp2048.csv'
            DataLoader.download(url, path)

            df = pd.read_csv(path)
            x = np.array([[int(j) for j in i] for i in df['smiles'].values]).astype(np.float32)
            y = df['HIV_active'].values.astype(np.int64)

            return Dataset(x, y, name='hiv', copy=False, autoshrink_y=True)
        else:
            raise ValueError('category must be in ["train", "test", "val"]')

    @staticmethod
    def prep_hiv_all(
        normalize_x_kwargs: Optional[Dict[str, Any]]=None, normalize_y_kwargs: Optional[Dict[str, Any]]=None, shuffle_seed: int=SEED_DEF
    ) -> TVT:

        train = DataLoader.prep_hiv(category='train')
        validn = DataLoader.prep_hiv(category='val')
        test = DataLoader.prep_hiv(category='test')
        return DataLoader.normalize_all_datasets(train, validn, test, normalize_x_kwargs, normalize_y_kwargs)

    """
    Notes:
        Converts feature names to int (and throws if such conversion is not posible). Assumes feature names are 1-indexed and converts them to 0-indexed.
        Converts feature values to float (and throws if such conversion is not possible).
        Assumes only 1 label exists and n_classes is the number of classes for that label. Incase of classification, converts labels to 0-indexed if it is not.
    """
    @staticmethod
    def read_libsvm_format(
        file_path: str, n_features: int, n_classes: int, name: str='', shuffle_seed: Optional[int]=1
    ) -> Dataset[np.ndarray]:
        is_classification = (n_classes > 0)

        with open(file_path, 'r') as f:
            content = f.read()
        assert ':  ' not in content, 'Error while reading: {}'.format(file_path)

        content = content.replace(': ', ':')
        content = content.strip()
        lines = content.split('\n')
        lines = [line.strip() for line in lines]

        x = np.zeros((len(lines), n_features), dtype=np.float32)
        y = np.zeros((len(lines),), dtype=np.int64 if is_classification else np.float32)

        for line_idx, line in enumerate(lines):
            for unit_idx, unit in enumerate(line.split()):
                if unit_idx == 0:
                    assert ':' not in unit
                    if is_classification:
                        y[line_idx] = int(unit.strip())
                    else:
                        y[line_idx] = float(unit.strip())
                else:
                    feat, val = unit.strip().split(':')
                    feat: int = int(feat)
                    val: float = float(val)
                    x[line_idx][feat - 1] = val

        if is_classification:
            # To get classes in [0..n_classes-1]
            y = y - np.min(y)

        return Dataset(x, y, name=name, copy=False).shuffle(seed=shuffle_seed)

    @staticmethod
    def normalize_all_datasets(
        train: Dataset[np.ndarray],
        validn: Dataset[np.ndarray],
        test: Dataset[np.ndarray],
        normalize_x_kwargs: Optional[Dict[str, Any]]=None,
        normalize_y_kwargs: Optional[Dict[str, Any]]=None
    ) -> TVT:

        if normalize_x_kwargs is not None:
            train = train.normalize_x(**normalize_x_kwargs)
            validn = validn.normalize_x(category='mirror', mirror_params=train.mirror_x_params)
            test = test.normalize_x(category='mirror', mirror_params=train.mirror_x_params)

        if normalize_y_kwargs is not None:
            train = train.normalize_y(**normalize_y_kwargs)
            validn = validn.normalize_y(category='mirror', mirror_params=train.mirror_y_params)
            test = test.normalize_y(category='mirror', mirror_params=train.mirror_y_params)

        return train, validn, test

    @staticmethod
    def get_base_perf(train: Dataset[np.ndarray], validn: Dataset[np.ndarray], test: Dataset[np.ndarray]) -> str:
        assert cast(np.ndarray, train['x']).dtype in [np.float32]
        assert cast(np.ndarray, train['y']).dtype in [np.int64, np.float32]
        assert cast(np.ndarray, validn['x']).dtype == cast(np.ndarray, train['x']).dtype
        assert cast(np.ndarray, validn['y']).dtype == cast(np.ndarray, train['y']).dtype
        assert cast(np.ndarray, test['x']).dtype == cast(np.ndarray, train['x']).dtype
        assert cast(np.ndarray, test['y']).dtype == cast(np.ndarray, train['y']).dtype

        sep = '------------------\n'
        ret = ''
        ret += sep
        ret += ' Base Perf\n'
        ret += sep

        is_classification = cast(np.ndarray, train['y']).dtype == np.int64
        n_features, n_classes = DataLoader.stats[train.name]['n_features'], DataLoader.stats[train.name]['n_classes']

        if is_classification:
            mode = stats.mode(train['y'])[0][0]
            model = FixedConstantPredictor(n_features, n_classes, mode)
        else:
            mean = cast(np.ndarray, train['y']).mean()
            model = FixedConstantPredictor(n_features, n_classes, mean)
        model.acc_func, model.acc_func_type = Utils.get_acc_def(is_classification)

        ret += ' train_acc={:.5f}\n'.format(model.acc(train))
        ret += ' validn_acc={:.5f}\n'.format(model.acc(validn))
        ret += ' test_acc={:.5f}\n'.format(model.acc(test))

        ret += sep
        return ret

    @staticmethod
    def is_classification(dataset: str) -> bool:
        n_classes = DataLoader.stats[dataset]['n_classes']
        return n_classes > 0

    @staticmethod
    def download(url: str, dst: str):
        if os.path.exists(dst):
            print(f'Using dataset from: {dst}')
            return False

        os.makedirs(os.path.dirname(dst), exist_ok=True)

        print(f'Downloading dataset from: {url}...', end='')
        r = requests.get(url)
        print('Done')

        print(f'Writing to: {dst}...', end='')
        if url.endswith('tgz'):
            os.mkdir(dst)
            tarobj = tarfile.open(fileobj=io.BytesIO(r.content))
            for i in tarobj.getnames():
                with open(f'{dst}/{os.path.basename(i)}', 'w') as f:
                    f.write(tarobj.extractfile(i).read().decode('utf-8'))
            tarobj.close()

        elif url.endswith('zip'):
            os.mkdir(dst)
            with zipfile.ZipFile() as zipobj:
                for i in zipobj.namelist():
                    with zipobj.open(i) as zipfile:
                        with open(f'{dst}/{i}', 'w') as f:
                            f.write(zipfile.read())

        else:
            if url.endswith('bz2'):
                towrite = bz2.decompress(r.content).decode('utf-8')
            else:
                towrite = r.text

            with open(dst, 'w') as f:
                f.write(towrite)
        print('Done')

        return True