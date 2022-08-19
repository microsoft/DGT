import numpy as np
import pandas as pd
from typing import Any, Set
import os
from xdata import DataLoader
import functools

from xconstants import REMOTE_SESS
if REMOTE_SESS:
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

def hpsearch_stats(summary_path: str, dataset_name: str):
    ms = ([
        'train_acc', 'validn_acc', 'test_acc',
        'dt_train_acc', 'dt_validn_acc', 'dt_test_acc',
        'cdt_train_acc', 'cdt_validn_acc', 'cdt_test_acc',

        'train_auc', 'validn_auc', 'test_auc',
        'dt_train_auc', 'dt_validn_auc', 'dt_test_auc',
        'cdt_train_auc', 'cdt_validn_auc', 'cdt_test_auc',
    ])
    df = pd.read_csv(summary_path)

    # analyze hyperparams, aka columns whose values change
    ignore_cols = ['config_idx', 'device_ids', 'config_dir', 'sat_info'] # columns whose values change but are not hyperparams
    ignore_cols.extend(ms)
    analyze_cols = set(df.columns) - set(ignore_cols)
    s = df.nunique()
    analyze_cols = analyze_cols.intersection(set(s.index[s > 1]))

    plots_path = f'{os.path.split(summary_path)[0]}/hpsearch_stats/{dataset_name}'
    try:
        os.makedirs(plots_path)
    except FileExistsError:
        pass
    is_classification = DataLoader.is_classification(dataset_name)

    heights = [None]
    if 'height' in df.columns:
        heights = list(df['height'].unique())

    for height in heights:
        dft = df
        path = plots_path
        if height is not None:
            dft = df.loc[df['height'] == height]
            path = f'{plots_path}/{height}'
            try:
                os.mkdir(path)
            except FileExistsError:
                pass

        if 'train_acc' in df.columns:
            plot(df=dft, analyze_cols=analyze_cols, metric='train_acc', plots_path=path, is_classification=is_classification)
            plot(df=dft, analyze_cols=analyze_cols, metric='validn_acc', plots_path=path, is_classification=is_classification)
            plot(df=dft, analyze_cols=analyze_cols, metric='test_acc', plots_path=path, is_classification=is_classification)
        if 'dt_train_acc' in df.columns:
            plot(df=dft, analyze_cols=analyze_cols, metric='dt_train_acc', plots_path=path, is_classification=is_classification)
            plot(df=dft, analyze_cols=analyze_cols, metric='dt_validn_acc', plots_path=path, is_classification=is_classification)
            plot(df=dft, analyze_cols=analyze_cols, metric='dt_test_acc', plots_path=path, is_classification=is_classification)
        if 'train_auc' in df.columns:
            plot(df=dft, analyze_cols=analyze_cols, metric='train_auc', plots_path=path, is_classification=is_classification)
            plot(df=dft, analyze_cols=analyze_cols, metric='validn_auc', plots_path=path, is_classification=is_classification)
            plot(df=dft, analyze_cols=analyze_cols, metric='test_auc', plots_path=path, is_classification=is_classification)
        if 'dt_train_auc' in df.columns:
            plot(df=dft, analyze_cols=analyze_cols, metric='dt_train_auc', plots_path=path, is_classification=is_classification)
            plot(df=dft, analyze_cols=analyze_cols, metric='dt_validn_auc', plots_path=path, is_classification=is_classification)
            plot(df=dft, analyze_cols=analyze_cols, metric='dt_test_auc', plots_path=path, is_classification=is_classification)

def ordering_keyf(col_name: str, x: Any) -> Any:
    if col_name == 'over_param':
        if x.strip() == '':
            return ()
        return tuple([int(float(i.strip())) for i in x.split(',')])

    if isinstance(x, int) or isinstance(x, float):
        return x
    else:
        try:
            ret = eval(x) # things like l1_lambda (i.e. non empty tuples) will be taken care here
        except:
            ret = x # strings like optimizer

        try:
            ret < ret
            return ret
        except TypeError as e:
            return str(ret)
"""
metric: metric used to select best config within a group
"""
def plot(df: pd.DataFrame, analyze_cols: Set[str], metric: str, plots_path: str, is_classification: bool):
    analyze_cols = list(analyze_cols - set(['height']))
    nrows = len(analyze_cols)
    if nrows > 0:
        analyze_cols = sorted(analyze_cols)
        fig, axs = plt.subplots(nrows=nrows, ncols=3, figsize=(20, 3.5 * nrows), squeeze=False)
        fig.suptitle(f'Best {metric} | height={df.height.iloc[0] if "height" in df.columns else -1}')

        prefix = 'dt_' if 'dt' in metric else ''
        postfix = '_acc' if 'acc' in metric else '_auc'
        train_l, validn_l, test_l = prefix + 'train' + postfix, prefix + 'validn' + postfix, prefix + 'test' + postfix

        markers = ['o', '^', 's']
        markersize = 5
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        for i, col in enumerate(analyze_cols):
            gb = df.loc[~df[metric].isna()].groupby(by=col, dropna=False)[metric]
            dft = df.loc[gb.idxmax()] if is_classification else df.loc[gb.idxmin()]
            dft = dft.copy()
            dft[col] = dft[col].fillna('' if dft[col].dtype == np.dtype('O') else -3.14) # for now, only to handle over_param case
            dft['__aux'] = dft[col].apply(functools.partial(ordering_keyf, col))
            dft = dft.sort_values(by='__aux')

            x = range(len(dft))
            axs[i][0].plot(x, dft[train_l], marker=markers[0], markersize=markersize, label=train_l, color=colors[0])
            axs[i][1].plot(x, dft[validn_l], marker=markers[1], markersize=markersize, label=validn_l, color=colors[1])
            axs[i][2].plot(x, dft[test_l], marker=markers[2], markersize=markersize, label=test_l, color=colors[2])

            for ax in axs[i]:
                ax.set_xticks(x)
                ax.set_xticklabels(dft[col])
                ax.grid()
                ax.legend()
                ax.set_xlabel(col)

        plt.tight_layout()
        plt.savefig(f'{plots_path}/{metric}.png')
        plt.close(fig)