import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from xcommon import Utils
from xconstants import (FPTYPE, INIT_MODEL_WITH_CART, REMOTE_SESS,
                        SAT_INFO_NODE_MAX)

if REMOTE_SESS:
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn.functional as F


class ENUtils:
    @staticmethod
    def plot_dense_progress(src_file: str, dst_file: str, start_point: int=0):
        fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(15, 15))
        df = pd.read_csv(src_file)
        msize = 10

        # Comparing train and val
        # train_acc, validn_acc
        ax[0][0].set_xlabel('Epochs done')
        ax[0][0].set_ylabel('Train vs Val acc')
        ax[0][0].plot(df['epoch'][start_point:], df['train_acc'][start_point:], label='train_acc', marker='.', markersize=msize)
        ax[0][0].plot(df['epoch'][start_point:], df['validn_acc'][start_point:], label='validn_acc', marker='.', markersize=msize)
        ax[0][0].grid()
        ax[0][0].legend()

        # Comparing train and val
        # dt_train_acc, dt_validn_acc
        ax[0][1].set_xlabel('Epochs done')
        ax[0][1].plot(df['epoch'][start_point:], df['dt_train_acc'][start_point:], label='dt_train_acc', marker='.', markersize=msize)
        ax[0][1].plot(df['epoch'][start_point:], df['dt_validn_acc'][start_point:], label='dt_validn_acc', marker='.', markersize=msize)
        ax[0][1].grid()
        ax[0][1].legend()

        # Comparing EN and DT
        # train_acc, dt_train_acc
        ax[1][0].set_xlabel('Epochs done')
        ax[1][0].set_ylabel('EN vs ENDT acc')
        ax[1][0].plot(df['epoch'][start_point:], df['train_acc'][start_point:], label='train_acc', marker='.', markersize=msize)
        ax[1][0].plot(df['epoch'][start_point:], df['dt_train_acc'][start_point:], label='dt_train_acc', marker='.', markersize=msize)
        ax[1][0].grid()
        ax[1][0].legend()

        # Comparing EN and DT
        # validn_acc, dt_validn_acc
        ax[1][1].set_xlabel('Epochs done')
        ax[1][1].plot(df['epoch'][start_point:], df['validn_acc'][start_point:], label='validn_acc', marker='.', markersize=msize)
        ax[1][1].plot(df['epoch'][start_point:], df['dt_validn_acc'][start_point:], label='dt_validn_acc', marker='.', markersize=msize)
        ax[1][1].grid()
        ax[1][1].legend()

        # Comparing train and val
        # train_loss, validn_loss
        ax[2][0].set_xlabel('Epochs done')
        ax[2][0].set_ylabel('Train vs Val loss')
        ax[2][0].plot(df['epoch'][start_point:], df['train_loss'][start_point:], label='train_loss', marker='.', markersize=msize)
        ax[2][0].plot(df['epoch'][start_point:], df['validn_loss'][start_point:], label='validn_loss', marker='.', markersize=msize)
        ax[2][0].grid()
        ax[2][0].legend()

        # Comparing train and val with regularization
        ax[2][1].set_xlabel('Epochs done')
        ax[2][1].plot(df['epoch'][start_point:], df['train_loss_wreg'][start_point:], label='train_loss_wreg', marker='.', markersize=msize)
        ax[2][1].plot(df['epoch'][start_point:], df['validn_loss_wreg'][start_point:], label='validn_loss_wreg', marker='.', markersize=msize)
        ax[2][1].grid()
        ax[2][1].legend()

        # sigslope
        label = 'stepsize' if 'stepsize' in df.columns else 'xsigslopes'
        ax[3][0].set_xlabel('Epochs done')
        ax[3][0].plot(df['epoch'][start_point:], df['xsigslope'][start_point:], label='xsigslope', marker='.', markersize=msize)
        ax[3][0].grid()
        ax[3][0].legend()

        # threshold
        ax[3][1].set_xlabel('Epochs done')
        ax[3][1].plot(df['epoch'][start_point:], df['xthreshold'][start_point:], label='xthreshold', marker='.', markersize=msize)
        ax[3][1].grid()
        ax[3][1].legend()

        # lr
        ax[4][0].set_xlabel('Epochs done')
        ax[4][0].plot(df['epoch'][start_point:], df['lr_pred'][start_point:], label='lr', marker='.', markersize=msize)
        ax[4][0].grid()
        ax[4][0].legend()

        # Comparing train and val
        # cdt_train_acc, cdt_validn_acc
        ax[4][1].set_xlabel('Epochs done')
        ax[4][1].plot(df['epoch'][start_point:], df['cdt_train_acc'][start_point:], label='cdt_train_acc', marker='.', markersize=msize)
        ax[4][1].plot(df['epoch'][start_point:], df['cdt_validn_acc'][start_point:], label='cdt_validn_acc', marker='.', markersize=msize)
        ax[4][1].grid()
        ax[4][1].legend()

        plt.tight_layout()
        plt.savefig(dst_file)
        plt.close(fig)

    @staticmethod
    def plot_sat_info(src_file: str, dst_file: str, start_point: int=0):
        sat_info = Utils.read_json(src_file)
        int_nodes = 2 ** sat_info['height'] - 1
        node_safe = True
        if int_nodes > SAT_INFO_NODE_MAX:
            int_nodes = 1
            node_safe = False

        fig, ax = plt.subplots(nrows=int_nodes, ncols=1, figsize=(15, 6 * int_nodes), squeeze=False)
        msize = 6
        sat_info_np = { k: np.array(v) for k, v in sat_info.items() }

        ax[0][0].set_title('Fraction of predicate layer\'s outputs (given all training data) whose absolute difference from 0.5 that fall in various intervals. \nInterval closer to 0.5 implies activation value closer to 0/1 implies smaller gradient')
        for i in range(int_nodes):
            ax[i][0].set_xlabel('Epochs')
            ax[i][0].set_ylabel('Fraction of pred outputs at {}'.format('all internal nodes combined' if not node_safe else 'node {}'.format(i)))
            for j in sat_info_np.keys():
                if j != 'epoch' and j != 'height':
                    ax[i][0].plot(sat_info_np['epoch'][start_point:], sat_info_np[j][start_point:, i], label=j, marker='.', markersize=10)
            ax[i][0].grid()
            ax[i][0].legend()

        plt.savefig(dst_file)
        plt.close(fig)

class XLinear(nn.Module):
    """
    Provides more options to nn.Linear.

    If 'weight' is not None, fixes the weights of the layer to this.

    If 'bias' is None, it means that bias is learnable. In this case, whether all bias units
    should have the same bias or not is given by 'same'.

    If 'bias' is not None, then the provided value is assumed to the fixed bias (that is not
    updated/learnt). The value of 'same' is ignored here.

    Notes:
        - Number of neurons is out_features
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight: Optional[torch.Tensor]=None,
        bias: Optional[torch.Tensor]=None,
        same: bool=False
    ):
        super().__init__()

        self._l = nn.Linear(in_features, out_features, bias=False)

        if weight is not None:
            self._l.weight = nn.Parameter(weight, requires_grad=False)

        if bias is None:
            self._bias = Utils.get_initialized_bias(in_features, 1 if same else out_features)
        else:
            self._bias = nn.Parameter(bias, requires_grad=False)

    def forward(self, x):
        return self._l(x) + self._bias

    @property
    def weight(self):
        return self._l.weight

    @property
    def bias(self):
        return self._bias

class Binarizer1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input >= 0).to(dtype=FPTYPE)#, 1

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.abs() > 1] = 0
        return grad_input

class Binarizer2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input >= 0).to(dtype=FPTYPE)#, 1

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input *= F.relu(1-input.abs())
        return grad_input

class ScaleBinarizer1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        fac = torch.norm(input,p=1)/np.prod(input.shape)
        ctx.mark_non_differentiable(fac)
        return  (input >= 0).to(dtype=FPTYPE)*fac, fac

    @staticmethod
    def backward(ctx, grad_output, grad_fac):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.abs() > 1] = 0
        return grad_input

class NoiseScaleBinarizer1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, randbinprob):
        #ctx.save_for_backward(input)
        fac = torch.norm(input,p=1)/np.prod(input.shape)
        mask = (torch.rand_like(input) < randbinprob).to(dtype=FPTYPE)
        ctx.save_for_backward(input, mask)
        ctx.mark_non_differentiable(fac)
        return  (1-mask)*input + (input >= 0).to(dtype=FPTYPE)*fac*mask, fac

    @staticmethod
    def backward(ctx, grad_output, grad_fac):
        input, mask = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.abs() > 1] = 0
        return grad_input,None

class Sparser1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        inpargmax = input.argmax(-1)
        output = torch.zeros_like(input)
        output[torch.arange(input.shape[0]), inpargmax] = 1
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input

class NoiseSoftSparser1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, randbinprob):
        #ctx.save_for_backward(input)
        inpargmax = input.argmax(-1)
        mask = (torch.rand_like(input) < randbinprob).to(dtype=FPTYPE)
        ctx.save_for_backward(input, mask)
        output = torch.zeros_like(input)
        output[torch.arange(input.shape[0]), inpargmax] = 1
        return (1-mask)*input + output*mask

    @staticmethod
    def backward(ctx, grad_output):
        input, mask = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input,None

class LSQ(torch.autograd.Function):
    qn, qp = 0, 1

    """
    input: (batch_size, n_nodes)
    stepsize: () single item

    Returns:
        (batch_size, n_nodes)
    """
    @staticmethod
    def forward(ctx, input: torch.Tensor, stepsize: torch.Tensor) -> torch.Tensor:
        # "Tensor arguments that track history (i.e., with requires_grad=True) will be converted to ones that don’t track history before the call,
        # and their use will be registered in the graph", so no need for no_grad()
        scaled_input = (input / stepsize)
        ctx.save_for_backward(scaled_input)
        return scaled_input.clamp(-LSQ.qn, LSQ.qp).round() * stepsize

    """
    grad_output: (batch_size, n_nodes)

    Returns:
        (batch_size, n_nodes), (batch_size,)
    """
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scaled_input, = ctx.saved_tensors

        grad_input = grad_output.clone()
        grad_input[(scaled_input >= LSQ.qp).logical_or(scaled_input <= -LSQ.qn)] = 0

        grad_stepsize = -scaled_input + scaled_input.round()
        grad_stepsize[scaled_input <= -LSQ.qn] = -LSQ.qn
        grad_stepsize[scaled_input >= LSQ.qp] = LSQ.qp
        grad_stepsize = grad_stepsize * grad_output
        grad_stepsize = grad_stepsize.sum(dim=1)

        # step size gradient scale
        grad_stepsize = grad_stepsize * (1 / math.sqrt(grad_output.shape[1] * LSQ.qp))

        return grad_input, grad_stepsize


class LSQLayer(nn.Module):
    init_when_cart = 2

    def __init__(self):
        super().__init__()

        self._s = nn.Parameter(torch.tensor(1, dtype=FPTYPE), requires_grad=True)
        self._stepsize_init = False

    """
    x: (n_samples, out_features)
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._stepsize_init:
            with torch.no_grad():
                if INIT_MODEL_WITH_CART:
                    self._s.fill_(LSQLayer.init_when_cart)
                else:
                    self._s.fill_(2 * x.abs().mean().item() / math.sqrt(LSQ.qp))
                    # self._s.fill_(2 * x.norm(dim=1).mean().item() / math.sqrt(LSQ.qp))
            self._stepsize_init = True

        return LSQ.apply(x, self._s)

    @property
    def stepsize(self) -> torch.Tensor:
        return self._s
