import json
import argparse
from xconstants import BB_ENUM
from typing import List

def Opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--description', '-d', type=str, default='')
    parser.add_argument('--model', type=str, default='DGT')
    parser.add_argument('--dataset', type=str, default='cpu8k')
    parser.add_argument('--ndshuffles', type=str, default='1')
    parser.add_argument('--datanorm', type=str, default='t')
    parser.add_argument('--seed', type=str, default="1")
    parser.add_argument('--compute_mstd', type=str, default='t')

    ### xconstants moved
    parser.add_argument('--proc_per_gpu', type=int, default=4)
    parser.add_argument('--num_gpu', type=int, default=4)

    ### xdgt parser
    parser.add_argument('--height', type=str, default='2,4,6')

    parser.add_argument('--reglist', type=str, default=None) ## dont change default none, use cmdline
    parser.add_argument('--use_no_reg', type=str, default='t') ## dont change default none, use cmdline
    parser.add_argument('--use_l1_reg', type=str, default='f')
    parser.add_argument('--use_l2_reg', type=str, default='f')

    parser.add_argument('--grad_clips', type=str, default=None) ## dont change default none, use cmdline
    parser.add_argument('--lab', type=int, default=0)
    parser.add_argument('--eps', type=float, default=0.5)
    parser.add_argument('--and_act', type=str, default="softmax", choices=['softmax', 'relu', 'leaky_relu'])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_sizes', type=str, default="128")
    parser.add_argument('--ulm', type=str, default='f') # use last model or not
    parser.add_argument('--sigslope_routine',
                        type=str, default='constant',
                        choices=['constant', 'multiconstant', 'increase', 'largeincrease'])
    parser.add_argument('--softslope_routine',
                        type=str, default='increase',
                        choices=['constant', 'multiconstant', 'increase', 'logincrease', 'largeincrease',])
    #parser.add_argument('--and_layer', type=str, default="linear", choices=['linear', 'multiply'])
    parser.add_argument('--black_box', type=BB_ENUM, choices=list(BB_ENUM), default=None) ## dont change default none, use cmdline
    parser.add_argument('--thresholds', type=str, choices=['constant', 'general'], default='constant')
    parser.add_argument('--br', type=str, default="0.5")
    parser.add_argument('--sigquant', type=str, default=None) ## dont change default none, use cmdline
    parser.add_argument('--softquant', type=str, default=None) ## dont change default none, use cmdline
    parser.add_argument('--batchnorm', type=str, default='f')
    parser.add_argument('--pred_reg', type=str, default='f')
    parser.add_argument('--criterion', type=str, default=None)
    parser.add_argument('--optimizer', type=str, default='RMS')
    parser.add_argument('--lr_sched', type=str, default='CosineWarm')
    parser.add_argument('--optimizer_kw', type=str, default='default', choices=['default', 'medium', 'big', 'xbig', 'old', 'singleold'])
    parser.add_argument('--lr_sched_kw', type=str, default='default', choices=['default', 'big', 'old', 'h1', 'h2', 'const', 'ablation'])

    parser.add_argument('--lr1', type=str, default="1e-4,1e-3,1e-2,1e-1,1")
    parser.add_argument('--lr2', type=str, default=None) ## dont change default none, use cmdline

    parser.add_argument('--over_param', type=str, default='[[],[15],[1,1],[10,10],[20,20],[1,1,1],[10,10,10],[20,20,20],[5,20,5],[10,10,10,10]]')

    ### vw parser
    parser.add_argument('--epsilon', type=str, default='0.1') # list allowed
    parser.add_argument('--num_tlogs', type=str, default='2000')

    args = parser.parse_args()


    ############## PARSER utils below

    args.seed = [int(x) for x in args.seed.split(',')]
    args.datanorm = parse_tf(args.datanorm)
    args.compute_mstd = parse_tf(args.compute_mstd)

    ### xconstants moved
    if args.num_gpu == -1:
        args.DEVICES_INFO = [(-1, args.proc_per_gpu)]
    else:
        args.DEVICES_INFO = [(gpu_idx, args.proc_per_gpu) for gpu_idx in range(args.num_gpu)]

    ### xdgt
    args.height = [int(x) for x in args.height.split(',')]

    if args.reglist is None:
        args.reglist = []
    else:
        args.reglist = [float(x) for x in args.reglist.split(',')]

    args.use_no_reg = parse_tf(args.use_no_reg)
    args.use_l1_reg = parse_tf(args.use_l1_reg)
    args.use_l2_reg = parse_tf(args.use_l2_reg)

    if len(args.reglist)>0 and not (args.use_l2_reg or args.use_l1_reg):
        print("WARNING: len of reglist is non-zero but both use_l1_reg and use_l1_reg are False. Changing them to True")
        args.use_l1_reg = True
        args.use_l2_reg = True

    if args.grad_clips is None:
        args.grad_clips = [None]
    else:
        args.grad_clips = [float(x) if x != 'None' else None for x in args.grad_clips.split(',')]

    args.br = [float(x) for x in args.br.split(',')]

    if args.sigquant is None:
        args.sigquant = [None]
    else:
        args.sigquant = args.sigquant.split(',')

    if args.softquant is None:
        args.softquant = [None]
    else:
        args.softquant = args.softquant.split(',')

    args.batchnorm = list(set([parse_tf(x) for x in args.batchnorm.split(',')]))
    args.pred_reg = list(set([parse_tf(x) for x in args.pred_reg.split(',')]))
    args.ulm = parse_list_tf(args.ulm)

    args.lr1 = [float(x) for x in args.lr1.split(',')]

    if args.lr2 is None:
        args.lr2 = [None]
    else:
        args.lr2 = [float(x) for x in args.lr2.split(',')]

    args.batch_sizes = [int(x) for x in args.batch_sizes.split(',')]

    args.lr_sched = args.lr_sched.split(',')
    args.optimizer = args.optimizer.split(',')
    args.ndshuffles = int(args.ndshuffles)

    args.over_param = json.loads(args.over_param)

    ### vw
    args.epsilon = [float(x) for x in args.epsilon.split(',')]
    args.num_tlogs = int(args.num_tlogs)

    return args

def parse_list_tf(list_tf: str) -> List[bool]:
    return list(set([parse_tf(i) for i in list_tf.strip().split(',')]))

def parse_tf(tf: str) -> bool:
    clean = tf.strip().lower()
    if clean == 't':
        return True
    elif clean == 'f':
        return False
    else:
        raise ValueError(f'{tf} must be in ["t", "f"]')