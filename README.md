## Paper
[Learning Accurate Decision Trees with Bandit Feedback via Quantized Gradient Descent](https://arxiv.org/abs/2102.07567)

## Setup

1. Install necessary packages

Create a new conda environment named `dgt_env` with `python==3.6.8`, `pytorch==1.7.0` and install all dependencies inside:

```
$ conda env create -f dgt_env.yml
$ conda activate dgt_env
```

2. Change working directory to `src`:

```
$ cd src
```

3. Run the algorithm

To reproduce some of our results, please run `bash run.sh`.
- The script by default runs our algorithm with height 6 on `ailerons`. Commands for `abalone`, `satimage`, and `pendigits` are commented out.
- To change height of the tree learnt, change the argument corresponding to `--height` flag.
- The `--proc_per_gpu` option denotes how many processes to run per GPU. It defaults to 4 which is ideal for a typical GPU but on a GPU with small memory, reducing it from 4 might be required.
- The `--num_gpu` option denotes how many GPUs to parallelize over (and assumes device ordinal of GPUs start with 0). It defaults to 1.

Note: For `abalone` dataset we report the final performance across 5 different shuffles.

4. Check Results

Final scores, i.e. mean test RMSE/Accuracy and standard deviation, can be found in the file `./out/exp@{dataset}_{height}@{start_time}/meanstd-exps/meanstd-run-summary.csv` under the columns `test_acc_mean` and `test_acc_std`.

## Code Contributors

[Ajaykrishna Karthikeyan](https://github.com/ajay0)  
[Naman Jain](https://github.com/Naman-ntc)

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.
