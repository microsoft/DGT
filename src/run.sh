RunDGT(){
    python main.py --dataset "$1"  --height "$2" \
    --sigslope_routine constant --sigquant ScaleBinarizer1 \
    --softslope_routine constant --softquant SoftSparser1 \
    --optimizer "$3" --optimizer_kw medium \
    --lr1 1e-2 --lr_sched "$4" --batch_sizes 128 \
    --reglist 1e-5,5e-6,1e-6 --use_l1_reg t --use_l2_reg f --use_no_reg f \
    --grad_clip 0.01 --pred_reg t --ulm f \
    --over_param "$7" \
    --ndshuffles "$5" --epochs "$6" --proc_per_gpu 4 --num_gpu 4 \
    -d "$1"_"$2"

}

RunDGT ailerons 6 RMS CosineWarm 1 200 [[16,16]] ;

# RunDGT abalone 6 RMS CosineWarm 5 200 [[16,16]] ;

# RunDGT satimage 6 RMS CosineWarm 1 200 [[16,16]] ;

# RunDGT pendigits 6 RMS CosineWarm 1 200 [[16,16]] ;
