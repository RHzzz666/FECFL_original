#!/bin/bash
for seed in 66
do
    dir='../save_results/shift/rotate/fedsem/rotated/cinic10'
    if [ ! -e $dir ]; then
    mkdir -p $dir
    fi
    current_time=$(date +"%Y%m%d-%H%M%S")
    current_time_safe=$(echo "$current_time" | tr ':' '_')
    python ./main_fedsem_shift.py --trial=1 \
    --rounds=50 \
    --num_users=100 \
    --frac=1 \
    --local_ep=10 \
    --local_bs=64 \
    --lr=0.01 \
    --momentum=0.5 \
    --model=simple-cnn \
    --dataset=cinic10 \
    --datadir='../data/' \
    --logdir='../logs/' \
    --savedir='../save_results/' \
    --partition='rotated' \
    --alg='fedsem' \
    --gpu=2 \
    --ncluster=4 \
    --seed=$seed \
    --shift_type='rotate' \
    --swap_p=0.5 \
    2>&1 | tee $dir'/'$current_time_safe'_1.txt'

done
