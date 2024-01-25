#!/bin/bash
for seed in 66
do
    dir='../save_results/shift/rotate/pacfl/rotated/cifar10'
    if [ ! -e $dir ]; then
    mkdir -p $dir
    fi
    current_time=$(date +"%Y%m%d-%H%M%S")
    current_time_safe=$(echo "$current_time" | tr ':' '_')
    python ./main_pacfl_shift.py --trial=1 \
    --rounds=50 \
    --num_users=100 \
    --frac=1 \
    --local_ep=10 \
    --local_bs=64 \
    --lr=0.01 \
    --momentum=0.5 \
    --model=simple-cnn \
    --dataset=cifar10 \
    --datadir='../data/' \
    --logdir='../logs/' \
    --savedir='../save_results/' \
    --partition='rotated' \
    --alg='pacfl' \
    --cluster_alpha=15.0 \
    --n_basis=3 \
    --linkage='average' \
    --gpu=0 \
    --seed=$seed \
    --shift_type='rotate' \
    --swap_p=0.5 \
    2>&1 | tee $dir'/'$current_time_safe'_1.txt'

done
