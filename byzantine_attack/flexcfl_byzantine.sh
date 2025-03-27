#!/bin/bash

#####使用噪声注入攻击（指定0-30号客户端，中等强度）
 for seed in 66
 do
     dir='../save_results/byzantine/noise_specific/flexcfl/cifar10'
     if [ ! -e $dir ]; then
     mkdir -p $dir
     fi
     current_time=$(date +"%Y%m%d-%H%M%S")
     current_time_safe=$(echo "$current_time" | tr ':' '_')
     python ./main_flexcfl_byzantine.py --trial=1 \
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
     --partition='homo' \
     --alg='flexcfl' \
     --nclusters=1 \
     --pretrain_epoch=10 \
     --gpu=2 \
     --print_freq=10 \
     --seed=$seed \
     --shift_type='noise_injection' \
     --random_target \
     --malicious_clients="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29" \
     --noise_ratio=0.25 \
     --noise_type='pure' \
     --noise_level=0.3 \
     2>&1 | tee $dir'/'$current_time_safe'_noise_specific.txt'
 done


 # 使用模糊后门攻击（指定0-30号客户端，中等强度）
 for seed in 66
 do
     dir='../save_results/byzantine/backdoor_hsv/flexcfl/cifar10'
     if [ ! -e $dir ]; then
     mkdir -p $dir
     fi
     current_time=$(date +"%Y%m%d-%H%M%S")
     current_time_safe=$(echo "$current_time" | tr ':' '_')
     python ./main_flexcfl_byzantine.py --trial=1 \
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
     --partition='homo' \
     --alg='flexcfl' \
     --nclusters=1 \
     --pretrain_epoch=10 \
     --gpu=0 \
     --print_freq=10 \
     --seed=$seed \
     --shift_type='backdoor_hsv' \
     --malicious_clients="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29" \
     --trigger_ratio=0.25 \
     --blur_radius=3 \
     --target_class=0 \
     2>&1 | tee $dir'/'$current_time_safe'_backdoor_hsv.txt'
 done

######使用旋转对抗攻击（指定0-30号客户端，中等强度）
for seed in 66
do
    dir='../save_results/byzantine/rotation_adv_specific/flexcfl/cifar10'
    if [ ! -e $dir ]; then
    mkdir -p $dir
    fi
    current_time=$(date +"%Y%m%d-%H%M%S")
    current_time_safe=$(echo "$current_time" | tr ':' '_')
    python ./main_flexcfl_byzantine.py --trial=1 \
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
    --partition='homo' \
    --alg='flexcfl' \
    --nclusters=1 \
    --pretrain_epoch=10 \
    --gpu=2 \
    --print_freq=10 \
    --seed=$seed \
    --shift_type='rotation_adversarial' \
    --malicious_clients="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29" \
    --target_class=0 \
    2>&1 | tee $dir'/'$current_time_safe'_flexcfl_rotation_adv_specific.txt'
done
