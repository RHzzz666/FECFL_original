for seed in 39
do
    dir='../save_results/fecfl/pathological/cifar10'
    if [ ! -e $dir ]; then
    mkdir -p $dir
    fi
    current_time=$(date +"%Y%m%d-%H%M%S")
    current_time_safe=$(echo "$current_time" | tr ':' '_')
    python ../main_fecfl.py --trial=1 \
    --rounds=200 \
    --num_users=100 \
    --frac=0.1 \
    --local_ep=10 \
    --local_bs=64 \
    --lr=0.01 \
    --momentum=0.5 \
    --model=simple-cnn \
    --dataset=cifar10 \
    --datadir='../data/' \
    --logdir='../logs/' \
    --savedir='../save_results/' \
    --partition='pathological' \
    --alg='fecfl' \
    --beta=0.1 \
    --local_view \
    --noise=0 \
    --cluster_alpha=0.00025 \
    --n_basis=5 \
    --linkage='average' \
    --gpu=1 \
    --print_freq=10 \
    --seed=$seed \
    2>&1 | tee $dir'/'$current_time_safe'_1.txt'

done

# CIFAR10 5 groups:
# seed=42, 0.002
# seed=1234, 0.0008
# seed=0, 0.0004
# seed=88, 0.0004
# seed=1, 0.0008
# seed=39 0.00025