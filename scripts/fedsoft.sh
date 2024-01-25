for trial in 1 2 3 4 5
do
    dir='../save_results/fedsoft/pathological/cifar10'
    if [ ! -e $dir ]; then
    mkdir -p $dir
    fi
    current_time=$(date +"%Y%m%d-%H%M%S")
    current_time_safe=$(echo "$current_time" | tr ':' '_')
    python ../main_fedsoft.py --trial=$trial \
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
    --alg='fedsoft' \
    --gpu=0 \
    --nclusters=5 \
    --estimation_interval=2 \
    2>&1 | tee $dir'/'$current_time_safe'_1.txt'

done
