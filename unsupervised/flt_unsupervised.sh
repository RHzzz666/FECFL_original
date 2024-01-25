for trial in 1 2 3 4 5
do
    dir='../save_results/unsupervised/flt/pathological/cifar10'
    if [ ! -e $dir ]; then
    mkdir -p $dir
    fi
    current_time=$(date +"%Y%m%d-%H%M%S")
    current_time_safe=$(echo "$current_time" | tr ':' '_')
    python ./main_flt_unsupervised.py --trial=$trial \
    --rounds=100 \
    --num_users=100 \
    --frac=0.1 \
    --local_ep=10 \
    --local_bs=64 \
    --lr=0.01 \
    --momentum=0.5 \
    --model=ae \
    --dataset=cifar10 \
    --datadir='../data/' \
    --logdir='../logs/' \
    --savedir='../save_results/' \
    --partition='pathological' \
    --alg='flt' \
    --gpu=0 \
    --nclusters=2 \
    --unsupervised=1 \
    2>&1 | tee $dir'/'$current_time_safe'_1.txt'

done
