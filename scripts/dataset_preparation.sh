dir='../save_results/preparation/pathological/cifar10'
if [ ! -e $dir ]; then
mkdir -p $dir
fi
current_time=$(date +"%Y%m%d-%H%M%S")
current_time_safe=$(echo "$current_time" | tr ':' '_')
python ../datasets_models.py --trial=1 \
--dataset=cifar10 \
--datadir='../data/' \
--logdir='../logs/' \
--savedir='../save_results/' \
--partition='pathological' \
2>&1 | tee $dir'/'$current_time_safe'_1.txt'


# options for dataset: fmnist, cifar10, cinic, stl10, cifar100, tiny
# options for partition: pathological, rotated, rgb_hsv, mix, noniid-#label2(heterogeneous)
# special for stl10: pathological#2

