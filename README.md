# Lightweight Clustered Federated Learning via Feature Extraction 

### Prepare the dataset
```bash
cd scripts
bash dataset_preparation.sh
```
Options for dataset: fmnist, cifar10, cinic, stl10, cifar100, tiny; 

Options for partition: pathological, rotated, rgb_hsv, mix, noniid-#label2(heterogeneous); Specially for stl10: pathological#2

### Run the code
```bash
cd scripts
bash fecfl.sh
```

### Unsuperivsed task
```bash
cd unsupervised
bash fecfl_unsupervised.sh
```

### Distribution shift
```bash
cd distribution_shift
bash fecfl_shift.sh
```
