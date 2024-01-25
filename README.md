# Lightweight Clustered Federated Learning via Feature Extraction 

### Prepare the dataset
```bash
cd scripts
bash dataset_preparation.sh
```

Options for dataset: ```fmnist```, ```cifar10```, ```cinic```, ```stl10```, ```cifar100```, ```tiny```; 

Options for partition: ```pathological```, ```rotated```, ```rgb_hsv```, ```mix```, ```noniid-#label2```(heterogeneous); Specially for ```stl10```: ```pathological#2```


For FMNIST, CIFAR10, STL10, CIFAR100, datasets are automatically downloaded.

For CINIC10, you should download the dataset from [here](https://datashare.is.ed.ac.uk/handle/10283/3192) and unzip in the folder `data` and rename it to `cinic10`.
For Tiny-ImageNet, you should download the dataset from [here](http://cs231n.stanford.edu/tiny-imagenet-200.zip) and unzip it in the folder `data`.

The directory structure of the datasets should look like this:

```
FECFL-->data-->cinic10-->data-->train
                |                |->test
                |                |->valid
                |->tiny-imagenet-200-->train
                |                     |->val
                |                     |->test
                |                     |->wnids.txt
                |                     |->words.txt
                ...
```


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
