# HDF5-FEMNIST

[![MacOS](https://img.shields.io/badge/platform-macOS-blue)](https://www.apple.com/macos/)
[![Linux](https://img.shields.io/badge/platform-Linux-blue)](https://www.linux.org/)
[![PyTorch](https://img.shields.io/badge/framework-PyTorch-orange)](https://pytorch.org/)

HDF5-FEMNIST enables easy access and fast loading to the FEMNIST dataset from [LEAF](https://leaf.cmu.edu) with the help of [HDF5](https://github.com/h5py/h5py).

There are currently limited public accessable federated datasets for research purpose. FEMNIST is one of the most popular and early datasets implemented under LEAF framework which is rarely used by majorities of researchers nowadays. [Tensorflow-Federated](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/emnist/load_data) has a buildin version of FEMNIST. But for the TyTorch users, there are now easy access to the FEMNIST datasets.

According to the idea of creating FEMNIST based on [NIST](https://www.nist.gov/srd/nist-special-database-19) dataset, this repo simplify the producdures of spliting the dataset into different users. Then the dataset is converted into HDF5 format for easy access and fast loading. Loading all ther writer datasets using PyTorch ImageFolder takes about 10 to 30 minutes, while loading the same dataset using HDF5 takes only a few seconds.

This repo is able to generate the HDF5 datasets for each writer in two setting. The first is digits only datasets, which contains only digits from 0 to 9. The second is the full datasets, which contains all the 62 classes of characters. We have to note that not all the writers have all the 62 classes of characters or even all 10 digits in digits only mode.
 The datasets are generated in the following structure:
```
HDF5-FEMNIST
├── f0000_14
│   ├── images
│   │   ├── n0 * 28 * 28 
│   └── labels
│       ├── n0 * 1
├── f0001_41
│   ├── images
│   │   ├── n1 * 28 * 28
│   └── labels
│       ├── n1 * 1
├── ...
├── f4099_10
│   ├── images
│   │   ├── n4099 * 28 * 28
│   └── labels
│       ├── n4099 * 1
```

In digits only mode, the labels are in the range of 0 to 9. In full mode, the labels are in the range of 0 to 61. The mapping from the label to the character is in the following table:

| Character |   0   |  ...  |   9   |   A   |  ...  |   Z   |   a   |  ...  |   z   |
| --------- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Label** |   0   |  ...  |   9   |  10   |  ...  |  35   |  36   |  ...  |  61   |


## Usage

To use the datasets, just clone the repo and run the following command:
```bash
./get_data.sh
```

You can specify the mode of the dataset by adding the argument `-d` followed by `True` (default, digits only mode) or `False` (all characters mode).

After converting the dataset to HDF5 format, you can use the following code to load the dataset:
```python
import h5py

# load the dataset
dataset = h5py.File('path_to_your_femnist.hdf5', 'r')

# get the key of each writer datasets
writers = sorted(dataset.keys())

# get the images and labels of the first writer as numpy array
images = dataset[writers[0]]['images']
labels = dataset[writers[0]]['labels']

# transform the images and labels to torch tensor
images_tensor = torch.from_numpy(images)
labels_tensor = torch.from_numpy(labels)
```

We also provide a [demo notebook](./demo.ipynb) to explore properties of the dataset. Feel free to play with it.

## Requirements
python with the following packages:
- h5py
- numpy
- Pillow
- tqdm
- pandas