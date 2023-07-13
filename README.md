# HDF5-FEMNIST

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
```

In digits only mode, the labels are in the range of 0 to 9. In full mode, the labels are in the range of 0 to 61. The mapping from the label to the character is in the following table:

| Label | Character | Label | Character | Label | Character | Label | Character |
| ----- | --------- | ----- | --------- | ----- | --------- | ----- | --------- |
| 0     | 0         | 16    | G         | 32    | W         | 48    | m         |
| 1     | 1         | 17    | H         | 33    | X         | 49    | n         |
| 2     | 2         | 18    | I         | 34    | Y         | 50    | o         |
| 3     | 3         | 19    | J         | 35    | Z         | 51    | p         |
| 4     | 4         | 20    | K         | 36    | a         | 52    | q         |
| 5     | 5         | 21    | L         | 37    | b         | 53    | r         |
| 6     | 6         | 22    | M         | 38    | c         | 54    | s         |
| 7     | 7         | 23    | N         | 39    | d         | 55    | t         |
| 8     | 8         | 24    | O         | 40    | e         | 56    | u         |
| 9     | 9         | 25    | P         | 41    | f         | 57    | v         |
| 10    | A         | 26    | Q         | 42    | g         | 58    | w         |
| 11    | B         | 27    | R         | 43    | h         | 59    | x         |
| 12    | C         | 28    | S         | 44    | i         | 60    | y         |
| 13    | D         | 29    | T         | 45    | j         | 61    | z         |
| 14    | E         | 30    | U         | 46    | k         |       |           |
| 15    | F         | 31    | V         | 47    | l         |       |           |

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
dataset = h5py.File('HDF5-FEMNIST/f0000_14', 'r')

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