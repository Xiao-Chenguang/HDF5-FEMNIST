import os
import pandas as pd
from PIL import Image
from tqdm import tqdm


root = 'data/'
size = (28, 28)


# get the writer info from by_class
df = pd.DataFrame(columns=['file', 'target', 'label'])

# get the writer id for each image
for label in range(30,40):
    for group in range(8):
        if not os.path.exists(f'{root}by_class/{label}/hsf_{group}.mit'):
            continue
        mit_path = f'{root}by_class/{label}/hsf_{group}.mit'
        temp_df = pd.read_csv(mit_path, sep=' ', header=None, skiprows=1, names=['file', 'target'])
        temp_df['label'] = label
        temp_df['writer'] = temp_df['target'].apply(lambda x: x.split('/')[0])
        temp_df['target'] = temp_df.apply(lambda x: f'write_digits/{x["writer"]}/{x["label"]}/', axis=1)
        temp_df['source'] = f'by_class/{label}/hsf_{group}/'
        df = pd.concat([df, temp_df], ignore_index=True)
df = df[['file', 'source', 'target', 'writer', 'label']]


writer_list = df['writer'].unique()
label_list = df['label'].unique()

# create all directories in advance
if not os.path.exists(root + 'write_digits'):
    os.makedirs(root + 'write_digits')

for writer in writer_list:
    if not os.path.exists(root + f'write_digits/{writer}'):
        os.makedirs(root + f'write_digits/{writer}')
    for label in label_list:
        if not os.path.exists(root + f'write_digits/{writer}/{label}'):
            os.makedirs(root + f'write_digits/{writer}/{label}')


print('Group images by writer...')
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    source = root + row['source'] + row['file']
    target = root + row['target'] + row['file']
    # print(source, target)
    # print(os.path.exists(source), os.path.exists(target))
    # read image
    img = Image.open(source)
    gray = img.convert('L')
    gray.thumbnail(size, Image.ANTIALIAS)
    # save gray image to target
    gray.save(target)


# convert original image to vector of 28 * 28 = 784
# same as in LEAF: https://github.com/TalwalkarLab/leaf/blob/master/data/femnist/preprocess/data_to_json.py
# line 62 - 68

# img = Image.open(file_path)
# gray = img.convert('L')
# gray.thumbnail(size, Image.ANTIALIAS)
# arr = np.asarray(gray).copy()
# vec = arr.flatten()
# vec = vec / 255  # scale all pixel values to between 0 and 1
# vec = vec.tolist()