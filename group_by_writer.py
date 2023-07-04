import os
import sys
import pandas as pd
from PIL import Image
from tqdm import tqdm


def main(root='data/', digits_only=False):

    size = (28, 28)
    split = 'write_digits' if digits_only else 'write_all'

    def class_list(digits_only=True):
        digits = [hex(i)[2:] for i in range(ord('0'), ord('9') + 1)]
        upper = [hex(i)[2:] for i in range(ord('A'), ord('Z') + 1)]
        lowwer = [hex(i)[2:] for i in range(ord('a'), ord('z') + 1)]

        return digits if digits_only else digits + upper + lowwer

    # get the writer info from by_class
    df = pd.DataFrame(columns=['file', 'target', 'label', 'writer', 'source'])

    # get the writer id for each image
    for label in class_list(digits_only=digits_only):
        for group in range(8):
            if not os.path.exists(f'{root}by_class/{label}/hsf_{group}.mit'):
                continue
            mit_path = f'{root}by_class/{label}/hsf_{group}.mit'
            temp_df = pd.read_csv(mit_path, sep=' ', header=None, skiprows=1, names=['file', 'target'])
            temp_df['label'] = label
            temp_df['writer'] = temp_df['target'].apply(lambda x: x.split('/')[0])
            temp_df['target'] = temp_df.apply(lambda x: f'{split}/{x["writer"]}/{x["label"]}/', axis=1)
            temp_df['source'] = f'by_class/{label}/hsf_{group}/'
            df = pd.concat([df, temp_df], ignore_index=True)
    df = df[['file', 'source', 'target', 'writer', 'label']]

    writer_list = df['writer'].unique()
    label_list = df['label'].unique()

    # create all directories in advance
    if not os.path.exists(root + split):
        os.makedirs(root + split)

    for writer in writer_list:
        if not os.path.exists(root + f'{split}/{writer}'):
            os.makedirs(root + f'{split}/{writer}')
        for label in label_list:
            if not os.path.exists(root + f'{split}/{writer}/{label}'):
                os.makedirs(root + f'{split}/{writer}/{label}')

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


if __name__ == '__main__':
    if len(sys.argv) == 2:
        main(digits_only=sys.argv[1].lower()=='true')
    else:
        main()

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
