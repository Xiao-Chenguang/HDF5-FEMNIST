import os
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm


def convert_to_hdf5(source, target):
    file = h5py.File(target, "w")

    writers = sorted(os.listdir(source))
    for writer in tqdm(writers):
        writer_group = file.create_group(writer)

        classes = sorted(os.listdir(os.path.join(source, writer)))
        imgs, labels = [], []
        for _class in classes:
            image_files = os.listdir(os.path.join(source, writer, _class))

            for image_file in image_files:
                image_path = os.path.join(source, writer, _class, image_file)
                image = Image.open(image_path).convert("L")
                image_np = np.array(image)
                imgs.append(image_np)
            labels.extend([int(_class)] * len(image_files))

        imgs = np.stack(imgs, axis=0)
        labels = np.array(labels)
        writer_group.create_dataset("images", data=imgs, dtype="uint8")
        writer_group.create_dataset("labels", data=labels, dtype="uint8")
    file.close()


if __name__ == "__main__":
    source = "path_to_images_root"
    target = "path_to_target.hdf5"
    convert_to_hdf5(source, target)
