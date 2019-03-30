import pickle
import os
import numpy as np
import zipfile
import pandas as p

from PIL import Image
from matplotlib.image import imsave


def preprocess_data(d_path, l_path, zipfile_path = 'train.zip', extract_dir = 'data'):
    '''get centered 32x32 patches of image and save in
    pickle format'''
    print('First launch. Data preprocessing started.')
    zip_ref = zipfile.ZipFile(zipfile_path, 'r')
    if not os.path.exists(extract_dir):
        os.mkdir(extract_dir)
    zip_ref.extractall(extract_dir)
    zip_ref.close()

    data_path = extract_dir
    labels_db = p.read_csv('labels_csv', sep = ',')

    all_images = []
    all_labels = []
    for _, _, files in os.walk(data_path):
        for file in files:
            img = Image.open(data_path + file)
            img = np.array(img)
            if img.shape[0] == 96:
                img = img[32:64, 32:64, :]
                all_images.append(img)
                all_labels.append(labels_db['label'][np.where(labels_db['id'] == file)])

    with open(d_path, 'wb') as f:
        pickle.dump(all_images, f)
    with open(l_path, 'wb') as f:
        pickle.dump(all_labels, f)


def load_data(data_path, labels_path = None):
    '''fetch data from pickle format'''
    if not os.path.exists(data_path):
        preprocess_data(data_path, labels_path)

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    if labels_path:
        with open(labels_path, 'rb') as f:
            labels = pickle.load(f)
        return data, labels

    return data
preprocess_data('./')




