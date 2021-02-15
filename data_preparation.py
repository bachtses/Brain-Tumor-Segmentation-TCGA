import tensorflow as tf
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
tf.enable_eager_execution()


'''
PUT all the photos from DATASET TCGA folder mixed imgs and masks
RUN this program
IT WILL separate photos into 2 folders: Images and Masks

'''

IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 3

DATASET_TRAINING_PATH = 'DATASET_Training/'

path, dirs, files = next(os.walk(DATASET_TRAINING_PATH))
data = len(files)
print("\n")
print("FOUND: ", data, "folders")
print("\n")

k = 0

for item in os.listdir(DATASET_TRAINING_PATH):  # Iterate Over Each Image
    if "mask" in item:
        # print(item)
        data = cv2.imread(os.path.join(DATASET_TRAINING_PATH, item))
        data = cv2.resize(data, (IMG_WIDTH, IMG_HEIGHT))
        # Convert BGR to RGB
        data_RGB = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        # plt.imshow(data_RGB)
        # plt.show()
        savepath = 'Masks/'
        cv2.imwrite(os.path.join(savepath, item), cv2.cvtColor(data_RGB, cv2.COLOR_RGB2BGR))
    else:
        # print(item)
        data = cv2.imread(os.path.join(DATASET_TRAINING_PATH, item))
        data = cv2.resize(data, (IMG_WIDTH, IMG_HEIGHT))
        # Convert BGR to RGB
        data_RGB = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        # plt.imshow(data_RGB)
        # plt.show()
        SAVE_PATH = 'Images/'
        cv2.imwrite(os.path.join(SAVE_PATH, item), cv2.cvtColor(data_RGB, cv2.COLOR_RGB2BGR))

    k = k+1
