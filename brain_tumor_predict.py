import tensorflow as tf
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
tf.enable_eager_execution()

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
n_classes = 2

#  Load Trained Model
model = tf.keras.models.load_model("my_model.h5")

#  Demonstrate Prediction
scores_matrix = np.zeros(0)
test_PATH = 'DATASET_Testing/'
for item in os.listdir(test_PATH):  # Iterate Over Each Image
    if "mask" not in item:
        path = os.path.join(test_PATH, item)
        img = cv2.imread(path)  # convert to array
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
        # print(np.shape(img))
        # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # plt.imshow(img)
        # plt.show()

        init_img = img
        img = np.expand_dims(img, axis=0)
        predict = model.predict(img, verbose=1)
        prediction_matrix = np.squeeze(predict[0])

        img_synthesis = np.zeros((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
        #  Labeling The Classes
        for i in range(IMG_HEIGHT):
            for j in range(IMG_WIDTH):
                if np.argmax(predict[0][i][j]) == 0:
                    img_synthesis[i][j] = np.array([255, 40, 40])
                # unlabeled
                elif np.argmax(predict[0][i][j]) == 1:
                    # img_synthesis[i][j] = init_img[i][j]
                    img_synthesis[i][j] = np.array([0, 0, 0])
        # plt.imshow(cv2.cvtColor(img_synthesis, cv2.COLOR_BGR2RGB))
        # plt.imshow(img_synthesis)
        # plt.show()

        maskname = item.replace('.tif', '')
        for k in os.listdir(test_PATH):  # Iterate Over Each Image
            if maskname in k and 'mask' in k:
                mask = cv2.imread(os.path.join(test_PATH, k))
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
                mask = cv2.resize(mask, (IMG_HEIGHT, IMG_WIDTH))
                # plt.imshow(mask)
                # plt.show()

        f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
        f.suptitle("LGG Segmentation")
        ax1.set_title('MRI Image')
        ax1.imshow(img[0])
        ax2.set_title('Original Mask')
        ax2.imshow(mask)
        ax3.set_title('Predicted Mask')
        # ax3.imshow(img_synthesis)
        ax3.imshow(img[0], alpha=1)
        ax3.imshow(img_synthesis, alpha=.7)
        plt.show()

        # Accuracy Calculation With Jaccard/IoU On Testing Data
        intersection = np.logical_and(mask, img_synthesis)
        union = np.logical_or(mask, img_synthesis)
        iou_score = np.sum(intersection) / np.sum(union)
        scores_matrix = np.append(scores_matrix, iou_score)


    print(item)
print("Model's Accuracy On Test Dataset: ", np.average(scores_matrix))

