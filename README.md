# Brain-Tumor-Segmentation-TCGA
Deep Learning for Brain Tumor Detection and Segmentation from MRI Scans 

You can download the Dataset from: https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation;

Extract all the photos from DATASET TCGA folder mixed imgs and masks.
RUN "data_preparation.py"

IT WILL separate photos into 2 folders: Images and Masks
It also will resize all the photos to 128x128 and convert them to RGB

Once you have the images at the IMAGES_PATH and their coresponding targets at MASKS_PATH
RUN "brain_tumor.py"

A deep neural network will be constructed. I used Convolutional Neural Networks on a U-Net architecture.

At the end of the code it will generate the trained model.h5

Then create a testing folder and 
RUN "brain_tumor_predict.py" to demonstrate the predictions of the model. 
