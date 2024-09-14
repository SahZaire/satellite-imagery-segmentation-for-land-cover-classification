# Satellite Imagery Segmentation for Land Cover Classification

## Project Overview

This project focuses on segmenting satellite images to classify different land cover types, including **urban areas**, **forests**, **water bodies**, and **agriculture**. The primary goal is to develop a robust model that can accurately segment and classify satellite imagery into predefined land cover categories. The project makes use of the **DeepGlobe Land Cover Classification Dataset**, employing various deep learning models to achieve high accuracy in segmentation tasks.

The **current model's performance** shows an **error rate of 17%** and an **IoU score of 73%**.

## Dataset

The **DeepGlobe Land Cover Classification Dataset** from [Kaggle](https://www.kaggle.com/competitions/deepglobe-land-cover-classification-challenge/data) is used for this project. The dataset contains high-resolution satellite images with corresponding labeled masks, enabling segmentation of different land cover classes. The dataset is split into training, validation, and test sets for model training and evaluation.

## Models Used

1. **RetinaNet** - Adapted for segmentation, RetinaNet is commonly used for object detection, but here it's utilized to identify and segment land cover types.
2. **VGG16** - A popular convolutional neural network, pre-trained on ImageNet, which is fine-tuned for this segmentation task.
3. **Inception** - A deep CNN architecture that performs exceptionally well in feature extraction, used for classifying and segmenting the satellite imagery.
4. **best_model** - A custom model developed specifically for this project, based on a combination of advanced architectures and tailored to improve segmentation accuracy.

## Future Work

The next step will focus on **fine-tuning the model** to further reduce the error rate and improve segmentation accuracy. The aim is to bring the error rate down to its **lowest possible level**, while optimizing the **IoU score** for better land cover classification.
