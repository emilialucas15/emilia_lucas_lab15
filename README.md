# LAB 15 — Brain Tumor MRI Classification on Android

Android app that classifies brain MRI images into four classes using a TensorFlow Lite model trained in Google Colab.

**Author:** Emilia Lucas  
**Course:** Artificial Intelligence — Universidad Yachay Tech

## Classes
`glioma` · `meningioma` · `notumor` · `pituitary`

## Model
- **Architecture:** MobileNetV2 pre-trained on ImageNet (frozen) + new classification layer (transfer learning)
- **Dataset:** [Brain Tumor MRI Dataset (Kaggle)](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset), 5,600 training images and 1,600 test images
- **Input:** 160×160 RGB image, pixel values 0–255 (normalization is inside the model)
- **Output:** 4 probabilities, in the same order as `labels.txt`
- **Size:** 8.9 MB (`.tflite`)

## Results
| Metric | Value |
|---|---|
| Test accuracy (1,600 unseen images) | **87.1 %** |
| No tumor | 99.8 % |
| Pituitary | 98.0 % |
| Meningioma | 80.5 % |
| Glioma | 70.3 % |

The most common error is glioma being predicted as meningioma.

## Project structure
