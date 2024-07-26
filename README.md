# Animal Image Classification Project

*Contributors: Pragya Kaushik, Savannah Fung*

## Introduction

This project aims to automate the classification of 151 different animal species using machine learning techniques. The initial baseline model showed significant issues with overfitting and poor generalization, leading to a series of modifications and improvements to enhance its performance. Additionally, we explored transfer learning and quantization-aware training (QAT) to achieve better accuracy and efficiency.

## Dataset

The dataset consists of 6270 RGB images with a resolution of 224x224 pixels, organized into 151 folders, each representing a different animal class. Despite the uniform resolution, the images exhibit variability in clarity and resolution, impacting model performance. The dataset shows a moderate imbalance, with an average of 41.52 samples per class and a standard deviation of 10.07.

## Baseline Model

The baseline Convolutional Neural Network (CNN) model includes:
- Four convolutional layers for feature extraction
- ReLU activation functions for non-linearity
- Max-pooling layers for spatial dimension reduction
- A fully connected layer for classification

### Performance
- Validation Accuracy: 36.61%
- Validation Loss: 5.0514
- Trainable Parameters: 857,239
- FLOPs: 0.69G

## Modifications and Improvements

### Key Adjustments - Ablation Study
1. **Increasing Batch Size**: From 16 to 64
   - Validation Accuracy: 41.50%
   - Validation Loss: 4.5523
2. **Adding Batch Normalization**: After each convolutional layer
   - Validation Accuracy: 58.03%
   - Validation Loss: 2.9831
3. **Decreasing Learning Rate**: To 0.0005
   - Validation Accuracy: 60.14%
   - Validation Loss: 2.7064

Other explored modifications included different learning rate schedulers, optimizers, residual layers, and data augmentation, but these did not yield better results.

## Transfer Learning

We employed transfer learning using MobileNetV3 Large, chosen for its efficiency and moderate accuracy. By fine-tuning the model on our dataset, we achieved a significant performance boost:
- **Accuracy**: 96.41%
- **Learning Rate Scheduler**: Used to optimize training
- **Increased Input Size**: From 112 to 224 to leverage full resolution

## Quantization-Aware Training (QAT)

To reduce computational cost, QAT was applied to MobileNetV3 Large, achieving:
- **Accuracy**: 92.81%
- **Computational Cost**: 0.002964856 GFLOPs

### Efficiency Comparison
- Non-quantized Model Efficiency: 224.95
- Quantized Model Efficiency: 31,303.05

## Conclusion

Our optimized model, leveraging MobileNetV3 Large with QAT, provides an effective and efficient solution for animal image classification. Future improvements include expanding the dataset and employing data augmentation techniques to further enhance performance.
