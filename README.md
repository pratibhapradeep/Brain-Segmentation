# Brain-Segmentation
Deep-Learning based system for brain tumor segmentation from MRI scans 

Our goal: develop an automated deep learning-based system for accurate segmentation of brain tumors in MRI scans (3D volumetric data)
We aim to evaluate a U-Net-based architecture on brain tumor segmentation using the BraTS database, which contains brain tumor MRI scans alongside segmentation masks
U-net is a popular fully convolutional neural network architecture primarily used for image segmentation; it can effectively learn from limited training data and has been widely adopted in medical image analysis and other segmentation tasks.


# Data
BraTS2020 dataset (369 full 3D brain MRI scans, 120 slices (images) per scan)
Modalities
- T1
- T1-CE
- T2
- FLAIR
Ground Truth: Pixel-wise tumor segmentation masks
Data Split: 80% train / 10% validation / 10% test 
16 train/2 val/2 test
1440/180/180 2D slices

# Algorithms and Software Used

Algorithms:

2D U-Net–style Volumetric CNN
Combined Dice + Focal loss for better segmentation + class imbalance handling
Residual connections to preserve spatial info across scales
Software:

Python (v3.8)
PyTorch (deep learning framework)
Supporting libraries: NumPy, OpenCV, Matplotlib, scikit-learn
Trained on: 
Google Colab with NVIDIA T4 GPU

# Preprocessing

Images registered to SRI24 anatomical atlas (Rigid transformation)
Cropped to middle 90 slices (to reduce data noise and focus on most tumor containing region) and centered 200x200 pixels 
Intensity normalization (min-max scaling, for better model convergence)

# 2D Volumetric CNN Architecture

2D connected U-Net-inspired CNN for slice-wise segmentation
Inputs: 4-channels composite (T1, T1CE, T2, FLAIR)
~12 million parameters
Residual connections
Improve gradient flow
Global feature access
Retain spatial context

# Training Procedure

Train/Val/Test Split: 80% train, 10% validation, 10% test
Input: Middle 90 slices per scan (cropped to 192×192)
Optimizer: Adam (lr = 0.0001)
Loss: Dice + Focal Loss
Batch Size: 4
Epochs: 10
Evaluation Metrics: Dice, IoU, Precision, Recall, F1

# Tumor Segmentation Performance

After optimizing our model parameters, we experimented with different optimizers and loss functions to train our 2D volumetric CNN
The best performance came from using the Adam optimizer with a combined Dice and Focal loss, achieving a Dice score of 0.932
For reference, a Dice score above 0.9 is considered strong segmentation in medical imaging

# Findings

The best performance came from using the Adam optimizer with a combined Dice and Focal loss, achieving a Dice score of 0.932
Confusion matrix, precision, and recall scores indicate a high sensitivity and specificity
Our performance matches current state-of-the-art techniques while using less parameters, which fulfilled our goal of finding a more efficient model

# Limitations

By using 2D images slice-by-slice, this approach compromises on global relationships between slices, which is relatively important with 3D volumetric data
May limit the continued improvement of using a 2D slice-by-slice process
We were not able to use the full dataset due to a lack of sufficient computational power (sessions kept crashing) — ideally, we would love to train our model on the full BraTS dataset in order to improve our model

# Future Steps

We would like to train our model on the full BraTS dataset available to refine and continue improvement of our model
Rather than training our architecture from scratch, as we did in this project, we would like to fine-tune a similar architecture trained on organ segmentation using the BraTS dataset and determine how that compares to our current findings

We were also interested in assessing the use of 3D-Vision Transformer models on this dataset, which are capable of capturing 3D global relationships (previously a major part of our project)
However, the combination of data-hungry Vision Transformers and large, 3D volumetric MRIs (192x192x90 after preprocessing) images were too heavy for our processing power to handle.
Future goal: acquire more power and assess the performance of Vision Transformers against our CNN-based approach
