# Optimizing Semantic Segmentation for Enhanced Football Analytics

## Description

This repository contains the code for the project "Optimizing Semantic Segmentation for Enhanced Football Analytics: A Pixel-level Approach," which will be published in the Procedia Computer Science Journal. The project focuses on leveraging advanced deep learning techniques to optimize semantic segmentation in the context of football analytics.

Semantic segmentation is a crucial technique in computer vision that involves classifying each pixel in an image into a specific category. In this project, we apply this technique to analyze football images, aiming to enhance various aspects of football analytics, including player tracking, strategy analysis, and event annotation.

Our approach involves training multiple state-of-the-art deep learning models on a dataset of manually annotated football images. The models are designed to segment key components of a football match, such as players, the ball, the field, and other relevant elements. By optimizing these models, we aim to provide precise and detailed insights into football games, which can be invaluable for coaches, analysts, and broadcasters.

The repository includes the full pipeline for training, testing, and deploying these models, along with the necessary code, dataset structure, and pre-trained weights. The results demonstrate significant improvements in segmentation accuracy, making this approach highly effective for real-world football analytics applications.

> Semantic Segmentation, a pivotal technique in image analysis, is adeptly leveraged in this research to bolster sports analytics, with a concentrated focus on football. A comprehensive pipeline is unveiled for an in-depth analysis of a select portion of the IAUFD 100k dataset, encompassing 2030 manually annotated football images. The methodology entails a thorough evaluation and comparison of diverse semantic segmentation models, supplemented by the integration of advanced pre-processing strategies and optimal training techniques. Such a holistic approach culminates in a marked enhancement in model performance, as evidenced by a significant uptick in the mean Intersection over Union (mIoU). This research offers granular, object-oriented insights that substantially augment player tracking, action recognition, and event detection in football. The conclusive remarks of the study highlight prospective avenues for further research, emphasizing the potential incorporation of Explainable AI and advanced Metamorphic and Security Testing to fortify sports analytics.

Paper: https://doi.org/10.1016/j.procs.2024.04.251

Authors: **Bharathi Malakreddy A, Sadanand Venkataraman, Mohammed Sinan Khan,  Nidhi, Srinivas Padmanabhuni, Santhi Natarajan**

Keywords _Semantic Segmentation; Sports Analytics; Football; Computer Vision; Deep Learning; Convolutional Neural Networks_

## Table of Contents

- [Overview](#overview)
- [Dataset Structure](#dataset-structure)
- [Results](#results)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Testing](#testing)
  - [Predicting](#predicting)
- [Command-line Arguments](#command-line-arguments)
  - [Training Parameters](#training-parameters)
  - [Testing Parameters](#testing-parameters)
  - [Predicting Parameters](#predicting-parameters)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Overview

This project involves training five models on a dataset of 2,000 manually annotated images from various real football events to optimize semantic segmentation for enhanced football analytics. The resulting model weights are stored in the `weights` folder.

## Dataset Structure

The dataset of manually annotated football images is located in the `Football_Seg` folder, organized into training, validation, and testing sections. Ensure your dataset follows this structure:

```plaintext
Football_Seg/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
├── class_dict.csv
└── evaluated_classes
```

### Components

- **train/**: Contains training data with `images` and `labels` subfolders.
- **valid/**: Contains validation data with `images` and `labels` subfolders.
- **test/**: Contains test data with `images` and `labels` subfolders.
- **class_dict.csv**: CSV file mapping class names to label values.
- **evaluated_classes**: Details classes evaluated during training/testing.

## Results

Performance of the pipeline models on the Football_Seg dataset:

| Model       | Backbone          | Epochs | Training mIoU (%) | Validation mIoU (%) |
|-------------|--------------------|--------|--------------------|---------------------|
| DeepLabV3+  | Xception-DeepLab   | 50     | 77.18              | 58.87               |
| DeepLabV3   | ResNet50           | 50     | 76.24              | 51.36               |
| PSPNet      | ResNet50           | 50     | 77.29              | 58.19               |
| UNet        | VGG16              | 50     | 69.36              | 53.77               |
| SegNet      | VGG16              | 40     | 59.47              | 52.20               |

## Requirements

To run this project, you need the following libraries:

- numpy
- pillow
- opencv-python
- tensorflow
- scipy

These can be installed using the `requirements.txt` file.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-repo/football-segmentation.git
    cd football-segmentation
    ```
2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
3. (Optional) Set up a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

## Usage

### Training

To train a model with your dataset, use the following command:

```bash
python train.py --model SegNet --base_model VGG16 --dataset Football_Seg --num_classes 6
```

#### Detailed Options:

| Argument        | Description                                  | Example                   | Options                                        |
|-----------------|----------------------------------------------|---------------------------|------------------------------------------------|
| `--model`         | Specifies the semantic segmentation model    | SegNet                    | SegNet, UNet, PSPNet, DeepLabV3, DeepLabV3Plus |
| `--base_model`    | Defines the backbone architecture            | VGG16                     | VGG16, ResNet50, Xception-DeepLab, MobileNetV2, Densenet121 |
| `--dataset`       | Path to the dataset                          | Football_Seg              | Any valid dataset path                         |
| `--loss`          | Loss function to use during training         | CE                        | CE, Focal_Loss                                 |
| `--num_classes`   | Number of output classes                     | 6                         | Any integer                                    |
| `--random_crop`   | Apply random cropping during augmentation     | True                      | True, False                                    |
| `--crop_height`   | Height for random cropping                   | 864                       | Any integer                                    |
| `--crop_width`    | Width for random cropping                    | 864                       | Any integer                                    |
| `--batch_size`    | Size of the training batches                 | 8                         | Any integer                                    |
| `--valid_batch_size` | Size of the validation batches             | 1                         | Any integer                                    |
| `--num_epochs`    | Number of training epochs                    | 50                        | Any integer                                    |
| `--initial_epoch` | Starting epoch number                        | 0                         | Any integer                                    |
| `--h_flip`        | Apply horizontal flip for data augmentation  | True                      | True, False                                    |
| `--v_flip`        | Apply vertical flip for data augmentation    | True                      | True, False                                    |
| `--brightness`    | Brightness adjustment range for augmentation | [0.7, 1.3]                | Any valid range                                |
| `--rotation`      | Rotation angle for data augmentation         | 0.35                      | Any float                                      |
| `--zoom_range`    | Zoom range for data augmentation             | [0.4, 1.6]                | Any valid range                                |
| `--channel_shift` | Channel shift range for data augmentation    | 0.1                       | Any float                                      |
| `--data_aug_rate` | Proportion of the data to apply augmentation | 0.1                       | Any float                                      |
| `--checkpoint_freq` | Frequency of saving model checkpoints       | 10                        | Any integer                                    |
| `--validation_freq` | Frequency of validation during training     | 2                         | Any integer                                    |
| `--num_valid_images` | Number of images to use for validation      | 18                        | Any integer                                    |
| `--data_shuffle`  | Shuffle data during training                 | True                      | True, False                                    |
| `--random_seed`   | Random seed for reproducibility              | 32                        | Any integer                                    |
| `--weights`       | Path to pre-trained weights                  | weights/weights_path      | Any valid path                                 |


### Testing

To evaluate the model, run:

```bash
python test.py --model SegNet --base_model VGG16 --dataset Football_Seg --num_classes 6 --weights "weights/weights_path"
```

#### Detailed Options:

| Argument      | Description                                    | Example                   | Options                                        |
|---------------|------------------------------------------------|---------------------------|------------------------------------------------|
| `--model`       | Specifies the semantic segmentation model      | SegNet                    | SegNet, UNet, PSPNet, DeepLabV3, DeepLabV3Plus |
| `--base_model`  | Defines the backbone architecture              | VGG16                     | VGG16, ResNet50, Xception-DeepLab, MobileNetV2, Densenet121 |
| `--dataset`     | Path to the dataset                            | Football_Seg              | Any valid dataset path                         |
| `--num_classes` | Number of output classes                       | 6                         | Any integer                                    |
| `--weights`     | Path to pre-trained model weights              | weights/weights_path      | Any valid path                                 |


### Predicting

To generate predictions for a single RGB image, use:

```bash
python predict.py --model DeepLabV3 --base_model ResNet50 --num_classes 6 --weights "weights/DeepLabV3_based_on_ResNet50.h5" --image_path "Prediction_images/image_path"
```

#### Detailed Options:

| Argument      | Description                                    | Example                   | Options                                        |
|---------------|------------------------------------------------|---------------------------|------------------------------------------------|
| `--model`       | Specifies the semantic segmentation model      | DeepLabV3                 | SegNet, UNet, PSPNet, DeepLabV3, DeepLabV3Plus |
| `--base_model`  | Defines the backbone architecture              | ResNet50                  | VGG16, ResNet50, Xception-DeepLab, MobileNetV2, Densenet121 |
| `--num_classes` | Number of output classes                       | 6                         | Any integer                                    |
| `--weights`     | Path to pre-trained model weights              | weights/DeepLabV3_based_on_ResNet50.h5 | Any valid path |
| `--image_path`  | Path to the input image for prediction         | Prediction_images/image_path | Any valid path                                 |

The predicted output will be saved in the `Predictions` folder.

## Command-line Arguments

### Training Parameters

| Argument      | Description                                  | Example                 | Options                                        |
|---------------|----------------------------------------------|-------------------------|------------------------------------------------|
| `--model`       | Choose the semantic segmentation model       | SegNet                  | SegNet, UNet, PSPNet, DeepLabV3, DeepLabV3Plus |
| `--base_model`  | Choose the backbone architecture             | VGG16                   | VGG16, ResNet50, Xception-DeepLab, MobileNetV2, Densenet121 |
| `--dataset`     | Path to the dataset                          | Football_Seg            | Any valid dataset path                         |
| `--loss`        | Choose the loss function for training        | CE                      | CE, Focal_Loss                                 |
| `--num_classes` | Number of output classes to segment          | 6                       | Any integer                                    |
| `--random_crop` | Apply random cropping during augmentation    | True                    | True, False                                    |
| `--crop_height` | Height for random cropping                   | 864                     | Any integer                                    |
| `--crop_width`  | Width for random cropping                    | 864                     | Any integer                                    |
| `--batch_size`  | Training batch size                          | 8                       | Any integer                                    |
| `--valid_batch_size` | Validation batch size                  | 1                       | Any integer                                    |
| `--num_epochs`  | Total number of training epochs              | 50                      | Any integer                                    |
| `--initial_epoch` | Initial epoch to start training from        | 0                       | Any integer                                    |
| `--h_flip`      | Apply horizontal flip for augmentation       | True                    | True, False                                    |
| `--v_flip`      | Apply vertical flip for augmentation         | True                    | True, False                                    |
| `--brightness`  | Brightness range for augmentation            | [0.7, 1.3]              | Any valid range                                |
| `--rotation`    | Rotation angle for augmentation              | 0.35                    | Any float                                      |
| `--zoom_range`  | Zoom range for augmentation                  | [0.4, 1.6]              | Any valid range                                |
| `--channel_shift` | Channel shift range for augmentation       | 0.1                     | Any float                                      |
| `--data_aug_rate` | Rate of data augmentation                  | 0.1                     | Any float                                      |
| `--checkpoint_freq` | Frequency to save model checkpoints       | 10                      | Any integer                                    |
| `--validation_freq` | Frequency to run validation              | 2                       | Any integer                                    |
| `--num_valid_images` | Number of images for validation         | 18                      | Any integer                                    |
| `--data_shuffle` | Shuffle data during training                | True                    | True, False                                    |
| `--random_seed` | Seed for random number generation            | 32                      | Any integer                                    |
| `--weights`     | Path to pre-trained weights                  | weights/weights_path    | Any valid path                                 |

### Testing Parameters

| Argument      | Description                                  | Example                 | Options                                        |
|---------------|----------------------------------------------|-------------------------|------------------------------------------------|
| `--model`       | Choose the semantic segmentation model       | SegNet                  | SegNet, UNet, PSPNet, DeepLabV3, DeepLabV3Plus |
| `--base_model`  | Choose the backbone architecture             | VGG16                   | VGG16, ResNet50, Xception-DeepLab, MobileNetV2, Densenet121 |
| `--dataset`     | Path to the dataset                          | Football_Seg            | Any valid dataset path                         |
| `--num_classes` | Number of output classes to segment          | 6                       | Any integer                                    |
| `--weights`     | Path to pre-trained model weights            | weights/weights_path    | Any valid path                                 |


### Predicting Parameters

| Argument      | Description                                  | Example                                       | Options                                        |
|---------------|----------------------------------------------|-----------------------------------------------|------------------------------------------------|
| `--model`       | Choose the semantic segmentation model       | DeepLabV3                                     | SegNet, UNet, PSPNet, DeepLabV3, DeepLabV3Plus |
| `--base_model`  | Choose the backbone architecture             | ResNet50                                      | VGG16, ResNet50, Xception-DeepLab, MobileNetV2, Densenet121 |
| `--num_classes` | Number of output classes to segment          | 6                                             | Any integer                                    |
| `--weights`     | Path to pre-trained model weights            | weights/DeepLabV3_based_on_ResNet50.h5        | Any valid path                                 |
| `--image_path`  | Path to the input image for prediction       | Prediction_images/image_path                  | Any valid path                                 |

## Acknowledgments

> We would like to thank Dr. Srinivas Padmanabhuni and AiEnsured for their invaluable guidance and support during this research. We also express our gratitude to The Vision Group of Science & Technology, Government of Karnataka, and The Principal, BMS Institute of Technology & Management, for providing the necessary infrastructure and resources.
