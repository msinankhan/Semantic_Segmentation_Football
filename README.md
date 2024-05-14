# Optimizing Semantic Segmentation for Enhanced Football Analytics

## Description

This repository contains the code for the project "Optimizing Semantic Segmentation for Enhanced Football Analytics: A Pixel-level Approach," soon to be published in the Procedia Computer Science Journal.

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

- `--model`: Specifies the semantic segmentation model to use.
  - Options: `SegNet`, `UNet`, `PSPNet`, `DeepLabV3`, `DeepLabV3Plus`
- `--base_model`: Defines the backbone architecture for the model.
  - Options: `VGG16`, `ResNet50`, `Xception-DeepLab`, `MobileNetV2`, `Densenet121`
- `--dataset`: Path to the dataset.
  - Example: `Football_Seg`
- `--loss`: Loss function to use during training.
  - Options: `CE` (Cross-Entropy), `Focal_Loss`
- `--num_classes`: Number of output classes.
  - Example: `6`
- `--random_crop`: Whether to apply random cropping.
  - Options: `True`, `False`
- `--crop_height`: Height of the crop (if random cropping is applied).
- `--crop_width`: Width of the crop (if random cropping is applied).
- `--batch_size`: Size of the training batches.
  - Example: `8`
- `--valid_batch_size`: Size of the validation batches.
  - Example: `1`
- `--num_epochs`: Number of training epochs.
  - Example: `50`
- `--initial_epoch`: Starting epoch number.
  - Example: `0`
- `--h_flip`: Whether to apply horizontal flip for data augmentation.
  - Options: `True`, `False`
- `--v_flip`: Whether to apply vertical flip for data augmentation.
  - Options: `True`, `False`
- `--brightness`: Brightness adjustment range for data augmentation.
  - Example: `[0.7, 1.3]`
- `--rotation`: Rotation angle for data augmentation.
  - Example: `0.35`
- `--zoom_range`: Zoom range for data augmentation.
  - Example: `[0.4, 1.6]`
- `--channel_shift`: Channel shift range for data augmentation.
  - Example: `0.1`
- `--data_aug_rate`: Proportion of the data to apply augmentation.
  - Example: `0.1`
- `--checkpoint_freq`: Frequency of saving model checkpoints.
  - Example: `10`
- `--validation_freq`: Frequency of validation during training.
  - Example: `2`
- `--num_valid_images`: Number of images to use for validation.
  - Example: `18`
- `--data_shuffle`: Whether to shuffle data during training.
  - Options: `True`, `False`
- `--random_seed`: Random seed for reproducibility.
  - Example: `32`
- `--weights`: Path to pre-trained weights (if applicable).

### Testing

To evaluate the model, run:

```bash
python test.py --model SegNet --base_model VGG16 --dataset Football_Seg --num_classes 6 --weights "weights/weights_path"
```

#### Detailed Options:

- `--model`: Specifies the semantic segmentation model to use.
  - Options: `SegNet`, `UNet`, `PSPNet`, `DeepLabV3`, `DeepLabV3Plus`
- `--base_model`: Defines the backbone architecture for the model.
  - Options: `VGG16`, `ResNet50`, `Xception-DeepLab`, `MobileNetV2`, `Densenet121`
- `--dataset`: Path to the dataset.
  - Example: `Football_Seg`
- `--num_classes`: Number of output classes.
  - Example: `6`
- `--weights`: Path to the pre-trained model weights.
  - Example: `weights/weights_path`

### Predicting

To generate predictions for a single RGB image, use:

```bash
python predict.py --model DeepLabV3 --base_model ResNet50 --num_classes 6 --weights "weights/DeepLabV3_based_on_ResNet50.h5" --image_path "Prediction_images/image_path"
```

#### Detailed Options:

- `--model`: Specifies the semantic segmentation model to use.
  - Options: `SegNet`, `UNet`, `PSPNet`, `DeepLabV3`, `DeepLabV3Plus`
- `--base_model`: Defines the backbone architecture for the model.
  - Options: `VGG16`, `ResNet50`, `Xception-DeepLab`, `MobileNetV2`, `Densenet121`
- `--num_classes`: Number of output classes.
  - Example: `6`
- `--weights`: Path to the pre-trained model weights.
  - Example: `weights/DeepLabV3_based_on_ResNet50.h5`
- `--image_path`: Path to the input image for prediction.
  - Example: `Prediction_images/image_path`

The predicted output will be saved in the `Predictions` folder.

## Command-line Arguments

### Training Parameters

- `--model`: Choose the semantic segmentation model.
  - Options: `SegNet`, `UNet`, `PSPNet`, `DeepLabV3`, `DeepLabV3Plus`
- `--base_model`: Choose the backbone architecture.
  - Options: `VGG16`, `ResNet50`, `Xception-DeepLab`, `MobileNetV2`, `Densenet121`
- `--dataset`: Path to the dataset.
  - Example: `Football_Seg`
- `--loss`: Choose the loss function for training.
  - Options: `CE`, `Focal_Loss`
- `--num_classes`: Number of output classes to segment.
  - Example: `6`
- `--random_crop`: Apply random cropping during augmentation.
  - Options: `True`, `False`
- `--crop_height`: Height for random cropping.
  - Example: `864`
- `--crop_width`: Width for random cropping.
  - Example: `864`
- `--batch_size`: Training batch size.
  - Example: `8`
- `--valid_batch_size`: Validation batch size.
  - Example: `1`
- `--num_epochs`: Total number of training epochs.
  - Example: `50`
- `--initial_epoch`: Initial epoch to start training from.
  - Example: `0`
- `--h_flip`: Apply horizontal flip for augmentation.
  - Options: `True`, `False`
- `--v_flip`: Apply vertical flip for augmentation.
  - Options: `True`, `False`
- `--brightness`: Brightness range for augmentation.
  - Example: `[0.7, 1.3]`
- `--rotation`: Rotation angle for augmentation.
  - Example: `0.35`
- `--zoom_range`: Zoom range for augmentation.
  - Example: `[0.4, 1.6]`
- `--channel_shift`: Channel shift range for augmentation.
  - Example: `0.1`
- `--data_aug_rate`: Rate of data augmentation.
  - Example: `0.1`
- `--checkpoint_freq`: Frequency to save model checkpoints.
  - Example: `10`
- `--validation_freq`: Frequency to run validation.
  - Example: `2`
- `--num_valid_images`: Number of images to use for validation.
  - Example: `18`
- `--data_shuffle`: Shuffle data during training.
  - Options: `True`, `False`
- `--random_seed`: Seed for random number generation.
  - Example: `32`
- `--weights`: Path to pre-trained weights (optional).

### Testing Parameters

- `--model`: Choose the semantic segmentation model.
  - Options: `SegNet`, `UNet`, `PSPNet`, `DeepLabV3`, `DeepLabV3Plus`
- `--base_model`: Choose the backbone architecture.
  - Options: `VGG16`, `ResNet50`, `Xception-DeepLab`, `MobileNetV2`, `Densenet121`
- `--dataset`: Path to the dataset.
  - Example: `Football_Seg`
- `--num_classes`: Number of output classes to segment.
  - Example: `6`
- `--weights`: Path to the pre-trained model weights.
  - Example: `weights/weights_path`

### Predicting Parameters

- `--model`: Choose the semantic segmentation model.
  - Options: `SegNet`, `UNet`, `PSPNet`, `DeepLabV3`, `DeepLabV3Plus`
- `--base_model`: Choose the backbone architecture.
  - Options: `VGG16`, `ResNet50`, `Xception-DeepLab`, `MobileNetV2`, `Densenet121`
- `--num_classes`: Number of output classes to segment.
  - Example: `6`
- `--weights`: Path to the pre-trained model weights.
  - Example: `weights/DeepLabV3_based_on_ResNet50.h5`
- `--image_path`: Path to the input image for prediction.
  - Example: `Prediction_images/image_path`

## Acknowledgments

We would like to thank Dr. Srinivas Padmanabhuni and AiEnsured for their invaluable guidance and support during this research. We also express our gratitude to The Vision Group of Science & Technology, Government of Karnataka, and The Principal, BMS Institute of Technology & Management, for providing the necessary infrastructure and resources.

## License

This project is licensed under the CC BY-NC-ND license. For more details, see [LICENSE](LICENSE).
