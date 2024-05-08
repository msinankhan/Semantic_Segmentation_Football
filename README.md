## Description

This repository contains the code for a project soon to be published in the Procedia Computer Science Journal. The title of the paper is "Optimizing Semantic Segmentation for Enhanced Football Analytics: A Pixel-level Approach."

## Overview

Five models were trained on a dataset of 2,000 manually annotated images from a variety of real football events. The resulting model weights are stored in the `weights` folder.

## Dataset Structure

The dataset of manually annotated football images used for training is located in the `Football_Seg` folder. It is divided into three sections: training, testing, and validation.

To ensure your dataset is organized correctly, follow this structure:

```plaintext
|-- dataset
|  |-- train
|  |  |-- images
|  |  |-- labels
|  |-- valid
|  |  |-- images
|  |  |-- labels
|  |-- test
|  |  |-- images
|  |  |-- labels
|  |-- class_dict.csv
|  |-- evaluated_classes
```

Here's a breakdown of the components:

- **train**: Contains the training data, with separate subfolders for `images` and `labels`.
- **valid**: Contains the validation data, also with `images` and `labels`.
- **test**: Houses the test data, with the same `images` and `labels` structure.
- **class_dict.csv**: A CSV file that maps class names to their corresponding label values.
- **evaluated_classes**: A file or list detailing which classes will be evaluated during training/testing.

Ensure your dataset conforms to this structure to facilitate a smooth training and testing process.



## Results

Here's the performance of the pipeline models on the Football_Seg dataset, formatted into a table:

| Model       | Backbone          | Number of Epochs | Training mIoU (%) | Validation mIoU (%) |
|-------------|-----------------|-----------------|-----------------|--------------------|
| DeepLabV3+  | Xception-DeepLab | 40 + 10          | 77.18            | 58.87               |
| DeepLabV3   | ResNet50          | 50               | 76.24            | 51.36               |
| PSPNet      | ResNet50          | 50               | 77.29            | 58.19               |
| UNet        | VGG16             | 50               | 69.36            | 53.77               |
| SegNet      | VGG16             | 40               | 59.47            | 52.20               |


## Requirements

To get started, install the required packages listed in `requirements.txt` with the following command:

```bash
pip install -r requirements.txt
```

This ensures that you have all the necessary dependencies and libraries to run the project without any compatibility issues. If you're setting up a virtual environment, make sure to activate it before running this command.


## Training

To train a model with your own dataset, use the following command to build it:

```bash
python train.py --model SegNet --base_model VGG16 --dataset Football_Seg --num_classes 6
```

The following is the list of detailed command-line parameters for training a model using `train.py`:

```bash
usage: train.py [-h]
                --model MODEL
                [--base_model BASE_MODEL]
                --dataset DATASET
                [--loss {CE,Focal_Loss}]
                --num_classes NUM_CLASSES
                [--random_crop RANDOM_CROP]
                [--crop_height CROP_HEIGHT]
                [--crop_width CROP_WIDTH]
                [--batch_size BATCH_SIZE]
                [--valid_batch_size VALID_BATCH_SIZE]
                [--num_epochs NUM_EPOCHS]
                [--initial_epoch INITIAL_EPOCH]
                [--h_flip H_FLIP]
                [--v_flip V_FLIP]
                [--brightness BRIGHTNESS [BRIGHTNESS ...]]
                [--rotation ROTATION]
                [--zoom_range ZOOM_RANGE [ZOOM_RANGE ...]]
                [--channel_shift CHANNEL_SHIFT]
                [--data_aug_rate DATA_AUG_RATE]
                [--checkpoint_freq CHECKPOINT_FREQ]
                [--validation_freq VALIDATION_FREQ]
                [--num_valid_images NUM_VALID_IMAGES]
                [--data_shuffle DATA_SHUFFLE]
                [--random_seed RANDOM_SEED]
                [--weights WEIGHTS]
```

Each parameter controls a specific aspect of the training process. Here's a brief explanation of some key parameters:

- **model**: Name of the model to be trained (e.g., `SegNet`, `FCN`, etc.).
- **base_model**: Base architecture used for the model (e.g., `VGG16`, `ResNet50`).
- **dataset**: Path to the dataset used for training.
- **loss**: Loss function used during training (e.g., `CE` for Cross-Entropy, `Focal_Loss`).
- **num_classes**: Number of output classes.
- **random_crop**: Whether to apply random cropping during data augmentation.
- **crop_height/crop_width**: Dimensions of the crop.
- **batch_size**: Size of training batches.
- **valid_batch_size**: Size of validation batches.
- **num_epochs**: Total number of training epochs.
- **initial_epoch**: Starting epoch number.
- **h_flip/v_flip**: Horizontal and vertical flip settings for data augmentation.
- **brightness/rotation/zoom_range**: Parameters for other augmentation techniques.
- **channel_shift**: Adjustments to color channels for augmentation.
- **data_aug_rate**: Proportion of the data to which augmentation is applied.
- **checkpoint_freq/validation_freq**: Frequency of saving model checkpoints and performing validation.
- **num_valid_images**: Number of images to use for validation.
- **data_shuffle**: Whether to shuffle the data during training.
- **random_seed**: Random seed for reproducibility.
- **weights**: Path to pre-trained model weights, if used.

Use this command line to fine-tune your model's training process according to your specific requirements.


Here are the optional command-line arguments for the `train.py` script:

```plaintext
optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         Choose the semantic segmentation method.
  --base_model BASE_MODEL
                        Choose the backbone model.
  --dataset DATASET     The path of the dataset.
  --loss {CE,Focal_Loss}
                        The loss function for training.
  --num_classes NUM_CLASSES
                        The number of classes to be segmented.
  --random_crop RANDOM_CROP
                        Whether to randomly crop the image.
  --crop_height CROP_HEIGHT
                        The height to crop the image.
  --crop_width CROP_WIDTH
                        The width to crop the image.
  --batch_size BATCH_SIZE
                        The training batch size.
  --valid_batch_size VALID_BATCH_SIZE
                        The validation batch size.
  --num_epochs NUM_EPOCHS
                        The number of epochs to train for.
  --initial_epoch INITIAL_EPOCH
                        The initial epoch of training.
  --h_flip H_FLIP       Whether to randomly flip the image horizontally.
  --v_flip V_FLIP       Whether to randomly flip the image vertically.
  --brightness BRIGHTNESS [BRIGHTNESS ...]
                        Randomly change the brightness (list).
  --rotation ROTATION   The angle to randomly rotate the image.
  --zoom_range ZOOM_RANGE [ZOOM_RANGE ...]
                        The range for zooming the image.
  --channel_shift CHANNEL_SHIFT
                        The channel shift range.
  --data_aug_rate DATA_AUG_RATE
                        The rate of data augmentation.
  --checkpoint_freq CHECKPOINT_FREQ
                        How often to save a checkpoint.
  --validation_freq VALIDATION_FREQ
                        How often to perform validation.
  --num_valid_images NUM_VALID_IMAGES
                        The number of images used for validation.
  --data_shuffle DATA_SHUFFLE
                        Whether to shuffle the data.
  --random_seed RANDOM_SEED
                        The random shuffle seed.
  --weights WEIGHTS     The path of weights to be loaded.
```

These arguments allow for customization of various aspects of the training process, such as the choice of model architecture, dataset paths, data augmentation settings, and more. Adjust them according to your specific requirements and preferences.



## Testing

To evaluate the model using your dataset, run the following command:

```bash
python test.py --model SegNet --base_model VGG16 --dataset Football_Seg --num_classes 6 --weights "weights/weights_path"
```

Make sure to replace `"weights/weights_path"` with the actual path to your model's weights file. This command will assess the model's performance based on your specific dataset and settings.

## Predicting

To generate predictions for a single RGB image, use the following command:

```bash
python predict.py --model DeepLabV3 --base_model ResNet50 --num_classes 6 --weights "weights/DeepLabV3_based_on_ResNet50.h5" --image_path "Prediction_images/image_path"
```

Replace `"Prediction_images/image_path"` with the actual path of the image you want to predict on. The predicted output will be saved in the `Predictions` folder.

