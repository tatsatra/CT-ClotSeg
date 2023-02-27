# CT-ClotSeg

This repositary provides the code for training of deep-learning models for the automated segmentation of ischemic stroke clots from combination of CT imaging modalities--CT Angiography (CTA) and non-contrast CT (NCCT).

## Table of contents
* [Pre-processing](#pre-processing)
* [Training setup](#training-setup)
* [Testing setup](#testing-setup)
* [Dependencies](#dependencies)

## Pre-processing
The tool is setup to perform a slice-by-slice segmenation of the clot region from the 3D CTA/NCCT image volumes using 2D segmentation networks. In our methodology, we pre-processed image volumes, where in smaller ROIs of the 3D CTA/NCCT volumes are extracted around the major arteries of the Circle-of-Willis (COW), followed by orientation, and resolution matching. The paper that references the brain atlas that was used for all the orientation matching can be found here. The sample ROI from the brain atlas can be found in ____. The same process is to be applied to the masks to get the data ready for clot segmentation model training. The final resolution used for us in our training/valdiaiton/testing was __ mm isotropic, resulting in consistent ___ slices of size ___x___. The pre-processing for our cohort was done using Slicer.

Nonetheless, the tool can take in any sized 3D volumes of CTA/NCCT as inputs as long as the size of each slice across all images is less than 256x256 pixels. The training center-pads all image patches to the size of 256x256.

## Training setup
After the pre-processing is complete, the process that our tool does, can be described as follows:

### Dividing data in training, validation and testing cohorts
Splits the dataset randomly into training, valdiaiton and testing cohorts. The percentages can be set by users from the ```config.py``` file: ```TRAIN_PERC = 0.6``` and ```VAL_PERC = 0.2```.
The testing percentage is computed as the complementary of the training and validation percentages. Also, if the splitting of data is not required again, as it was already done previously, it can be turned on/off by changing the following in the ```config.py``` file: ```TRAIN_TEST_SPLIT == True```. The outputs of the data splitting is stored with following directory structure,


### Extracting 2D slices of CTA, NCCT and masks for 2D model training
Extracts 2D slices from the 3D volumes of the CTA/NCCT images and the corresponding masks. This is done separately for the training and validation cohorts. Slicing of the testing cohort is done on the fly in the testing module of the tool. If the slicing of the images is not required again, as it was already done previously, it can be turned on/off by changing the following in the ```config.py``` file: ```SLICE_EXTRACT == True```. 

📦myModels
 ┣ 📜DataGeneratorUNet2D.py
 ┗ 📜unet2D.py

### Training/Validation 2D slice augmentation
Following the slicing of the images and masks, the 2D slices are augmented using the ```albumentations``` package available [here](https://albumentations.ai/). The following augmentations were allowed for the aimges and the masks (where appropriate):
* HorizontalFlip,
* VerticalFlip,
* RandomBrightnessContrast,
* RandomGamma,
* AdvancedBlur,
* CLAHE.
The overall probability of allowing an image to be augmented was set to ```p=0.80```. The number of samples (target) for the training and validation augmentations can be set in the ```config.py``` file as well under the variable names ```TARGET_SAMPLE_SIZE_AUG_TRAIN``` and ```TARGET_SAMPLE_SIZE_AUG_VAL``` respectively. If the augmentation of the images is not required again, as it was already done previously, it can be turned on/off by changing the following in the ```config.py``` file: ```SLICE_EXTRACT == True```.

* Data augmentation
* Model compilation
* Model training/validation
* Model testing

## Testing setup
coming soon!

## Dependencies
Coming soon!

The directory structure expected for the training is expected as follows:
