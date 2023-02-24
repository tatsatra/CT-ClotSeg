# CT-ClotSeg

This repositary provides the code for training of deep-learning models for the automated segmentation of ischemic stroke clots from combination of CT imaging modalities--CT Angiography (CTA) and non-contrast CT (NCCT).

## Table of contents
* [Pre-processing](#pre-processing)
* [Training setup](#training-setup)
* [Testing setup](#testing-setup)
* [Dependencies](#dependencies)

## Pre-processing
The tool is setup to perform a slice-by-slice segmenation of the clot region from the 3D CTA/NCCT image volumes using 2D segmentation networks. The tool expects pre-processed image volumes, where in smaller ROIs of the 3D CTA/NCCT volumes are extracted around the major arteries of the Circle-of-Willis (COW), followed by orientation, and resolution matching. The paper that references the brain atlas that was used for all the orientation matching can be found here. The sample ROI from the brain atlas can be found in ____. The same process is to be applied to the masks to get the data ready for clot segmentation model training.

## Training setup
After the pre-processing is complete, the process that our tool does, can be described as follows:
* Dividing data in training, validation and testing cohorts
* Extracting 2D slices of CTA, NCCT and masks for 2D model training
* Data augmentation
* Model compilation
* Model training/validation
* Model testing

## Testing setup
coming soon!

## Dependencies
Coming soon!

The directory structure expected for the training is expected as follows:
