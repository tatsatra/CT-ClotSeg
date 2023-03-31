# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 14:52:41 2023

@author: tatsatra
"""

import os


#########################################
########  FOR SLICING OF IMAGES  ########
#########################################

DIR_PATH = "D:/Munjal/Clot_DL/"

TRAIN_TEST_SPLIT = True
RANDOM_SEED = 1
TRAIN_PERC = 0.6
VAL_PERC = 0.2

IMAGE_DIR_IN = f"{DIR_PATH}/DL_vol_files/binary/"
MASK_DIR_IN = f"{DIR_PATH}/DL_seg_files/binary/"

DIR_OUT = f"{DIR_PATH}/Processed/"

SLICE_EXTRACT = True

TARGET_SAMPLE_SIZE_AUG_TRAIN = 100
TARGET_SAMPLE_SIZE_AUG_VAL = 20

DATA_AUG = True
NO_OF_TRANSFORMS = 8


MODEL_TRAINING = True
INPUT_SHAPE = (256, 256, 2)
OUTPUT_SHAPE = (256, 256, 1)
BATCH_SIZE = 20
EPOCHS = 5
LEARNING_RATE = 0.0005

MODEL_OUT = f"{DIR_PATH}/TrainedModels/"