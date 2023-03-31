# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 14:52:41 2023

@author: tatsatra
"""

import os
from datetime import datetime

#########################################
########  FOR SLICING OF IMAGES  ########
#########################################

DIR_PATH = os.getcwd()

TRAIN_TEST_SPLIT = False
RANDOM_SEED = 1
TRAIN_PERC = 0.6
VAL_PERC = 0.2

IMAGE_DIR_IN = f"{DIR_PATH}/DL_vol_files/binary/"
MASK_DIR_IN = f"{DIR_PATH}/DL_seg_files/binary/"

DIR_OUT = f"{DIR_PATH}/Processed/"

SLICE_EXTRACT = False

TARGET_SAMPLE_SIZE_AUG_TRAIN = 10000
TARGET_SAMPLE_SIZE_AUG_VAL = 2000

DATA_AUG = False
NO_OF_TRANSFORMS = 8


MODEL_TRAINING = True
INPUT_SHAPE = (256, 256, 2)
OUTPUT_SHAPE = (256, 256, 1)
BATCH_SIZE = 20
EPOCHS = 100
LEARNING_RATE = 0.001

dAndT = datetime.now().strftime("%Y%m%d-%H%M%S")

MODEL_OUT = f"{DIR_PATH}/Trainings/" + dAndT + "/Best_Model/"
CKPT_OUT = f"{DIR_PATH}/Trainings/" + dAndT + "/Checkpoint_Files/"
FILE_CP_OUT = f"{DIR_PATH}/Trainings/" + dAndT + "/Relevant_Files/"
