# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 14:52:01 2023

@author: Tatsat Patel
"""

import numpy as np
import os
import config
import random
import pickle

from utils import copyCohortDirs, sliceCohortImages, dataAug

import sys
sys.path.insert(1, './myModels')

from myModels.unet2D import unet2D
from myModels.DataGeneratorUNet2D import DataGeneratorUNet2D

# Splitting data into training, validaiton and testing

if config.TRAIN_TEST_SPLIT == True:
    caseList = os.listdir(config.MASK_DIR_IN)
    random.Random(config.RANDOM_SEED).shuffle(caseList)
    
    training = caseList[:int(np.ceil(len(caseList)*config.TRAIN_PERC))]
    validation = caseList[int(np.ceil(len(caseList)*config.TRAIN_PERC)):int((np.ceil(len(caseList)*config.TRAIN_PERC)+np.floor(len(caseList)*config.VAL_PERC)))]
    test = caseList[int((np.ceil(len(caseList)*config.TRAIN_PERC)+np.floor(len(caseList)*config.VAL_PERC))):]
    
    print('\nTrain/Test split is set to True!')
    print('\nTrain, Validation, Test Split : ' + str(config.TRAIN_PERC) + ':' + str(config.VAL_PERC) + ':' + str(1-(config.TRAIN_PERC+config.VAL_PERC)))
    
    print('\nTraining cohort: ', training)
    copyCohortDirs('Training', training)
    print('\nValidation cohort: ', validation)
    copyCohortDirs('Validation', validation)
    print('\nTesting cohort: ', test)
    copyCohortDirs('Testing', test)
    


# Extracting slices from the 3D images

if config.SLICE_EXTRACT == True:
    
    print('\nSlice Extraction set to True! To be completed for Training and Validation cohorts...')
    
    sliceCohortImages('Training', training)
    sliceCohortImages('Validation', validation)
    
    print('Slicing Complete!')
    


# Data Augmentation

if config.DATA_AUG == True:
    
    dataAug('Training', config.TARGET_SAMPLE_SIZE_AUG_TRAIN, config.NO_OF_TRANSFORMS)
    dataAug('Validation', config.TARGET_SAMPLE_SIZE_AUG_VAL, config.NO_OF_TRANSFORMS)
    


# Model Training

if config.MODEL_TRAINING == True:
    
    model = unet2D(LR = config.LEARNING_RATEà)
    
    training_generator = DataGeneratorUNet2D(input_shape=config.INPUT_SHAPE,
                                        output_shape=config.OUTPUT_SHAPE,
                                        case_dir = config.DIR_OUT + 'Augmented/Training/',
                                        batch_size = config.BATCH_SIZE,
                                        shuffle=True).generate(os.listdir(config.DIR_OUT + '/Augmented/Training/' + 'masks/'))
    
    validation_generator = DataGeneratorUNet2D(input_shape=config.INPUT_SHAPE,
                                        output_shape=config.OUTPUT_SHAPE,
                                        case_dir = config.DIR_OUT + 'Augmented/Validation/',
                                        batch_size = config.BATCH_SIZE,
                                        shuffle=True).generate(os.listdir(config.DIR_OUT + 'Augmented/Validation/' + 'masks/'))
    
    history = model.fit_generator(training_generator,
                            steps_per_epoch = (len(os.listdir(config.DIR_OUT + 'Augmented/Training/' + 'masks/')))//(config.BATCH_SIZE),
                            validation_data = validation_generator,
                            validation_steps = (len(os.listdir(config.DIR_OUT + 'Augmented/Validation/' + 'masks/')))//(config.BATCH_SIZE),
                            epochs = config.EPOCHS,
                            verbose = 1)
    
    if not os.path.isdir(config.MODEL_OUT):
        os.makedirs(config.MODEL_OUT)
    
    model_json = model.to_json()
    with open( config.MODEL_OUT + 'Model.json', 'w' ) as json_file:
        json_file.write(model_json)
    model.save_weights(config.MODEL_OUT + 'Model.h5')
 
    #new_history=history.history
    with open( config.MODEL_OUT + 'Model_history.pkl', 'wb' ) as f:
        pickle.dump(history, f)