# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 15:32:56 2023

@author: tatsatra
"""

import os
import shutil
import config
import numpy as np
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
#import cv2
import albumentations as A


def copyCohortDirs(cohort, caseList):
    
    if not os.path.isdir(config.DIR_OUT + '/Original/' + cohort + '/'):
        os.makedirs(config.DIR_OUT + '/Original/' + cohort + '/images/')
        os.makedirs(config.DIR_OUT + '/Original/' + cohort + '/masks/')
    
    for i in np.arange(len(caseList)):
        shutil.copy(config.MASK_DIR_IN + '/' + caseList[i], config.DIR_OUT + '/Original/' + cohort + '/masks/' + caseList[i])
        shutil.copy(config.IMAGE_DIR_IN + '/' + caseList[i][:-21] + '_CTA_Binary', config.DIR_OUT + '/Original/' + cohort + '/images/' + caseList[i][:-21] + '_CTA_Binary')
        shutil.copy(config.IMAGE_DIR_IN + '/' + caseList[i][:-21] + '_NCCT_Reg_Binary', config.DIR_OUT + '/Original/' + cohort + '/images/' + caseList[i][:-21] + '_NCCT_Binary')
        
    return None


def sliceCohortImages(cohort, caseList):
    
    if not os.path.isdir(config.DIR_OUT + '/Sliced/' + cohort + '/'):
        os.makedirs(config.DIR_OUT + '/Sliced/' + cohort + '/images/')
        os.makedirs(config.DIR_OUT + '/Sliced/' + cohort + '/masks/')
        
    print('\nOverall Progress With Slicing: (' + cohort + ')')
    for i in tqdm(np.arange(len(caseList))):
        image_CTA = np.load(config.DIR_OUT + '/Original/' + cohort + '/images/' + caseList[i][:-21] + '_CTA_Binary')
        image_NCCT = np.load(config.DIR_OUT + '/Original/' + cohort + '/images/' + caseList[i][:-21] + '_NCCT_Binary')
        mask = np.load(config.DIR_OUT + '/Original/' + cohort + '/masks/' + caseList[i][:-21] + '_seg_corrected_Binary')
        
        for j in np.arange(np.shape(image_CTA)[2]):
            np.save(config.DIR_OUT + '/Sliced/' + cohort + '/images/' + caseList[i][:-21] + '_' + str(j) + '_CTA', image_CTA[:, :, j])
            np.save(config.DIR_OUT + '/Sliced/' + cohort + '/images/' + caseList[i][:-21] + '_' + str(j) + '_NCCT', image_NCCT[:, :, j])
            np.save(config.DIR_OUT + '/Sliced/' + cohort + '/masks/' + caseList[i][:-21] + '_' + str(j) + '_seg', mask[:, :, j])
    
    return None


def generateRandomKey(totalAugTypes):
    
    randOne = random.randint(0, totalAugTypes-1)
    key = [0]*randOne + [1]*(totalAugTypes-randOne)
    random.shuffle(key)
    
    return key

            
def visualize(image):
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(image)
    
    
def dataAug(cohort, targetSampleSize, augNos):
    
    if not os.path.isdir(config.DIR_OUT + '/Augmented/' + cohort + '/'):
        os.makedirs(config.DIR_OUT + '/Augmented/' + cohort + '/images/')
        os.makedirs(config.DIR_OUT + '/Augmented/' + cohort + '/masks/')
    
    totalAugTypes = augNos
    count = 0
    
    dataDir = config.DIR_OUT + '/Sliced/' + cohort + '/masks/'
    caseList = os.listdir(dataDir)
    
    while count <= targetSampleSize:
        for i in np.arange(len(caseList)):
            toDoOrNotToDo = generateRandomKey(totalAugTypes)
            
            transform = A.Compose([
                A.HorizontalFlip(p=toDoOrNotToDo[0]),
                A.VerticalFlip(p=toDoOrNotToDo[1]),
                A.RandomBrightnessContrast(p=toDoOrNotToDo[2]),
                A.RandomGamma(p=toDoOrNotToDo[5]),
                A.AdvancedBlur(p=toDoOrNotToDo[6]),
                A.CLAHE(p=toDoOrNotToDo[7])], p=0.80, additional_targets={'image0': 'image'})
            
            cta = np.load(config.DIR_OUT + '/Sliced/' + cohort + '/images/' + caseList[i][:-8] + '_CTA.npy')
            ncct = np.load(config.DIR_OUT + '/Sliced/' + cohort + '/images/' + caseList[i][:-8] + '_NCCT.npy')
            mask = np.load(config.DIR_OUT + '/Sliced/' + cohort + '/masks/' + caseList[i])
            
            cta = cta.astype(np.uint8)
            ncct = ncct.astype(np.uint8)
            
            transformed = transform(image = cta, image0 = ncct, mask = mask)
            
            #print(np.max(mask))
            if np.max(mask)>10:
                visualize(transformed['image'])
                visualize(transformed['image0'])
                visualize(transformed['mask']*255)
            
            np.save(config.DIR_OUT + '/Augmented/' + cohort + '/images/' + caseList[i][:-8] + '_' + str(count) + '_CTA', transformed['image'])
            np.save(config.DIR_OUT + '/Augmented/' + cohort + '/images/' + caseList[i][:-8] + '_' + str(count) + '_NCCT', transformed['image0'])
            np.save(config.DIR_OUT + '/Augmented/' + cohort + '/masks/' + caseList[i][:-8] + '_' + str(count), transformed['mask'])
            
            count += 1
            if count >= targetSampleSize:
                break
    
    return None        