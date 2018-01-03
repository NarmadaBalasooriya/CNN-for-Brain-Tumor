from __future__ import division, print_function, absolute_import

from skimage import color, io
from scipy.misc import imresize
import numpy as np
import os
from glob import glob
import cv2

from six.moves import cPickle
import pickle


np.set_printoptions(suppress=True)
########################################
### Imports picture files
########################################

# TumorA = astrocytoma = 0
# TumorB = glioblastoma_multiforme = 1
# TumorC = oligodendroglioma = 2
# healthy = 3
# unknown = 4

files_path_tumorA = 'I:/HIGHER_STUDIES/RESEARCH/DeepLearningPython/Dataset/dataset_for_the_research/brain_astrocytoma'
files_path_tumorB = 'I:/HIGHER_STUDIES/RESEARCH/DeepLearningPython/Dataset/dataset_for_the_research/brain_glioblastoma_multiforme'
files_path_tumorC = 'I:/HIGHER_STUDIES/RESEARCH/DeepLearningPython/Dataset/dataset_for_the_research/brain_oligodendroglioma'
files_path_healthy = 'I:/HIGHER_STUDIES/RESEARCH/DeepLearningPython/Dataset/dataset_for_the_research/healthy_brain'
files_path_tumor_unknown = 'I:/HIGHER_STUDIES/RESEARCH/DeepLearningPython/Dataset/dataset_for_the_research/brain_unknown'

tumorA_path = os.path.join(files_path_tumorA, 'image*.jpg')
tumorB_path = os.path.join(files_path_tumorB, 'image*.jpg')
tumorC_path = os.path.join(files_path_tumorC, 'image*.jpg')
no_tumor_path = os.path.join(files_path_healthy, 'image*.jpg')
tumor_unknown_path = os.path.join(files_path_tumor_unknown, 'image*.jpg')

print("tumor A path")

tumorA = sorted(glob(tumorA_path))
#tumorA = glob(tumorA_path)
tumorB = sorted(glob(tumorB_path))
tumorC = sorted(glob(tumorC_path))
no_tumor = sorted(glob(no_tumor_path))
tumor_unknown = sorted(glob(tumor_unknown_path))

n_files = len(tumorA) + len(tumorB) + len(tumorC) + len(no_tumor) + len(tumor_unknown)
print("######print here")
print(n_files)
print("##########")

size_image = 64

allX = np.zeros((n_files, size_image, size_image, 3), dtype='uint8')
ally = np.zeros((n_files), dtype='int32')
count = 0
y_count = 0
for f in tumorA:
    try:
        img = io.imread(f)
        new_img = imresize(img, (size_image, size_image, 3))
        allX[count] = np.array(new_img)
        ally[y_count] = 0
        count += 1
        y_count += 1
    except:
        continue

print("tumorA done")
for f in tumorB:
    try:
        img = io.imread(f)
        new_img = imresize(img, (size_image, size_image, 3))
        allX[count] = np.array(new_img)
        ally[y_count] = 1
        count += 1
        y_count += 1
    except:
        continue
print("tumorB done")
for f in tumorC:
    try:
        img = io.imread(f)
        new_img = imresize(img, (size_image, size_image, 3))
        allX[count] = np.array(new_img)
        ally[y_count] = 2
        count += 1
        y_count += 1
    except:
        continue
print("tumorC done")
for f in no_tumor:
    try:
        img = io.imread(f)
        new_img = imresize(img, (size_image, size_image, 3))
        allX[count] = np.array(new_img)
        ally[y_count] = 3
        count += 1
        y_count += 1
    except:
        continue
print("no tumor done")
for f in tumor_unknown:
    try:
        img = io.imread(f)
        new_img = imresize(img, (size_image, size_image, 3))
        allX[count] = np.array(new_img)
        ally[y_count] = 4
        count += 1
        y_count += 1
    except:
        continue
print("unknown done")
print("images are arrayed")

print("data are split")

f = open('full_dataset_final.pkl', 'wb')

print("pickle file open")
cPickle.dump((allX, ally), f, protocol=cPickle.HIGHEST_PROTOCOL)
print("pickle dumped")
f.close()

print("finished")
