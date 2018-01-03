from __future__ import division, print_function, absolute_import

from skimage import color, io
from scipy.misc import imresize, toimage
import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold

import os
from glob import glob

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.metrics import Accuracy

import decimal
from six.moves import cPickle
import pickle

import h5py

np.set_printoptions(suppress=True)
########################################
### Imports picture files
########################################

# TumorA = astrocytoma = 0
# TumorB = glioblastoma_multiforme = 1
# TumorC = oligodendroglioma = 2
# healthy = 3
# unknown = 4

f = open('full_dataset_final.pkl', 'rb')
print("pickle file open")

allX, allY = pickle.load(f)
print("pickle opened")
f.close()
"""

h5f = h5py.File('dataset.h5', 'r')
allX = h5f['train_X']
allY = h5f['train_Y']
"""

size_image = 64


###################################
# Define model architecture
###################################

# Input is a 64x64 image with 1 color channel (grayscale image)
network = input_data(shape=[None, size_image, size_image, 3])

# 1: Convolution layer with 32 filters, each 3x3
conv_1 = conv_2d(network, nb_filter=16, filter_size=5, activation='relu', name='conv_1')
print("layer 1")

# 2: Max pooling layer
network = max_pool_2d(conv_1, 2)
print("layer 2")

conv_2 = conv_2d(network, nb_filter=16, filter_size=3, activation='relu', name='conv_2')
print("layer 3")

# 4: Convolution layer with 64 filters
conv_3 = conv_2d(conv_2, nb_filter=32, filter_size=3, activation='relu', name='conv_3')
print("layer 4")
network = max_pool_2d(conv_3, 2)

# 5: Convolution layer with 128 filters
conv_4 = conv_2d(network, nb_filter=32, filter_size=3, activation='relu', name='conv_4')
print("layer 5")

# 6: Max pooling layer
network = max_pool_2d(conv_4, 2)
print("layer 6")

# 3: Convolution layer with 64 filters
conv_5 = conv_2d(network, nb_filter=64, filter_size=3, activation='relu', name='conv_5')
print("layer 7")

# 4: Convolution layer with 64 filters
conv_6 = conv_2d(conv_5, nb_filter=64, filter_size=2, activation='relu', name='conv_6')
print("layer 8")

network = max_pool_2d(conv_6, 2)

# 5: Convolution layer with 128 filters
conv_7 = conv_2d(network, nb_filter=128, filter_size=2, activation='relu', name='conv_7')
print("layer 9")

# 6: Max pooling layer
network = max_pool_2d(conv_7, 2)
print("layer 10")

# 7: normalize the network
#network = local_response_normalization(network)
print("layer 7")

# 8: Fully-connected 1024 node layer
network = fully_connected(network, 512, activation='relu')
print("layer 11")

# 9: Dropout layer to combat overfitting
network = dropout(network, 0.5)
print("layer 9")

# 10: Fully-connected 512 node layer
#network = fully_connected(network, 256, activation='relu')
#print("layer 10")

# 11: Dropout layer to combat overfitting
#network = dropout(network, 0.5)
print("layer 11")

# 12: Fully-connected layer with two outputs
network = fully_connected(network, 5, activation='softmax')
print("layer 12")

network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.0001)

# Wrap the network in a model object
model = tflearn.DNN(network, tensorboard_verbose = 0)

print("model created done")


###################################################
# Prepare train & test samples and train the model
###################################################

## Using 3-fold cross validation

no_folds = 6

accuracy_array = np.zeros((no_folds), dtype='float64')
accuracy_array2 = np.zeros((no_folds), dtype='float64')

i=0
split_no = 1

kf = KFold(n_splits=no_folds, shuffle = True, random_state=42)

#train_splits = []
#test_splits = []

    ###################################
    # Train model for 100 epochs
    ###################################
for train_index, test_index in kf.split(allX):

    X, X_test = allX[train_index], allX[test_index]
    Y, Y_test = allY[train_index], allY[test_index]

    #train_splits.append(train_index)
    #test_splits.append(test_index)

    Y = to_categorical(Y, 5)
    Y_test = to_categorical(Y_test, 5)

    print("train split: " , split_no)
    split_no += 1

    model.fit(X, Y, n_epoch=20, run_id='cancer_detector', shuffle=True,
        show_metric=True)

    #model.save('model_cancer_detector.tflearn')

    print("Network trained")

    score = model.evaluate(X_test, Y_test)
    score2 = model.evaluate(X, Y)

    accuracy_array[i] = score[0] * 100
    accuracy_array2[i] = score2[0] * 100
    i += 1

    print("accuracy checked")
    print("")
    print("accuracy for test dataset: ", accuracy_array)
    print("")
    print("accuracy for whole dataset: ", accuracy_array2)


print("done training using 10 fold validation")

#print("length train split array: ", len(train_splits), " length of test split array: ", len(test_splits))



max_accuracy = accuracy_array[np.argmax(accuracy_array)]
max_accuracy = round(max_accuracy, 3)

max_accuracy2 = accuracy_array2[np.argmax(accuracy_array2)]
max_accuracy2 = round(max_accuracy2, 3)

print("")

#no_iteration = 100
#kf = KFold(n_splits=no_iteration)
#x_splits = kf.split(allX)
y_label = 0
#label2 = 0
j = 0
k = 0
c = 0
b = 0
y_pred = np.zeros((len(allY)), dtype='int32')
y_true = np.zeros((len(allY)), dtype='int32')

x_list = np.array_split(allX, 90)
y_list = np.array_split(allY, 90)

i = 0

for j in x_list:
	x_test = x_list[i]
	y_test = y_list[i]
	
	y_label = model.predict(x_test)
	print("running here")
	
	b = 0
	for k in y_label:
		y_pred[c] = np.argmax(y_label[b])
		y_true[c] = y_test[b]
		c += 1
		b += 1
	i += 1
	
"""
for j, k in x_splits:
	X, X_test = allX[j], allX[k]
	Y, Y_test = allY[j], allY[k]
	img = imresize(allX[0], (64,64,3))
	#print("len test", len(Y_test))
	
	y_label = model.predict(X_test)
	b = 0
	#print("predict running ", len(y_label) )
	#print("")
	for i in y_label:
		y_pred[c] = np.argmax(y_label[b])
		y_true[c] = Y_test[b]
		c += 1
		b += 1
"""
#print("j is", j, "k is ", k, " splits are ", kf.split(allX))	
print("Prediction finished", c)
print("")
print(len(y_true), " bla bla ", len(y_pred))
print("")

print("calculate f1 score")
f1Score = f1_score(y_true, y_pred, average=None)
print(f1Score)

print("")
print("calculate confusion matrix")

confusionMatrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])

print("confusion Matrix Created")
print(confusionMatrix)

print("")
print("")
print ("-----------------------------------------------------------------------------")
print ( "    Cancer Tumor detector using Convolutional Neural Networks - 3-Fold cross validation") 
print ("Author - Narmada Balasooriya")
print ("-----------------------------------------------------------------------------")
print("")
print("accuracy for the test dataset")
print(accuracy_array)
print("")
print("accuracy for the whole dataset")
print(accuracy_array2)
print("")
print("Maximum accuracy for test dataset: ", max_accuracy, '%')
print("")
print("Maximum accuracy for whole dataset: ", max_accuracy2, '%')
print("")
print("F1 score for the whole dataset")
print(f1Score)
print("")
print("confusion Matrix")
print(confusionMatrix)
print("")
print ("-----------------------------------------------------------------------------")

