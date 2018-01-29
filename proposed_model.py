### Import the necessary libraries ###
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
### Imports picture file into the model
########################################

# TumorA = astrocytoma = 0
# TumorB = glioblastoma_multiforme = 1
# TumorC = oligodendroglioma = 2
# healthy = 3
# unknown = 4

f = open('full_dataset_final.pkl', 'rb')
print("pickle file open")

## Load from the file for X(image data) and Y(tumor type)
allX, allY = pickle.load(f)
print("pickle opened")
f.close()

## image size set to 64x54 for faster computations ##
size_image = 64


###################################
# Define model architecture
###################################

# Input is a 64x64 image with 3 color channel
network = input_data(shape=[None, size_image, size_image, 3])

# 1: Convolution layer with 16 filters, each 5x5
conv_1 = conv_2d(network, nb_filter=16, filter_size=5, activation='relu', name='conv_1')
print("layer 1")

# 2: Max pooling layer
network = max_pool_2d(conv_1, 2)
print("layer 2")

# 3: Convolution layer with 32 filters -> filter size = 3x3
conv_2 = conv_2d(network, nb_filter=32, filter_size=3, activation='relu', name='conv_2')
print("layer 3")

# 4: Convolution layer with 64 filters -> filter size = 3x3
conv_3 = conv_2d(conv_2, nb_filter=64, filter_size=3, activation='relu', name='conv_3')
print("layer 4")

# 5: Convolution layer with 128 filters -> filter size = 2x2
conv_4 = conv_2d(conv_3, nb_filter=128, filter_size=2, activation='relu', name='conv_4')
print("layer 5")

# 6: Max pooling layer
network = max_pool_2d(conv_4, 2)
print("layer 6")

# 7: Fully-connected 512 nodes layer -> activation function = ReLU
network = fully_connected(network, 512, activation='relu')
print("layer 7")

# 8: Dropout layer to combat overfitting
network = dropout(network, 0.5)
print("layer 8")

# 9: Fully-connected layer with 5 outputs for five tumor categories
network = fully_connected(network, 5, activation='softmax')
print("layer 9")


# Regression layer with loss=categorical crossentropy, optimizer=adam, learning rate=0.0001
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.0001)

# Wrap the network in a model object
model = tflearn.DNN(network, tensorboard_verbose = 0)

print("model created done")


###################################################
# Prepare train & test samples and train the model
###################################################

## Using 6-fold cross validation

no_folds = 6 # for 6 fold cross validation

accuracy_array = np.zeros((no_folds), dtype='float64') # accuracies of the test dataset for each split in cross validation
accuracy_array2 = np.zeros((no_folds), dtype='float64') # accuracies for the complete dataset for each split in cross validation

i=0 # counter
split_no = 1 # counter for each split

kf = KFold(n_splits=no_folds, shuffle = True, random_state=42) # create split criteria using KFold in Sklearn.model_selection

#train_splits = []
#test_splits = []

    ###################################
    # Train model for 100 epochs
    ###################################
for train_index, test_index in kf.split(allX):

    # split dataset using kf criteria into train and test dataset
    X, X_test = allX[train_index], allX[test_index]
    Y, Y_test = allY[train_index], allY[test_index]

    # create output labels for whole dataset and test dataset
    Y = to_categorical(Y, 5)
    Y_test = to_categorical(Y_test, 5)

    print("train split: " , split_no)
    split_no += 1 # iterate split no

    # Train the network for 10 epochs per split (shuffles data)  -> total no of training epochs=60
    model.fit(X, Y, n_epoch=10, run_id='cancer_detector', shuffle=True,
        show_metric=True)

    #model.save('model_cancer_detector.tflearn')
    print("Network trained")

    # Calculate accuracies for test dataset and whole dataset in each split run
    score = model.evaluate(X_test, Y_test)
    score2 = model.evaluate(X, Y)

    # populate the accuracy arrays
    accuracy_array[i] = score[0] * 100
    accuracy_array2[i] = score2[0] * 100
    i += 1 # iterate

    print("accuracy checked")
    print("")
    print("accuracy for test dataset: ", accuracy_array) # print accuracy for the test dataset
    print("")
    print("accuracy for whole dataset: ", accuracy_array2) # print accuracy for the whole dataset


print("done training using 6 fold validation")

# Retrieve the maximum accuracy of the accuracy arrays
max_accuracy = accuracy_array[np.argmax(accuracy_array)]
max_accuracy = round(max_accuracy, 3)

max_accuracy2 = accuracy_array2[np.argmax(accuracy_array2)]
max_accuracy2 = round(max_accuracy2, 3)

print("")

###################################################
## Test the model to predict labels ###############
###################################################


#no_iteration = 100
#kf = KFold(n_splits=no_iteration)
#x_splits = kf.split(allX)

# initiate y_label
y_label = 0

# counters
j = 0
k = 0
c = 0
b = 0

# create Y_true and y_pred np.arrays to save the corresponding label (true label and predicted label) -> labels are shown at the beginning of the program
y_pred = np.zeros((len(allY)), dtype='int32')
y_true = np.zeros((len(allY)), dtype='int32')

# split allX and allY into 90 sections
x_list = np.array_split(allX, 90)
y_list = np.array_split(allY, 90)

i = 0

for j in x_list:

    # get the (i)th section from x_list and y_list to x_test and y_test (arrays renew for each j)
    x_test = x_list[i]
    y_test = y_list[i]

    # y_label=predict results for the (i)th section in x_test
    y_label = model.predict(x_test)
    print("running here")

    b = 0 # b is reset in each (j)th iteration
    for k in y_label:
        y_pred[c] = np.argmax(y_label[b]) # get the index of the maximum probability (prediction) for (b)th array in y_label
        y_true[c] = y_test[b] # (b)th element is copied to y_true array
        c += 1
        b += 1
    i += 1


#print("j is", j, "k is ", k, " splits are ", kf.split(allX))

##################################
# Test prints ####################
##################################

print("Prediction finished", c)
print("")
print(len(y_true), " bla bla ", len(y_pred))
print("")

# calculates F1-Score for the whole dataset
print("calculate f1 score")
f1Score = f1_score(y_true, y_pred, average=None)
print(f1Score)

print("")

# calculates Confusion Matrix for the whole dataset
print("calculate confusion matrix")
confusionMatrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])
print("confusion Matrix Created")
print(confusionMatrix)


##################################
## Print the Results #############
##################################

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

