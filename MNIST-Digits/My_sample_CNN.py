import pandas as pd
from sklearn.cross_validation import train_test_split
from nolearn.lasagne import NeuralNet, BatchIterator
from lasagne import layers
from lasagne.nonlinearities import softmax, rectify
from lasagne.updates import momentum, nesterov_momentum, sgd, rmsprop
import numpy as np
from matplotlib import pyplot
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

from helpers import nudge_dataset, rotate_dataset, crop_images

def plot_loss(net):
    """
    Plot the training loss and validation loss versus epoch iterations with respect to
    a trained neural network.
    """
    train_loss = np.array([i["train_loss"] for i in net.train_history_])
    valid_loss = np.array([i["valid_loss"] for i in net.train_history_])
    pyplot.plot(train_loss, linewidth = 3, label = "train")
    pyplot.plot(valid_loss, linewidth = 3, label = "valid")
    pyplot.grid()
    pyplot.legend()
    pyplot.xlabel("epoch")
    pyplot.ylabel("loss")
    #pyplot.ylim(1e-3, 1e-2)
    pyplot.yscale("log")
    pyplot.show()


train_df = pd.read_csv('./data/train.csv')
test_df = pd.read_csv('./data/test.csv')

train_label = train_df.values[:, 0]
train_data = train_df.values[:, 1:]
print "train:", train_data.shape, train_label.shape

test_data = test_df.values
print "test:", test_data.shape


##################
# "Crop images"
# train_data= crop_images (train_data)
# test_data= crop_images (test_data)
# print("cropped")
# print "train:", train_data.shape, train_label.shape
##################

train_data = train_data.astype(np.float) #added 32
train_label = train_label.astype(np.int32)

"Augment data: https://github.com/nicholaslocascio/kaggle-mnist-digits/blob/master/predict.py"
# train_data, train_label = nudge_dataset(train_data, train_label, nudge_size = 2)
# print("nudged")
train_data, train_label = rotate_dataset(train_data, train_label, n_rotations=1)
print("Rotated")
print "train:", train_data.shape, train_label.shape
print "test:", test_data.shape

##################


train_data, train_label = shuffle(train_data, train_label, random_state = 13)

train_data = train_data.reshape(-1, 1, 28, 28)
test_data = test_data.reshape(-1, 1, 28, 28)
"""
train_data = train_data.reshape(-1, 1, 26, 26)
test_data = test_data.reshape(-1, 1, 26, 26)
"""


# print("Cropped")
# print "train:", train_data.shape, train_label.shape

# train_data = train_data.reshape(-1, 1, 27, 27)
# test_data = test_data.reshape(-1, 1, 27, 27)
# print "train:", train_data.shape, train_label.shape


CUDA_CONVNET = True
if CUDA_CONVNET:
    from lasagne.layers.cuda_convnet import Conv2DCCLayer, MaxPool2DCCLayer
    Conv2DLayer = Conv2DCCLayer
    MaxPool2DLayer = MaxPool2DCCLayer
else:
    Conv2DLayer = layers.Conv2DLayer
    MaxPool2DLayer = layers.MaxPool2DLayer


cnn = NeuralNet(
    layers = [  # three layers: one hidden layer
        ('input', layers.InputLayer),

        ('conv1', Conv2DLayer),
        ('pool1', MaxPool2DLayer),
        ('dropout1', layers.DropoutLayer),

        ('conv2', Conv2DLayer),
        ('pool2', MaxPool2DLayer),
        # ('dropout2', layers.DropoutLayer),

        ('conv3', Conv2DLayer),
        # ('pool3', MaxPool2DLayer),
        ('dropout3', layers.DropoutLayer),

        ('hidden4', layers.DenseLayer),
        ('dropout4', layers.DropoutLayer),

        ('output', layers.DenseLayer),
        ],
    # layer parameters:
    input_shape = (None, 1, 28, 28),  # 28x28 input pixels per batch
    # input_shape = (None, 1, 27, 27), #cropped!

    conv1_num_filters = 32, conv1_filter_size = (5, 5),
     pool1_ds = (2, 2),
     dropout1_p = 0.3,
    conv2_num_filters = 64, conv2_filter_size=(5, 5),
     pool2_ds=(2, 2),
      dropout2_p = 0.4,
    conv3_num_filters = 64, conv3_filter_size = (4, 4),
    # pool3_ds = (2, 2),
    dropout3_p = 0.4,

    hidden4_num_units = 800,
     dropout4_p = 0.3,
    output_num_units = 10,  # 10 labels

    conv1_nonlinearity = rectify, conv2_nonlinearity = rectify, conv3_nonlinearity = rectify,
    hidden4_nonlinearity = rectify,
    output_nonlinearity = softmax,  # output layer uses softmax function

    # optimization method:
    # update = nesterov_momentum,
    update = rmsprop,
    update_learning_rate = 0.005,
    # update_learning_rate = 0.0005,

    eval_size = 0.1,

    max_epochs = 30,  # we want to train this many epochs #200
    verbose = 1,
    )

cnn.fit(train_data, train_label)

pred = cnn.predict(test_data)

output = pd.DataFrame(data = {"ImageId": range(1, 28001), "Label": pred})
output.to_csv("./fc_2hidden_predict.csv", index = False, quoting = 3)

plot_loss(cnn)