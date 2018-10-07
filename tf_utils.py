import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import pandas as pd
import csv
import sklearn.model_selection
import heli_inference
# def load_dataset():
#     TRAIN_FILE = pd.read_csv("train.csv")
#     TEST_FILE = pd.read_csv("test.csv")
#     train_dataset = h5py.File('datasets/train_signs.h5', "r")
#     train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
#     train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels
#
#     test_dataset = h5py.File('datasets/test_signs.h5', "r")
#     test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
#     test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels
#
#     classes = np.array(test_dataset["list_classes"][:]) # the list of classes
#
#     train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
#     test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
#
#     return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def read_data(test_data='testdata.csv', n=2, label=1):
    '''
    加载数据的功能
    n:特征数据起始位
    label：是否是监督样本数据
    '''
    csv_reader = csv.reader(open(test_data))
    data_list = []
    for one_line in csv_reader:
        data_list.append(one_line)
    X_list = []
    y_list = []
    for one_line in data_list[1:]:
        if label == 1:
            y_list.append(int(one_line[-1]))  # 标志位
            one_list = [float(o) for o in one_line[n:-1]]
            X_list.append(one_list)
        else:
            one_list = [float(o) for o in one_line[n:]]
            X_list.append(one_list)
    return X_list, y_list


def split_data(X_list, y_list, ratio=0.30):
    '''
    按照指定的比例，划分样本数据集
    ratio: 测试数据的比率
    '''
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_list, y_list, test_size=ratio, random_state=50)
    print('--------------------------------split_data shape-----------------------------------')

    print(len(X_train), len(y_train))
    print(len(X_test), len(y_test))
    print(type(X_train))
    print(X_train[1:2],'\n')
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    y_train=convert_to_one_hot(y_train,heli_inference.OUTPUT_NODE)
    y_test = convert_to_one_hot(y_test, heli_inference.OUTPUT_NODE)
    return X_train, X_test, y_train, y_test



def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    m = X.shape[1]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y


def predict(X, parameters):

    W1 = tf.convert_to_tensor(parameters["W1"])
    b1 = tf.convert_to_tensor(parameters["b1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    b2 = tf.convert_to_tensor(parameters["b2"])
    W3 = tf.convert_to_tensor(parameters["W3"])
    b3 = tf.convert_to_tensor(parameters["b3"])

    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3}

    x = tf.placeholder("float", [12288, 1])

    z3 = forward_propagation_for_predict(x, params)
    p = tf.argmax(z3)

    sess = tf.Session()
    prediction = sess.run(p, feed_dict = {x: X})

    return prediction
