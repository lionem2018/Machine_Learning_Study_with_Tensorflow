# reference: https://github.com/terryum/TensorFlow_Exercises/blob/master/3a_MLP_MNIST_160516.ipynb

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.set_random_seed(777)  # for reproducibility


# function to set input layer, hidden layer, output layer
def model_myNN(_X, _W, _b):
    Layer1 = tf.nn.sigmoid(tf.add(tf.matmul(_X, _W['W1']), _b['b1']))
    Layer2 = tf.nn.sigmoid(tf.add(tf.matmul(Layer1, _W['W2']), _b['b2']))
    Layer3 = tf.add(tf.matmul(Layer2, _W['W3']), _b['b3'])
    # Layer3 = tf.nn.sigmoid(tf.add(tf.matmul(Layer2,_W['W3']), _b['b3']))
    return Layer3


# Load MNIST data (training sets and testing sets)
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
X_train = mnist.train.images
Y_train = mnist.train.labels
X_test = mnist.test.images
Y_test = mnist.test.labels\

# the number of inputs in NN
# 네트워크 상의 입력 개수
dimX = X_train.shape[1]
# the number of outputs in NN
# 네트워크 상의 출력 개수
dimY = Y_train.shape[1]
# the number of training data
# training 데이터 개수
nTrain = X_train.shape[0]
# the number of testing data
# testing 데이터 개수
nTest = X_test.shape[0]

# print("Shape of (X_train, X_test, Y_train, Y_test)")
# print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

# the number of outputs from each layer
# 각 레이어의 출력 개수
# 단, 시작과 끝은 각각 네트워크 상의 입력 개수, 출력 개수와 동일
nLayer0 = dimX
nLayer1 = 256
nLayer2 = 256
nLayer3 = dimY

# 표준편차 지정
sigma_init = 0.1   # For randomized initialization

# set weights of each layer randomly
W = {
    'W1': tf.Variable(tf.random_normal([nLayer0, nLayer1], stddev=sigma_init)),
    'W2': tf.Variable(tf.random_normal([nLayer1, nLayer2], stddev=sigma_init)),
    'W3': tf.Variable(tf.random_normal([nLayer2, nLayer3], stddev=sigma_init))
}
# set bias of each layer randomly
b = {
    'b1': tf.Variable(tf.random_normal([nLayer1])),
    'b2': tf.Variable(tf.random_normal([nLayer2])),
    'b3': tf.Variable(tf.random_normal([nLayer3]))
}

# create placeholder for X, Y
X = tf.placeholder(tf.float32, [None, dimX], name="input")
Y = tf.placeholder(tf.float32, [None, dimY], name="output")

# set logits
Y_pred = model_myNN(X, W, b)

# cost/loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Y_pred, labels=Y))

# parameters
training_epochs = 30
display_epoch = 5
batch_size = 100   # For each time, we will use 100 samples to update parameters
learning_rate = 0.001

# set optimizer (you can change to GradientDescentOptimizer)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# predict results and compute accuracy
# 예측과 정확도 계산
correct_prediction = tf.equal(tf.argmax(Y_pred, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        # the number of batch chunk
        # 한 epoch를 수행하기 위해 몇 개의 Batch 덩어리를 수행할 것인가
        nBatch = int(nTrain / batch_size)
        # np.random.permutation(<number>): 0에서 <number>-1까지 랜덤하게 배치된 배열 반환
        myIdx = np.random.permutation(nTrain)
        for ii in range(nBatch):
            X_batch = X_train[myIdx[ii * batch_size:(ii + 1) * batch_size], :]
            Y_batch = Y_train[myIdx[ii * batch_size:(ii + 1) * batch_size], :]
            # print X_batch.shape, Y_batch.shape
            sess.run(optimizer, feed_dict={X: X_batch, Y: Y_batch})

        # display results when (5 * N) epoch
        # (5 * N) 번째 epoch가 되면 결과 출력
        if (epoch + 1) % display_epoch == 0:
            loss_temp, accuracy_temp = sess.run([loss, accuracy], feed_dict={X: X_train, Y: Y_train})
            print("(epoch {})".format(epoch + 1))
            print("[Loss / Training Accuracy] {:05.4f} / {:05.4f}".format(loss_temp, accuracy_temp))
            print(" ")

    print("[Test Accuracy] ", accuracy.eval({X: X_test, Y: Y_test}))

