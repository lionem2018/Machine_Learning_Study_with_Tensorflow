import os
import tensorflow as tf
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.set_random_seed(777)  # for reproducibility

dataset = pd.read_csv("./data-03-diabetes.csv", delimiter=",")
dataset = dataset.values

x_data = dataset[:701, :-1]
y_data = dataset[:701, [-1]]
x_test = dataset[701:, :-1]
y_test = dataset[701:, [-1]]

X = tf.placeholder(dtype=tf.float32, shape=[None, 8])
Y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

W1 = tf.Variable(tf.random_normal([8, 3]), name='weight1')
b1 = tf.Variable(tf.random_normal([3]), name='bias1')
layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([3, 3]), name='weight2')
b2 = tf.Variable(tf.random_normal([3]), name='bias2')
layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2)

W3 = tf.Variable(tf.random_normal([3, 3]), name='weight3')
b3 = tf.Variable(tf.random_normal([3]), name='bias3')
layer3 = tf.sigmoid(tf.matmul(layer2, W3) + b3)

W4 = tf.Variable(tf.random_normal([3, 1]), name='weight4')
b4 = tf.Variable(tf.random_normal([1]), name='bias4')

hypothesis = tf.sigmoid(tf.matmul(layer3, W4) + b4)
cost = - tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, cost_val)

    # Accuracy report
    h, c, a = sess.run([hypothesis, prediction, accuracy], feed_dict={X: x_test, Y: y_test})
    print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)
