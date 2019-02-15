# Lab11-3. Improve Deep CNN - (1)

## Using Python Class

    import tensorflow as tf
    
    from tensorflow.examples.tutorials.mnist import input_data
    
    tf.set_random_seed(777)  # for reproducibility
    
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    # Check out https://www.tensorflow.org/get_started/mnist/beginners for
    # more information about the mnist dataset
    
    # hyper parameters
    learning_rate = 0.001
    training_epochs = 15
    batch_size = 100
    
    
    # Create 'Model' class for CNN
    class Model:
        
        # define initialzier
        def __init__(self, sess, name):
            self.sess = sess
            self.name = name
            self._build_net()
        
        # define X, Y, layers, cost function, optimizer, and accuracy
        def _build_net(self):
            with tf.variable_scope(self.name):
                # dropout (keep_prob) rate 0.7~0.5 on training, but should be 1
                # for testing
                self.keep_prob = tf.placeholder(tf.float32)
    
                # input place holders
                self.X = tf.placeholder(tf.float32, [None, 784])
                # img 28*28*1 (black/white)
                X_img = tf.reshape(self.X, [-1, 28, 28, 1])
                self.Y = tf.placeholder(tf.float32, [None, 10])
    
                # L1 ImgIn shape=(?, 28, 28, 1)
                W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
                #   Conv   -> (?, 28, 28, 32)
                #   Pool   -> (?, 14, 14, 32)
                L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
                L1 = tf.nn.relu(L1)
                L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                L1 = tf.nn.dropout(L1, keep_prob=self.keep_prob)
    
                # L2 ImgIn shape=(?, 14, 14, 32)
                W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
                #   Conv   -> (?, 14, 14, 64)
                #   Pool   -> (?, 7, 7, 64)
                L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
                L2 = tf.nn.relu(L2)
                L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                L2 = tf.nn.dropout(L2, keep_prob=self.keep_prob)
    
                # L3 ImgIn shape=(?, 7, 7, 64)
                W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
                #   Conv   -> (?, 7, 7, 128)
                #   Pool   -> (?, 4, 4, 128)
                L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
                L3 = tf.nn.relu(L3)
                L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                L3 = tf.nn.dropout(L3, keep_prob=self.keep_prob)
    
                L3_flat = tf.reshape(L3, [-1, 4 * 4 * 128])
    
                # L4 FC 4*4*128 inputs -> 625 outputs
                W4 = tf.get_variable("W4", shape=[4 * 4 * 128, 625], initializer=tf.contrib.layers.xavier_initializer())
                b4 = tf.Variable(tf.random_normal([625]))
                L4 = tf.nn.relu(tf.matmul(L3_flat, W4) + b4)
                L4 = tf.nn.dropout(L4, keep_prob=self.keep_prob)
    
                # L5 Final FC 625 inputs -> 10 outputs
                W5 = tf.get_variable("W5", shape=[625, 10], initializer=tf.contrib.layers.xavier_initializer())
                b5 = tf.Variable(tf.random_normal([10]))
                self.logits = tf.matmul(L4, W5) + b5
    
            # define cost/loss & optimizer
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.Y))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
    
            correct_prediction = tf.equal(
                tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
        # define 'Predict' function
        def predict(self, x_test, keep_prob=1.0):
            return self.sess.run(self.logits, feed_dict={self.X: x_test, self.keep_prob: keep_prob})
        
        # define getting accuracy function
        def get_accuracy(self, x_test, y_test, keep_prob=1.0):
            return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.keep_prob:keep_prob})
        
        # define training function
        def train(self, x_data, y_data, keep_prob=0.7):
            return self.sess.run([self.cost, self.optimizer], feed_dict={
                self.X: x_data, self.Y: y_data, self.keep_prob: keep_prob})
    
    
    # initialize
    sess = tf.Session()
    m1 = Model(sess, "m1")
    
    sess.run(tf.global_variables_initializer())
    
    print('Learning started!')
    
    # train my model
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)
    
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = m1.train(batch_xs, batch_ys)
            avg_cost += c / total_batch
    
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
    
    print('Learning Finished!')
    
    # Test model and check accuracy
    print('Accuracy:', m1.get_accuracy(mnist.test.images, mnist.test.labels))

[return]

Learning started!

Epoch: 0001 cost = 0.371345862

Epoch: 0002 cost = 0.099684447

Epoch: 0003 cost = 0.073611460

Epoch: 0004 cost = 0.060713852

Epoch: 0005 cost = 0.054853792

Epoch: 0006 cost = 0.046815755

Epoch: 0007 cost = 0.043182768

Epoch: 0008 cost = 0.041135250

Epoch: 0009 cost = 0.036382303

Epoch: 0010 cost = 0.035693539

Epoch: 0011 cost = 0.033119943

Epoch: 0012 cost = 0.030694664

Epoch: 0013 cost = 0.029657168

Epoch: 0014 cost = 0.027402092

Epoch: 0015 cost = 0.026222912

Learning Finished!

Accuracy: 0.9931
