# Lab11-3. Improve Deep CNN - (2)

## Using TensorFlow Layers

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
    
    
    class Model:
    
        def __init__(self, sess, name):
            self.sess = sess
            self.name = name
            self._build_net()
    
        def _build_net(self):
            with tf.variable_scope(self.name):
                # dropout (keep_prob) rate 0.7~0.5 on training, but should be 1
                # for testing
                self.training = tf.placeholder(tf.bool)
    
                # input place holders
                self.X = tf.placeholder(tf.float32, [None, 784])
    
                # img 28*28*1 (black/white), Input Layer
                X_img = tf.reshape(self.X, [-1, 28, 28, 1])
                self.Y = tf.placeholder(tf.float32, [None, 10])
    
                # Convolutional Layer #1
                # 파라미터를 설정해주면 나머지 필요한 정보들은 X_img를 통해 자동으로 알아냄 (W1의 크기 등)
                conv1 = tf.layers.conv2d(inputs=X_img, filters=32, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)
                # Pooling Layer #1
                pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], padding="SAME", strides=2)
                # tf.layers.dropout(<inputs>, <rate>, <training>): <training>에 boolean 형 값을 전달하여 rate 사용 여부를 설정할 수 있음
                                                                   training과 test에 따라 각기 다른 rate를 주는 데 있어서 발생하는 실수 방지
                dropout1 = tf.layers.dropout(inputs=pool1, rate=0.3, training=self.training)
    
                # Convolutional Layer #2 and Pooling Layer #2
                conv2 = tf.layers.conv2d(inputs=dropout1, filters=64, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)
                pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], padding="SAME", strides=2)
                dropout2 = tf.layers.dropout(inputs=pool2, rate=0.3, training=self.training)
    
                # Convolutional Layer #3 and Pooling Layer #3
                conv3 = tf.layers.conv2d(inputs=dropout2, filters=128, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)
                pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], padding="SAME", strides=2)
                dropout3 = tf.layers.dropout(inputs=pool3, rate=0.3, training=self.training)
    
                # Dense(fully connected) Layer with Relu
                # tf.layers.dense(<Input>, <OutputNum>, <Activative Function>): Input을 입력받아 OutputNum 만큼의 출력을 하는 1 layer 생성
                flat = tf.reshape(dropout3, [-1, 4 * 4 * 128])
                dense4 = tf.layers.dense(inputs=flat, units=625, activation=tf.nn.relu)
                dropout4 = tf.layers.dropout(inputs=dense4, rate=0.5, training=self.training)
    
                # Logits (no activation) Layer: L5 Final FC 625 inputs -> 10 outputs
                self.logits = tf.layers.dense(inputs=dropout4, units=10)
    
            # define cost/loss & optimizer
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.Y))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
    
            correct_prediction = tf.equal(
                tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    [
        def predict(self, x_test, training=False):
            return self.sess.run(self.logits, feed_dict={self.X: x_test, self.training: training})
    
        def get_accuracy(self, x_test, y_test, training=True):
            return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.training: training})
    
        def train(self, x_data, y_data, training=True):
            return self.sess.run([self.cost, self.optimizer], feed_dict={
                self.X: x_data, self.Y: y_data, self.training: training})
    
    
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

Epoch: 0001 cost = 0.287732364

Epoch: 0002 cost = 0.088516344

Epoch: 0003 cost = 0.069093782

Epoch: 0004 cost = 0.055456760

Epoch: 0005 cost = 0.047444916

Epoch: 0006 cost = 0.044083203

Epoch: 0007 cost = 0.040865059

Epoch: 0008 cost = 0.035836818

Epoch: 0009 cost = 0.034940255

Epoch: 0010 cost = 0.033088212

Epoch: 0011 cost = 0.031969194

Epoch: 0012 cost = 0.029649430

Epoch: 0013 cost = 0.027662150

Epoch: 0014 cost = 0.028236628

Epoch: 0015 cost = 0.026297147

Learning Finished!

Accuracy: 0.9893
