#Lab10. Improved NN Training - (2)

## Xavier Initialization

![picture_xavier_he](picture_xavier_he.PNG)

    import tensorflow as tf
    import matplotlib.pyplot as plt
    import random
    
    tf.set_random_seed(777)  # for reproducibility
    
    from tensorflow.examples.tutorials.mnist import input_data
    
    # Check out https://www.tensor-flow.org/get_started/mnist/beginners for
    # more information about the mnist dataset
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    
    # parameters
    learning_rate = 0.001
    training_epoch = 15
    batch_size = 100
    
    nb_classes = 10
    
    # MNIST data image of shape 28 * 28 = 784
    X = tf.placeholder(tf.float32, [None, 784])
    # 0 - 9 digits recognition = 10 classes
    Y = tf.placeholder(tf.float32, [None, nb_classes])
    
    # weights & bias for nn layers
    # tf.get_variable(<name>, <shape>, <initializer>): Use initializer different from tf.Variable()
                                                       직접 값을 주는 tf.Vairable()과 달리, Initializer를 사용
    W1 = tf.get_variable("W1", shape=[784, 256], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.Variable(tf.random_normal([256]))
    L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
    
    W2 = tf.get_variable("W2", shape=[256, 256], initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.Variable(tf.random_normal([256]))
    L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
    
    W3 = tf.get_variable("W3", shape=[256, 10], initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.Variable(tf.random_normal([nb_classes]))
    hypothesis = tf.matmul(L2, W3) + b3
    
    # define cost/loss & optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=hypothesis, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    # initialize
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    # train my model
    for epoch in range(training_epoch):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)
    
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            feed_dict = {X: batch_xs, Y: batch_ys}
            c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
            avg_cost += c / total_batch
    
        print('Epoch:', '%04d' % (epoch + 1), 'cost=', '{:.9f}'.format(avg_cost))
    
    print("Learning finished!")
    
    # Test the model and check accuracy
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('Accuracy:', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))
    
    # Get one and predict
    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label:", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
    print("Prediction:", sess.run(tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r: r + 1]}),)
    
    # Show random MNIST image
    plt.imshow(
        mnist.test.images[r:r + 1].reshape(28, 28),
        cmap="Greys",
        interpolation="nearest",
    )
    plt.show()

[return]

Epoch: 0001 cost= 0.301023195

Epoch: 0002 cost= 0.114273898

Epoch: 0003 cost= 0.074346720

Epoch: 0004 cost= 0.052530829

Epoch: 0005 cost= 0.038945061

Epoch: 0006 cost= 0.029831838

Epoch: 0007 cost= 0.025161842

Epoch: 0008 cost= 0.020610626

Epoch: 0009 cost= 0.014980993

Epoch: 0010 cost= 0.014886079

Epoch: 0011 cost= 0.012384883

Epoch: 0012 cost= 0.009630271

Epoch: 0013 cost= 0.008312832

Epoch: 0014 cost= 0.012351731

Epoch: 0015 cost= 0.008281912

Learning finished!

Accuracy: 0.9759

Label: [7]

Prediction: [7]


## Original vs. Xavier

    # Original
    W1 = tf.Variable(tf.random_normal([784, 256]), name='weight1')
    
    # Xavier
    W1 = tf.get_variable("W1", shape=[784, 256], initializer=tf.contrib.layers.xavier_initializer())
