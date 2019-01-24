# Lab03. Minimize cost of Linear Regression - (3)

    import tensorflow as tf

    # tf Graph Input
    X = [1, 2, 3]
    Y = [1, 2, 3]

    # Set wrong model weight
    # W = tf.Variable(5.0)
    W = tf.Variable(-3.0)

    # Linear model
    hypothesis = X * W
    # cost/loss function
    cost = tf.reduce_mean(tf.square(hypothesis - Y))
    # Minimize: Gradient Descent Magic
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train = optimizer.minimize(cost)

    # Launch the graph in a session
    sess = tf.Session()
    # Initializes global variables in the graph
    sess.run(tf.global_variables_initializer())

    for step in range(10):
        print(step, sess.run(W))
        sess.run(train)

[return]

0 -3.0

1 0.7333336

2 0.98222226

3 0.9988148

4 0.99992096

5 0.9999947

6 0.99999964

7 0.99999994

8 1.0

9 1.0
