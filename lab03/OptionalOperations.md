# Lab03. Minimize cost of Linear Regression - (4)

    import tensorflow as tf

    X = [1, 2, 3]
    Y = [1, 2, 3]

    # Set wrong model weights
    W = tf.Variable(5.)

    # Linear model
    hypothesis = X * W
    # Manual gradient
    gradient = tf.reduce_mean((W * X - Y) * X) * 2
    # cost/loss function
    cost = tf.reduce_mean(tf.square(hypothesis - Y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

    # 수동으로 gradient를 구한 후(compute_gradients()), 적용하도록(apply_gradients()) 함수 호출
    # 아래 두 라인은 train = optimizer.minimize(cost)로 대체 가능
    # Get gradients 
    gvs = optimizer.compute_gradients(cost)
    # Apply gradients
    apply_gradients = optimizer.apply_gradients(gvs)

    # Launch the graph in a session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for step in range(100):
        print(step, sess.run([gradient, W, gvs]))
        sess.run(apply_gradients)

[return]

0 [37.333332, 5.0, [(37.333336, 5.0)]]

1 [33.84889, 4.6266665, [(33.84889, 4.6266665)]]

2 [30.689657, 4.2881775, [(30.689657, 4.2881775)]

. . .

97 [0.0027837753, 1.0002983, [(0.0027837753, 1.0002983)]]

98 [0.0025234222, 1.0002704, [(0.0025234222, 1.0002704)]]

99 [0.0022875469, 1.0002451, [(0.0022875469, 1.0002451)]]
