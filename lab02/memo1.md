# Lab02 Implement Linear Regression with TensorFlow - (1)

    import tensorflow as tf

    ########## Build graph using TF operations ##########

    # X and Y data
    x_train = [1, 2, 3]
    y_train = [1, 2, 3]

    # tf.Variable(<parameters>): 변수 생성, 단 초기값을 넘겨주어야 함
    # tf.random_normal(<parameter>): 정규분포로부터 난수 반환, shape 지정해야 함
    W = tf.Variable(tf.random_normal([1]), name='weight')
    b = tf.Variable(tf.random_normal([1]), name='bias')

    # Our hypothesis XW+b
    hypothesis = x_train * W + b

    # cost/loss function
    cost = tf.reduce_mean(tf.square(hypothesis - y_train))

    # Minimize
    # Optimizer.minimize(cost): 최소비용을 찾아주는 함수,
    #                           동시에 내부에서 gradients를 계산하여 W와 b를 변경
    Optimizer  = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train = Optimizer.minimize(cost)

    ########## Run/update graph and get results ##########

    # Launch the graph in a session
    sess = tf.Session()

    # Initializes global variables in the graph
    # tf.global_variables_initializer(): 텐서플로우 구동 전, 연결된 모든 변수 초기화
    sess.run(tf.global_variables_initializer())

    # Fit the line
    for step in range(2001):
        sess.run(train)
        if step % 20 == 0:
            print(step, sess.run(cost), sess.run(W), sess.run(b))
            
[return]

0 3.9867065 [-0.3141922] [0.9445478]

20 0.24609073 [0.3934589] [1.184285]

40 0.19290614 [0.48346514] [1.1556826]

60 0.17492299 [0.51359653] [1.1039459]

. . .

1960 1.8656017e-05 [0.99498343] [0.01140374]

1980 1.6943579e-05 [0.99521923] [0.01086782]

2000 1.538781e-05 [0.99544394] [0.01035707]