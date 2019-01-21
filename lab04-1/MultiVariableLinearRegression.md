# Lab04-1. Implement Multi-Variable Linear Regression with TensorFlow - (1)

    import tensorflow as tf

    tf.set_random_seed(777)  # for reproducibility

    x1_data = [73., 93., 89., 96., 73.]
    x2_data = [80., 88., 91., 98., 66.]
    x3_data = [75., 93., 90., 100., 70.]

    y_data = [152., 185., 180., 196., 142]

    # placeholders for a tensor that will be always fed
    x1 = tf.placeholder(tf.float32)
    x2 = tf.placeholder(tf.float32)
    x3 = tf.placeholder(tf.float32)

    Y = tf.placeholder(tf.float32)

    w1 = tf.Variable(tf.random_normal([1]), name='weight1')
    w2 = tf.Variable(tf.random_normal([1]), name='weight2')
    w3 = tf.Variable(tf.random_normal([1]), name='weight3')
    b = tf.Variable(tf.random_normal([1]), name='bias')

    hypothesis = x1 * w1 + x2 * w2 + x3 * w3 + b

    # cost/loss function
    cost = tf.reduce_mean(tf.square(hypothesis - Y))

    # Minimize. Need a very small learning rate for this data set
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
    train = optimizer.minimize(cost)

    # Launch the graph is a session
    sess = tf.Session()
    # Initializes global variables in the graph
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                                       feed_dict={x1: x1_data, x2: x2_data, x3: x3_data, Y: y_data})
        if step % 10 == 0:
            print(step, "Cost:", cost_val, "\nPrediction:\n", hy_val)

[return]

0 Cost: 62547.29 

Prediction:

[-75.96345  -78.27629  -83.83015  -90.80436  -56.976482]

10 Cost: 14.468628 

Prediction:

 [145.26407 187.59541 178.152   194.48586 145.81136]
 
20 Cost: 13.822504 

Prediction:

 [145.9478  188.39003 178.94907 195.3521  146.41212]
 
. . .

1980 Cost: 4.9475894 

Prediction:

 [148.14153 186.8927  179.62645 195.81604 144.46869]
 
1990 Cost: 4.9222474 

Prediction:

 [148.15    186.8869  179.62906 195.81778 144.4612 ]
 
2000 Cost: 4.8970113 

Prediction:

 [148.15845 186.8811  179.63167 195.81953 144.45372]
