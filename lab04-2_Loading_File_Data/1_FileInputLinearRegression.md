# Lab04-2. Loading Data From File with TensorFlow - (1)

## Slicing
    
    nums = range(5)  # range is a built-in function that creates a list of integers
    print nums  # Prints "[0, 1, 2, 3, 4]"
    print nums[2:4]  # Get a slice from index 2 to 4 (exclusive); prints "[2, 3]"
    print nums[2:]  # Get a slice from index 2 to the end; prints "[2, 3, 4]"
    print nums[:2]  # Get a slice from the start to index 2 (exclusive); prints "[0, 1]"
    print nums[:]  # Get a slice of the whole list; prints "[0, 1, 2, 3, 4]"
    print nums[:-1]  # Slice indices can be negative; prints "[0, 1, 2, 3]"
    nums[2:4] = [8, 9]  # Assign a new sublist to a slice
    print nums # Prints "[0, 1, 8, 9, 4]"
    
## Loading Data From File

    import numpy as np

    xy = np.loadtxt('data-01-test-score.csv', delimiter=',', dtype=np.float32)
    x_data = xy[:, 0:-1]
    y_data = xy[:, [-1]]

    # Make sure the shape and data are OK
    print(x_data, "\nx_data shape:", x_data.shape)
    print(y_data, "\ny_data shape:", y_data.shape)
    
[return]
[[ 73.  80.  75.]

 [ 93.  88.  93.]
 
 [ 89.  91.  90.]
 
. . .

 [ 78.  83.  85.]
 
 [ 76.  83.  71.]
 
 [ 96.  93.  95.]] 
 
x_data shape: (25, 3)

[[152.]

 [185.]
 
 [180.]
 
. . .

 [175.]

 [149.]
 
 [192.]] 
 
y_data shape: (25, 1)

## Training Linear Regression Using File Data

    import tensorflow as tf
    
    . . .

    # placeholders from a tensor that will be always fed
    X = tf.placeholder(tf.float32, shape=[None, 3])
    Y = tf.placeholder(tf.float32, shape=[None, 1])

    W = tf.Variable(tf.random_normal([3, 1]), name='weight')
    b = tf.Variable(tf.random_normal([1]), name='bias')

    # Hypothesis
    hypothesis = tf.matmul(X, W) + b

    # Simplified cost/loss function
    cost = tf.reduce_mean(tf.square(hypothesis - Y))

    # Minimize
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
    train = optimizer.minimize(cost)

    # Launch the graph in a session
    sess = tf.Session()
    # Initializes global variables in the graph
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                                       feed_dict={X: x_data, Y: y_data})
        if step % 10 == 0:
            print(step, "Cost:", cost_val, "\nPrediction:\n", hy_val)
            
[return]

0 Cost: 21027.002 

Prediction:

 [[22.048063 ]
 
 [21.619787 ]
 
 [24.096693 ]
 
. . .

 [15.924551 ]
 
 [31.36112  ]
 
 [24.986364 ]]
 
10 Cost: 95.97634 

Prediction:

 [[157.11063]
 
 [183.99281]
 
 [184.06302]
 
. . .

  [161.75204]
  
 [167.48862]
 
 [193.25117]]
 
20 Cost: 94.257256 

Prediction:

 [[158.01503 ]
 
 [185.11975 ]
 
 [185.15112 ]
 
. . .

 [162.77228 ]
 
 [168.35251 ]
 
 [194.40303 ]]
 
. . .


1980 Cost: 25.00525 

Prediction:

 [[154.44974]

 [185.55818]

 [182.92065]

. . .

 [164.71878]

 [158.30646]

 [192.80391]]

1990 Cost: 24.863276 

Prediction:

 [[154.43932]

 [185.55841]

 [182.91356]

. . .

[164.72627]

 [158.27443]

 [192.79778]]

2000 Cost: 24.722479 

Prediction:

 [[154.42892]

 [185.5586 ]

 [182.90646]

. . . 

 [164.73372]

 [158.24257]

 [192.79166]]
 