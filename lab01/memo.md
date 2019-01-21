# Lab01. Start TensorFlow

## Remove Error message

    # if your console prints
    # 'Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX',
    # add these lines
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

## Print HelloWorld with TensorFlow
   
    import tensorflow as tf
        
    # Create a constant op
    # This op is added as a node to the default graph
    # tf.constant: 상수 텐서 생성
    hello = tf.constant('Hello, TensorFlow!')
    
    # start a TF session
    # tf.Session(): 그래프 실행
    sess = tf.Session()
    
    # run the op and get result
    # sess.run(<parameter>: parameter로 전달된 op(s)에 대응하는
    #                       그래프의 부분집합 실행
    print(sess.run(hello))
    
[return]

b'Hello, TensorFlow!'

-'b' indicates Bytes literals.-

## Create an Adding Node
    
    node1 = tf.constant(3.0, tf.float32)
    node2 = tf.constant(4.0)  # also tf.float32 implicitly
    node3 = tf.add(node1, node2)
    
    print("node1:", node1, "node2:", node2)
    print("node3:", node3)
    
    print("sess.run(node1, node2):", sess.run([node1, node2]))
    print("sess.run(node3):", sess.run(node3))
    
[return]

node1: Tensor("Const_1:0", shape=(), dtype=float32) node2: Tensor("Const_2:0", shape=(), dtype=float32)

node3: Tensor("Add:0", shape=(), dtype=float32)

sess.run(node1, node2): [3.0, 4.0]

sess.run(node3): 7.0

    
## Use Placeholders

![picture_graph](picture_graph.png)
    
    # tf.placeholder(): 실행하기 직전 데이터 전달 가능
    # # placeholder(
    #       dtype,        # 데이터 타입 (반드시 명시)
    #       shape=None,   # 입력 데이터의 형태(상수 or 다차원 배열)
    #       name=None     # 해당 placeholder의 이름
    #   )
    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)
    adder_node = a + b  # + provides a shortcut for tf.add(a,b)
    
    # feed_dict={}: 각 텐서에 값 맵핑
    print(sess.run(adder_node, feed_dict={a: 3, b: 4.5}))
    print(sess.run(adder_node, feed_dict={a: [1, 3], b: [2, 4]}))
   
[return]

7.5

[3. 7.]
    
## Tensors
    
    3  # a rank 0 tensor; this is a scalar with shape [ ]
    [1., 2., 3.]  # a rank 1 tensor; this is a vector with shape [3]
    [[1., 2., 3.], [4., 5., 6.]]  # a rank 2 tensor; a matrix with shape [2, 3]
    [[[1., 2., 3.]], [7., 8., 9.]]  # a rank 3 tensor with shape [2, 1, 3]
