# Lab08. Tensor Manipulation

'Ctrl + F'을 통해 검색하시면 보다 쉽게 원하는 내용을 보실 수 있습니다.

- [Import Package](#import-package)
- [Simple Array](#simple-array)
- [2D Array](#2d-array)
- [Shape, Rank, Axis](#shape-rank-axis)
- [Matmul vs. Multiply](#matmul-vs-multiply)
- [Broadcasting](#broadcasting)
- [Reduce Mean](#reduce-mean)
- [Reduce Sum](#reduce-sum)
- [Argmax](#argmax)
- [Reshape](#Reshape)
- [One Hot](#one-hot)
- [Casting](#casting)
- [Stack](#stack)
- [Ones and Zeros Like](#ones-and-zeros-like)
- [Zip](#zip)

## Import Package

    import tensorflow as tf
    import numpy as np
    import pprint
    
    tf.set_random_seed(777)  # for reproducibility
    
    pp = pprint.PrettyPrinter(indent=4)
    sess = tf.InteractiveSession()

## Simple Array

    t = np.array([0., 1., 2., 3., 4., 5., 6.])
    pp.pprint(t)
    print(t.ndim)  # rank
    print(t.shape)  # shape
    print(t[0], t[1], t[-1])
    print(t[2:5], t[4:-1])
    print(t[:2], t[3:])

[return]

array([0., 1., 2., 3., 4., 5., 6.])

1

(7,)

0.0 1.0 6.0

[2. 3. 4.] [4. 5.]

[0. 1.] [3. 4. 5. 6.]

## 2D Array

    t = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]])
    pp.pprint(t)
    print(t.ndim)  # rank
    print(t.shape)  # shape

[return]

array([[ 1.,  2.,  3.],

[ 4.,  5.,  6.],

[ 7.,  8.,  9.],
       
[10., 11., 12.]])
       
2

(4, 3)

## Shape, Rank, Axis
    
    # Shape(모양), Rank(차원), Axis(축)
    # Rank 구하는 법: 맨 처음 나오는 '['의 개수 
    # Shape 구하는 법: Rank를 구한 뒤, 가장 깊은 원소의 개수부터 구함
    # if Rank = 4,  [A, B, C, D] <= 깊은 곳부터 원소의 개수 구하기 D -> C -> B -> A
    # Axis 구하는 법: Rank-1의 개수(0 ~ Rank-1)를 가짐,
    #                 가장 깊은 원소를 나누는 축이 가장 큰 수
    #
    # ex 1)
    # Shape: [4]
    # Rank: 1
    # Axis: 0
    t = tf.constant([1, 2, 3, 4])
    pp.pprint(tf.shape(t).eval())
    
    # ex 2)
    # Shape: [2, 2]
    # Rank: 2
    # Axis: 0 ~ 1
    t = tf.constant([[1, 2],
                     [3, 4]])
    pp.pprint(tf.shape(t).eval())
    
    # ex 3)
    # Shape: [1, 2, 3, 4]
    # Rank: 4
    # Axis: 0 ~ 3
    t = tf.constant([[[[1, 2, 3, 4],
                       [5, 6, 7, 8],
                       [9, 10, 11, 12]],
                       [[13, 14, 15, 16],
                       [17, 18, 19, 20],
                       [21, 22, 23, 24]]]])
    pp.pprint(tf.shape(t).eval())
    
    # [
    #     [
    #         [
    #             [1,2,3,4], 
    #             [5,6,7,8],
    #             [9,10,11,12]
    #         ],
    #         [
    #             [13,14,15,16],
    #             [17,18,19,20], 
    #             [21,22,23,24]
    #         ]
    #     ]
    # ]

[return]

array([4])

array([2, 2])

array([1, 2, 3, 4])

## Matmul vs. Multiply
    
    # Because of Broadcasting!!
    matrix1 = tf.constant([[1., 2.], [3., 4.]])
    matrix2 = tf.constant([[1.], [2.]])
    print("Matrix 1 shape", matrix1.shape)
    print("Matrix 2 shape", matrix2.shape)
    # Matmul
    pp.pprint(tf.matmul(matrix1, matrix2).eval())
    # multiply
    pp.pprint((matrix1 * matrix2).eval())
    
[return]

Matrix 1 shape (2, 2)

Matrix 2 shape (2, 1)

array([[ 5.],

[11.]], dtype=float32)
       
array([[1., 2.],

[6., 8.]], dtype=float32)

## Broadcasting

    # Operations between the same shapes
    matrix1 = tf.constant([[3., 3.]])
    matrix2 = tf.constant([[2., 2.]])
    pp.pprint((matrix1 + matrix2).eval())
    
    # If the shapes is not same, Second matrix will change the same shape of first thing
    # 만약 shape가 동일하지 않다면, 자동으로 맞춰줌
    matrix1 = tf.constant([[1., 2.]])
    matrix2 = tf.constant(3.)
    pp.pprint((matrix1 + matrix2).eval())  # matrix2: 3. => [[3., 3.]]
    
    matrix1 = tf.constant([[1., 2.]])
    matrix2 = tf.constant([3., 4.])
    pp.pprint((matrix1 + matrix2).eval())  # matrix2: [3., 4.] => [[3., 4.]]
    
    matrix1 = tf.constant([[1., 2.]])
    matrix2 = tf.constant([[3.], [4.]])
    pp.pprint((matrix1 + matrix2).eval())  # matrix1: [[1., 2.], [1., 2.]], matrix2: [[3., 3.], [4., 4.]]
    
[return]

array([[5., 5.]], dtype=float32)

array([[4., 5.]], dtype=float32)

array([[4., 6.]], dtype=float32)

array([[4., 5.],

[5., 6.]], dtype=float32)

## Reduce Mean

    # Be careful INT type
    pp.pprint(tf.reduce_mean([1, 2], axis=0).eval())
    
    x = [[1., 2.],
         [3., 4.]]
    
    # Compute mean of all elements
    pp.pprint(tf.reduce_mean(x).eval())
    # when axis = 0,
    # 같은 열끼리 평균을 냄
    # [((1. + 3.) / 2), ((2. + 4.) / 2)]
    pp.pprint(tf.reduce_mean(x, axis=0).eval())
    # when axis = 1,
    # 같은 행끼리 평균을 냄
    # [((1. + 2.) / 2), ((3. + 4.) / 2)]
    pp.pprint(tf.reduce_mean(x, axis=1).eval())
    # (axis = -1) == (axis = Rank - 1)
    # axis = -1은 가장 큰 axis를 뜻함
    pp.pprint(tf.reduce_mean(x, axis=-1).eval())

[return]

1

2.5

array([2., 3.], dtype=float32)

array([1.5, 3.5], dtype=float32)

array([1.5, 3.5], dtype=float32)

## Reduce Sum

    # Compute sum of all elements
    # (1. + 2. + 3. + 4.)
    pp.pprint(tf.reduce_sum(x).eval())
    # When axis = 0,
    # [(1. + 3.), (2. + 4.)]
    pp.pprint(tf.reduce_sum(x, axis=0).eval())
    # When axis = 1,
    # [(1. + 2.), (3. + 4.)]
    pp.pprint(tf.reduce_sum(x, axis=1).eval())
    # step1) [(1. + 2.), (3. + 4.)] = [3., 7.]
    # step2) (3. + 7.) / 2 = 5.0
    pp.pprint(tf.reduce_mean(tf.reduce_sum(x, axis=-1)).eval())
    
[return]

10.0

array([4., 6.], dtype=float32)

array([3., 7.], dtype=float32)

5.0

## Argmax

    x = [[0, 1, 2],
         [2, 1, 0]]
    
    # 비교 대상들 중 몇 번째 원소가 제일 값이 큰지 리턴
    # [ 0 < 2 => 1,  1 == 1 => 0, 2 > 0 => 0]
    pp.pprint(tf.argmax(x, axis=0).eval())
    # [ 0 < 1 < 2 => 2, 2 > 1 > 0 => 0]
    pp.pprint(tf.argmax(x, axis=1).eval())
    # [ 0 < 1 < 2 => 2, 2 > 1 > 0 => 0]
    pp.pprint(tf.argmax(x, axis=-1).eval())
    
[return]

array([1, 0, 0], dtype=int64)

array([2, 0], dtype=int64)

array([2, 0], dtype=int64)

## Reshape

    t = np.array([[[0, 1, 2],
                   [3, 4, 5]],
                 [[6, 7, 8],
                  [9, 10, 11]]])
    
    pp.pprint(t.shape)
    # 맨 안쪽 원소의 개수는 그대로 둠
    # -1: 다른 나머지 차원 크기를 맞추고 남은 크기를
    #     해당 차원에 할당하는 것
    pp.pprint(tf.reshape(t, shape=[-1, 3]).eval())
    
    pp.pprint(tf.reshape(t, shape=[-1, 1, 3]).eval())
    # Rank를 줄여줌
    # [[0], [1], [2]] => [0, 1, 2]
    pp.pprint(tf.squeeze([[0], [1], [2]]).eval())
    # Rank를 늘려줌
    # [0, 1, 2] => [[0], [1], [2]]
    pp.pprint(tf.expand_dims([0, 1, 2], 1).eval())
    
[return]

(2, 2, 3)

array([[ 0,  1,  2],

[ 3,  4,  5],
       
[ 6,  7,  8],
       
[ 9, 10, 11]])
       
array([[[ 0,  1,  2]],

[[ 3,  4,  5]],

[[ 6,  7,  8]],

[[ 9, 10, 11]]])

array([0, 1, 2])

array([[0],

[1],

[2]])

## One Hot
    # depth: The number of classes
    #        나누어질 class의 개수를 뜻함
    # 즉, 0 => [1, 0, 0]
    # tf.one_hot()은 기본적으로 한 차원 더 늘어나게 설계되어있음
    pp.pprint(tf.one_hot([[0], [1], [2], [0]], depth=3).eval())
    
    # one_hot()을 2차원 배열로 쓰기 위해서는 다음과 같이
    # shape를 변경하는 별도의 작업 필요함
    t = tf.one_hot([[0], [1], [2], [0]], depth=3)
    pp.pprint(tf.reshape(t, shape=[-1, 3]).eval())
    
[return]

array([[[1., 0., 0.]],

[[0., 1., 0.]],

[[0., 0., 1.]],

[[1., 0., 0.]]], dtype=float32)
array([[1., 0., 0.]

[0., 1., 0.],

[0., 0., 1.],

[1., 0., 0.]], dtype=float32)

## Casting

    pp.pprint(tf.cast([1.8, 2.2, 3.3, 4.9], tf.int32).eval())

    pp.pprint(tf.cast([True, False, 1 == 1, 0 == 1], tf.int32).eval())
    
[return]

array([1, 2, 3, 4])

array([1, 0, 1, 0])

## Stack

    x = [1, 4]
    y = [2, 5]
    z = [3, 6]
    
    # Pack along first dim.
    # 각각의 matrix를 행을 중심으로 나누어 한 matrix에 저장
    pp.pprint(tf.stack([x, y, z]).eval())
    # 각각의 matrix를 열을 중심으로 나누어 한 matrix에 저장
    # x = [1, 4]  =>  [1] [4]
    pp.pprint(tf.stack([x, y, z], axis=1).eval())
    
[return]

array([[1, 4],

[2, 5],
       
[3, 6]])
       
array([[1, 2, 3],

[4, 5, 6]])

## Ones and Zeros Like

    x = [[0, 1, 2],
         [2, 1, 0]]
    
    # shape는 x와 동일하나, 모두 1로 채워짐
    pp.pprint(tf.ones_like(x).eval())
    # shape는 x와 동일하나, 모두 0으로 채워짐
    pp.pprint(tf.zeros_like(x).eval())
    
[return]

array([[1, 1, 1],

[1, 1, 1]])
       
array([[0, 0, 0],

[0, 0, 0]])

## Zip
    # zip을 통해 복수의 matrix를 처리할 수 있음
    # x <= [1, 2, 3]
    # y <= [4, 5, 6]
    for x, y in zip([1, 2, 3], [4, 5, 6]):
        print(x, y)
    
    # x <= [1, 2, 3]
    # y <= [4, 5, 6]
    # z <= [7, 8, 9]
    for x, y, z in zip([1, 2, 3], [4, 5, 6], [7, 8, 9]):
        print(x, y, z)
        
[return]

1 4

2 5

3 6

1 4 7

2 5 8

3 6 9