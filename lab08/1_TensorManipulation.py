import os
import tensorflow as tf
import numpy as np
import pprint

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.set_random_seed(777)  # for reproducibility

pp = pprint.PrettyPrinter(indent=4)
sess = tf.InteractiveSession()

###### Simple Array #####

t = np.array([0., 1., 2., 3., 4., 5., 6.])
pp.pprint(t)
print(t.ndim)  # rank
print(t.shape)  # shape
print(t[0], t[1], t[-1])
print(t[2:5], t[4:-1])
print(t[:2], t[3:])

##### 2D Array #####
t = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]])
pp.pprint(t)
print(t.ndim)  # rank
print(t.shape)  # shape

##### Shape(모양), Rank(차원), Axis(축) ######

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

##### Matmul vs. multiply #####
# Because of Broadcasting!!
matrix1 = tf.constant([[1., 2.], [3., 4.]])
matrix2 = tf.constant([[1.], [2.]])
print("Matrix 1 shape", matrix1.shape)
print("Matrix 2 shape", matrix2.shape)
# Matmul
pp.pprint(tf.matmul(matrix1, matrix2).eval())
# multiply
pp.pprint((matrix1 * matrix2).eval())

##### Broadcasting #####
# Operations between the same shapes
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2., 2.]])
pp.pprint((matrix1 + matrix2).eval())

matrix1 = tf.constant([[1., 2.]])
matrix2 = tf.constant(3.)
pp.pprint((matrix1 + matrix2).eval())  # matrix2: 3. => [[3., 3.]]

matrix1 = tf.constant([[1., 2.]])
matrix2 = tf.constant([3., 4.])
pp.pprint((matrix1 + matrix2).eval())  # matrix2: [3., 4.] => [[3., 4.]]

matrix1 = tf.constant([[1., 2.]])
matrix2 = tf.constant([[3.], [4.]])
pp.pprint((matrix1 + matrix2).eval())  # matrix1: [[1., 2.], [1., 2.]], matrix2: [[3., 3.], [4., 4.]]

##### Reduce mean #####
# Be careful INT type
pp.pprint(tf.reduce_mean([1, 2], axis=0).eval())

x = [[1., 2.],
     [3., 4.]]

# Compute mean of all elements
pp.pprint(tf.reduce_mean(x).eval())
# When axis = 0,
# 같은 열끼리 평균을 냄
pp.pprint(tf.reduce_mean(x, axis=0).eval())
# When axis = 1,
# 같은 행끼리 평균을 냄
pp.pprint(tf.reduce_mean(x, axis=1).eval())
# (axis = -1) == (axis = Rank - 1)
# axis = -1은 가장 큰 axis를 뜻함
pp.pprint(tf.reduce_mean(x, axis=-1).eval())

##### Reduce sum #####
x = [[1., 2.],
     [3., 4.]]

# Compute sum of all elements
pp.pprint(tf.reduce_sum(x).eval())
# When axis = 0,
pp.pprint(tf.reduce_sum(x, axis=0).eval())
# When axis = 1,
pp.pprint(tf.reduce_sum(x, axis=1).eval())

pp.pprint(tf.reduce_mean(tf.reduce_sum(x, axis=-1)).eval())

##### Argmax #####
x = [[0, 1, 2],
     [2, 1, 0]]

pp.pprint(tf.argmax(x, axis=0).eval())

pp.pprint(tf.argmax(x, axis=1).eval())

pp.pprint(tf.argmax(x, axis=-1).eval())


##### Reshape #####
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
pp.pprint(tf.squeeze([[0], [1], [2]]).eval())
# Rank를 늘려줌
pp.pprint(tf.expand_dims([0, 1, 2], 1).eval())

##### One hot #####
# depth: The number of classes
#        나누어질 class의 개수를 뜻함
# 즉, 0 => [1, 0, 0]
# tf.one_hot()은 기본적으로 한 차원 더 늘어나게 설계되어있음
pp.pprint(tf.one_hot([[0], [1], [2], [0]], depth=3).eval())
# one_hot()을 2차원 배열로 쓰기 위해서는 다음과 같이
# shape를 변경하는 별도의 작업 필요함
t = tf.one_hot([[0], [1], [2], [0]], depth=3)
pp.pprint(tf.reshape(t, shape=[-1, 3]).eval())

##### Casting #####
pp.pprint(tf.cast([1.8, 2.2, 3.3, 4.9], tf.int32).eval())

pp.pprint(tf.cast([True, False, 1 == 1, 0 == 1], tf.int32).eval())

##### Stack #####
x = [1, 4]
y = [2, 5]
z = [3, 6]

# Pack along first dim.
pp.pprint(tf.stack([x, y, z]).eval())

pp.pprint(tf.stack([x, y, z], axis=1).eval())

##### Ones and Zeros like #####
x = [[0, 1, 2],
     [2, 1, 0]]

# shape는 x와 동일하나, 모두 1로 채워짐
pp.pprint(tf.ones_like(x).eval())
# shape는 x와 동일하나, 모두 0으로 채워짐
pp.pprint(tf.zeros_like(x).eval())

##### Zip #####
# zip을 통해 복수의 matrix를 처리할 수 있음
for x, y in zip([1, 2, 3], [4, 5, 6]):
    print(x, y)

for x, y, z in zip([1, 2, 3], [4, 5, 6], [7, 8, 9]):
    print(x, y, z)
