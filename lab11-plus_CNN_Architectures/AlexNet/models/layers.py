import tensorflow as tf


def weight_variable(shape, stddev=0.01):
    """
    Initialize a weight variable with given shape,
    by sampling randomly from Normal(0.0, stddev^2)
    새로운 가중치 변수를 주어진 shape에 맞게 선언하고,
    Normal(0.0, stddev^2)의 정규분포로부터의 샘플링을 통해 초기화함
    :param shape: list(int)
    :param stddev: float, standard deviation of Normal distribution for weights
                   샘플링 대상이 되는 정규분포의 표준편차 값
    :return weights: tf.Variable
    """
    weights = tf.get_variable('weights', shape, tf.float32,
                              tf.random_normal_initializer(mean=0.0, stddev=stddev))
    return weights


def bias_variable(shape, value=1.0):
    """
    Initialize a bias variable with given shape,
    with given constant value
    새로운 바이어스 변수를 주어진 shape에 맞게 선언하고,
    주어진 상수값으로 초기화함
    :param shape: list(int)
    :param value: float, initial value for biases, 바이어스 초기화 값
    :return: tf.Variable
    """
    biases = tf.get_variable('biases', shape, tf.float32,
                             tf.constant_initializer(value=value))
    return biases


def conv2d(x, W, stride, padding='SAME'):
    """
    Compute a 2D convolution from given input and filter weights
    주어진 입력값과 필터 가중치 간의 2D 컨볼루션을 수행함
    :param x: tf.Tensor, shape: (N, H, W, C)
    :param W: tf.Tensor, shape: (fh, fw, ic, oc)
    :param stride: int, the stride of the sliding window for each dimension
                        필터의 각 방향으로의 이동 간격
    :param padding: str, either 'SAME' or 'VALID'
                         the type of padding algorithm to use
                         'SAME' 또는 'VALID',
                         컨볼루션 연산시 입력값에 대해 적용할 패딩 알고리즘
    :return: tf.Tensor
    """
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)


def max_pool(x, side_l, stride, padding='SAME'):
    """
    Perform max pooling on given input
    주어진 입력값에 대해 최댓값 풀링(max pooling)을 수행함
    :param x: tf.Tensor, shape: (N, H, W, C)
    :param side_l: int, the side length of the pooling window for each dimension
                        풀링 윈도우의 한 변의 길이
    :param stride: int, the stride of the sliding window for each dimension
                        풀링 윈도우의 각 방향으로의 이동 간격
    :param padding: str, either 'SAME' or 'VALID',
                         the type of padding algorithm to use
                         'SAME' 또는 'VALID'
                         풀링 연산 시 입력값에 대해 적용할 패딩 알고리즘
    :return: tf.Tensor
    """
    return tf.nn.max_pool(x, ksize=[1, side_l, side_l, 1], strides=[1, stride, stride, 1], padding=padding)


def conv_layer(x, side_l, stride, out_depth, padding='SAME', **kwargs):
    """
    Add a new convolutional layer
    새로운 컨볼루션 층을 추가함
    :param x: tf.Tensor, shape: (N, H, W, C)
    :param side_l: int, the side length of the filters for each dimension
                        필터의 한 변의 길이
    :param stride: int, the stride of the filters for each dimension
                        필터의 각 방향으로의 이동 간격
    :param out_depth: int, the total number of filters to be applied
                           입력값에 적용할 필터의 총 개수
    :param padding: str, either 'SAME' or 'VALID'
                         the type of padding algorithm to use
                         'SAME' 또는 'VALID'
                         컨볼루션 연산 시 입력값에 대해 적용할 패딩 알고리즘
    :param kwargs: dict, extra arguments, including weights/biases initialization hyperparameters
                         추가 인자, 가중치/바이어스 초기화를 위한 하이퍼파라미터들을 포함함
        - weight_stddev: float, standard deviation of Normal distribution for weights
                                샘플링 대상이 되는 정규분포의 표준편차 값
        - biases_value: float, initial value for biases
                               바이어스의 초기화 값
    :return: tf.Tensor
    """
    weight_stddev = kwargs.pop('weights_stddev', 0.01)
    biases_value = kwargs.pop('biases_value', 0.1)
    in_depth = int(x.get_shape()[-1])

    filters = weight_variable([side_l, side_l, in_depth, out_depth], stddev=weight_stddev)
    biases = bias_variable([out_depth], value=biases_value)
    return conv2d(x, filters, stride, padding=padding) + biases


def fc_layer(x, out_dim, **kwargs):
    """
    Add a new fully-connect layer
    새로운 완전 연결 층을 추가함
    :param x: tf.Tensor, shape: (N, D)
    :param out_dim: int, the dimension of output vector
                         출력 벡터의 차원 수
    :param kwargs: dict, extra arguments, including weights/biases initialization hyperparameters
                         추가 인자, 가중치/바이어스 초기화를 위한 하이퍼파라미터들을 포함함
        - weight_stddev: float, standard deviation of Normal distribution for weights
                                샘플링 대상이 되는 정규분포의 표준편차 값
        - biases_value: float, initial value for biases
                               바이어스의 초기화 값
    :return: tf.Tensor
    """
    weights_stddev = kwargs.pop('weights_stddev', 0.01)
    biases_value = kwargs.pop('biases_value', 0.1)
    in_dim = int(x.get_shape()[-1])

    weights = weight_variable([in_dim, out_dim], stddev=weights_stddev)
    biases = bias_variable([out_dim], value=biases_value)
    return tf.matmul(x, weights) + biases
