import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.set_random_seed(777)  # for reproducibility

# tf.data를 이용하여 파일에서 데이터 읽어오기
# .skip(1) : 첫번째 줄은 제외하고(헤더)
# .repeat() : 파일의 끝에 도달하더라도 처음부터 무한 반복
# .batch(10) : 한 번에 10개씩 묶어서 사용
iterator = tf.data.TextLineDataset("data-01-test-score.csv").skip(1).repeat().batch(10).make_initializable_iterator()

# 반복자가 다음 데이터를 읽어오도록 dataset 노드에 명령을 저장
dataset = iterator.get_next()

# csv를 읽어서 데이터로 변환한다.
lines = tf.decode_csv(dataset, record_defaults=[[0.], [0.], [0.], [0.]])
# 변환된 데이터의 첫번째 열부터 마지막-1번째 열까지 합쳐서 train_x_batch에 할당
train_x_batch = tf.stack(lines[0:-1], axis=1)
# 마지막 열을 합쳐서 train_y_batch에 할당
train_y_batch = tf.stack(lines[-1:], axis=1)

# placeholders for a tensor that will be always fed
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
sess.run(iterator.initializer)

for step in range(2001):
    # sess.run(iterator.initializer)
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
    cost_val, hy_val, _ = sess.run(
        [cost, hypothesis, train], feed_dict={X: x_batch, Y: y_batch})
    if step % 10 == 0:
        print(step, "Cost:", cost_val, "\nPrediction:\n", hy_val)
