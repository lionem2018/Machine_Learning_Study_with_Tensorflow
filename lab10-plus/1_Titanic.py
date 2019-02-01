# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github,com/kaggle/docker-python
# For example, here;s several helpful packages to load in
# https://www.kaggle.com/klepacz/tensor-flow
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split

import tensorflow as tf

# Input data files are available in the "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


# 파일에서 데이터를 읽어오는 함수
def load_file(is_test):
    # 데이터 셋의 용도에 따라 불러오는 파일을 달리함
    if is_test:
        data_df = pd.read_csv("./input/test.csv")
    else:
        data_df = pd.read_csv("./input/train.csv")

    # Pclass: 승선권 클래스, Sex: 성, Age: 나이, Fare: 티켓의 요금, Embarked: 승선한 항(위치, 항구)
    cols = ["Pclass", "Sex", "Age", "Fare", "Embarked_0", "Embarked_1", "Embarked_2"]

    # female 혹은 male 값을 0과 1로 맵핑
    data_df['Sex'] = data_df['Sex'].map({'female': 0, 'male': 1}).astype(int)

    # handle missing values of age
    # 결측값을 평균으로 채우기
    data_df["Age"] = data_df["Age"].fillna(data_df["Age"].mean())
    data_df["Fare"] = data_df["Fare"].fillna(data_df["Fare"].mean())

    # 아무 값이나 상관 없을 것으로 보아 결측값 'S'로 채우기
    # 'S', 'C', 'Q'에 따라 0, 1, 2로 맵핑
    data_df['Embarked'] = data_df['Embarked'].fillna('S')
    data_df['Embarked'] = data_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
    # Embarked 항목을 one-hot 형태로 변경
    data_df = pd.concat([data_df, pd.get_dummies(data_df['Embarked'], prefix='Embarked')], axis=1)

    # print(data_df.head())
    # 필요한 항목만 추출
    data = data_df[cols].values

    # 데이터 셋의 용도에 따라 결과 분석용의 싱글 컬럼의 값을 달리함
    if is_test:
        sing_col = data_df["PassengerId"].values
    else:
        sing_col = data_df["Survived"].values

    return sing_col, data


# 무작위로 데이터셋과 라벨을 셔플함
def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels


#
def reformat(labels):
    labels = (np.arange(n_classes) == labels[:, None]).astype(np.float32)
    return labels


# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


# load training data set
survived, data_train = load_file(0)

# Split data
X_train, X_test, y_train, y_test = train_test_split(data_train, survived, test_size=0.2, random_state=42)
print('Training set', data_train.shape, survived.shape)

train_dataset, train_labels = randomize(X_train, y_train)
test_dataset, test_labels = randomize(X_test, y_test)

n_classes = 2

train_labels = reformat(train_labels)
test_labels = reformat(test_labels)

print('Training set', train_dataset.shape, train_labels.shape)
print('Testing set', test_dataset.shape, test_labels.shape)

# Parameters
learning_rate = 0.1

# Network Parameter
n_input = 7  # data_train.shape[1]
n_classes = 2  # survived.shape[1]

n_hidden_1 = 32  # 1st layer number of features
n_hidden_2 = 64  # 2nd layer number of features

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# Store layers weights & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=0.01)),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=0.01)),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes], stddev=0.01))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# This is only used during test time
logits = tf.nn.softmax(pred)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

training_epoch = 10
batch_size = 256
display_step = 1
step_size = 1000

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epoch):
        avg_cost = 0

        # Loop over step_size
        for step in range(step_size):
            # Pick an offset within the training data, which has been randomized
            # Note: we could use better randomization across epochs.
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            # Generate a minibatch
            batch_data = train_dataset[offset: (offset + batch_size), :]
            batch_labels = train_labels[offset: (offset + batch_size), :]

            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_data, y: batch_labels})
            avg_cost += c / step_size

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%02d' % (epoch + 1), "cost={:.4f}".format(avg_cost))

    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("\nAccuracy:", accuracy.eval({x: test_dataset, y: test_labels}))

    # Build the submission file
    passId, data_test = load_file(1)

    outputs = sess.run([logits], feed_dict={x: data_test})
    outputs = [x[1] for x in outputs[0]]

    submission = ['PassengerId, Survived']

    for prediction, id in zip(outputs, passId):
        submission.append('{0}, {1}'.format(id, int(prediction)))

    submission = '\n'.join(submission)

    with open('submission.csv', 'w') as outfile:
        outfile.write(submission)
