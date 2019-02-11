# Lab10-plus. Kaggle Example - (1)

## Titanic: Machine Learning from Disaster (from Kaggle)

    # code to downlaod and laod
    import numpy as np  # linear algebra
    import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
    import tensorflow as tf
    
    tf.set_random_seed(777)  # for reproducibility
    
    
    # Normalize x data
    # x 값을 normalize 해주는 함수
    def MinMaxScaler(data):
        numerator = data - np.min(data, 0)
        denominator = np.max(data, 0) - np.min(data, 0)
        # noise term prevents the zero division
        return numerator / (denominator + 1e-7)
    
    
    # 파일에서 데이터를 읽어오는 함수
    def load_file(is_test):
        # 데이터 셋의 용도에 따라 불러오는 파일을 달리함
        if is_test:
            data_df = pd.read_csv("./input/test.csv")
        else:
            data_df = pd.read_csv("./input/train.csv")
    
        # Pclass : 승선권 클래스, Sex: 성, Age: 나이, Fare: 티켓의 요금, Embarked: 승선한 항(위치, 항구)
        cols = ["Pclass", "Sex", "Age", "Fare",
                "Embarked_0", "Embarked_1", "Embarked_2"]
    
        # female 혹은 male 값을 0과 1로 맵핑
        data_df['Sex'] = data_df['Sex'].map({'female': 0, 'male': 1}).astype(int)
    
        # handle missing values of age
        # 결측값을 평균으로 채우기
        data_df["Age"] = data_df["Age"].fillna(data_df["Age"].mean())
        data_df["Fare"] = data_df["Fare"].fillna(data_df["Fare"].mean())
    
        # 어떤 값이 되어도 구하고자 하는 값에는 영향을 주지 않으므로 결측값 'S'로 전부 채우기
        data_df['Embarked'] = data_df['Embarked'].fillna('S')
        # 'S', 'C', 'Q'에 따라 0, 1, 2로 맵핑
        data_df['Embarked'] = data_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
        # Embarked 항목을 one-hot 형태로 변경
        data_df = pd.concat([data_df, pd.get_dummies(data_df['Embarked'], prefix='Embarked')], axis=1)
    
        # print(data_df.head())
        # 필요한 항목만 추출
        data = data_df[cols].values
    
        # 데이터 셋의 용도에 따라 결과 분석용의 싱글 column 값을 달리함
        if is_test:
            sing_col = data_df["PassengerId"].values  # Need it for submission
        else:
            sing_col = data_df["Survived"].values
    
        return sing_col, data
    
    
    # Load data and min/max
    # TODO: clean up this code
    
    # Load training data set
    y_train, x_train = load_file(0)
    y_train = np.expand_dims(y_train, 1)
    train_len = len(x_train)
    
    # Get train file
    passId, x_test = load_file(1)
    
    print(x_train.shape, x_test.shape)
    # np.vstack: 열의 수가 같은 두 개 이상의 배열을 위 아래로 연결
    x_all = np.vstack((x_train, x_test))
    print(x_all.shape)
    
    # 모든 데이터 셋의 x 값을 normalize 한 후 다시 train용과 test용으로 나누어줌
    x_min_max_all = MinMaxScaler(x_all)
    x_train = x_min_max_all[:train_len]
    x_test = x_min_max_all[train_len:]
    print(x_train.shape, x_test.shape)
    
    # Get test labels
    real_result = pd.read_csv("./input/gender_submission.csv")
    real_result = real_result['Survived']
    real_result = np.expand_dims(real_result, 1)
    
    # Parameters
    learning_rate = 0.1
    
    # Network Parameters
    n_input = 7  # x_train.shape[1]
    
    n_hidden_1 = 32  # 1st layer number of features
    n_hidden_2 = 64  # 2nd layer number of features
    
    # placeholders for a tensor that will be always fed.
    X = tf.placeholder(tf.float32, shape=[None, n_input])
    Y = tf.placeholder(tf.float32, shape=[None, 1])
    
    # ReLU - Xavier Initialization
    # W1 = tf.Variable(tf.random_normal([n_input, 3]), name='weight1')
    W1 = tf.get_variable("W1", shape=[n_input, 3], initializer=tf.contrib.layers.variance_scaling_initializer())
    b1 = tf.Variable(tf.random_normal([3]), name='bias1')
    L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
    
    # Sigmoid - He Initialization
    # W2 = tf.Variable(tf.random_normal([3, 1]), name='weight2')
    W2 = tf.get_variable("W2", shape=[3, 1], initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.Variable(tf.random_normal([1]), name='bias2')
    
    # Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(tf.matmul(X, W)))
    hypothesis = tf.sigmoid(tf.matmul(L1, W2) + b2)
    
    # cost/loss function
    cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1 - Y)*tf.log(1 - hypothesis))
    
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    # Accuracy computation
    # True if hypothesis>0.5 else False
    predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
    
    training_epochs = 15
    batch_size = 32
    display_step = 1
    step_size = 1000
    
    # Launch the graph
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
    
        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            avg_accuracy = 0.
            # Loop over step_size
            for step in range(step_size):
                # Pick an offset within the training data, which has been randomized.
                # Note: we could use better randomization across epochs.
                offset = (step * batch_size) % (y_train.shape[0] - batch_size)
                # Generate a minibatch.
                batch_data = x_train[offset:(offset + batch_size), :]
                batch_labels = y_train[offset:(offset + batch_size), :]
    
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c, a = sess.run([optimizer, cost, accuracy], feed_dict={X: batch_data,
                                                                           Y: batch_labels})
                avg_cost += c / step_size
                avg_accuracy += a / step_size
    
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%02d' % (epoch + 1), "cost={:.4f}".format(avg_cost),
                      "train accuracy={:.4f}".format(avg_accuracy))
        print("Optimization Finished!")
    
        # Results (creating submission file)
        # Test model
        correct_prediction = tf.equal(tf.argmax(predicted, 1), tf.argmax(Y, 1))
    
        # Accuracy report
        accuracy_test = sess.run(accuracy, feed_dict={X: x_test, Y: real_result})
        print("\nAccuracy: ", accuracy_test)
    
        outputs = sess.run(predicted, feed_dict={X: x_test})
        submission = ['PassengerId,Survived']
    
        # list.append(<parameter>): 리스트의 맨 마지막에 <parameter>를 추가하는 함수
        for id, prediction in zip(passId, outputs):
            submission.append('{0},{1}'.format(id, int(prediction)))
    
        submission = '\n'.join(submission)
    
        with open('submission.csv', 'w') as outfile:
            outfile.write(submission)

[return]

(891, 7) (418, 7)

(1309, 7)

(891, 7) (418, 7)

Epoch: 01 cost=0.4654 train accuracy=0.7861

Epoch: 02 cost=0.4587 train accuracy=0.7916

Epoch: 03 cost=0.4586 train accuracy=0.7914

Epoch: 04 cost=0.4586 train accuracy=0.7915

Epoch: 05 cost=0.4587 train accuracy=0.7913

Epoch: 06 cost=0.4587 train accuracy=0.7914

Epoch: 07 cost=0.4587 train accuracy=0.7914

Epoch: 08 cost=0.4587 train accuracy=0.7913

Epoch: 09 cost=0.4587 train accuracy=0.7912

Epoch: 10 cost=0.4587 train accuracy=0.7912

Epoch: 11 cost=0.4587 train accuracy=0.7912

Epoch: 12 cost=0.4587 train accuracy=0.7912

Epoch: 13 cost=0.4587 train accuracy=0.7912

Epoch: 14 cost=0.4587 train accuracy=0.7912

Epoch: 15 cost=0.4587 train accuracy=0.7912

Optimization Finished!

Accuracy:  0.9138756


## Reference

### Titanic Overview

https://www.kaggle.com/c/titanic

### Code

https://www.kaggle.com/klepacz/tensor-flow

https://github.com/hunkim/KaggleZeroToAll/blob/master/k0-01-titanic/titanic.ipynb