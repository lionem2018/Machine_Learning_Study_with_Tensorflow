import os
import numpy as np
import tensorflow as tf
from datasets import asirra as dataset
from models.nn import AlexNet as ConvNet
from learning.evaluators import AccuracyEvaluator as Evaluator


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

"""
1. Load dataset
   원본 데이터셋을 메모리에 로드함
"""
root_dir = os.path.join('./', 'asirra')    # FIXME
test_dir = os.path.join(root_dir, 'test')

# Load test set
# 테스트 데이터셋을 로드함
X_test, y_test = dataset.read_asirra_subset(test_dir, one_hot=True)
test_set = dataset.DataSet(X_test, y_test)

# Sanity check
# 중간 점검
print('Test set stats:')
print(test_set.images.shape)
print(test_set.images.min(), test_set.images.max())
print((test_set.labels[:, 1] == 0).sum(), (test_set.labels[:, 1] == 1).sum())


"""
2. Set test hyperparameters
   테스트를 위한 하이퍼파라미터 설정
"""
hp_d = dict()
image_mean = np.load('./results/asirra_mean.npy')  # load mean image
hp_d['image_mean'] = image_mean

# FIXME: Test hyperparameters
#        테스트 관련 하이퍼파라미터
hp_d['batch_size'] = 256
hp_d['augment_pred'] = True


"""
3. Build graph, load weights, initialize a session and start test
   Graph 생성, 파라미터 로드, session 초기화 및 테스트 시작
"""
# Initialize
# 초기화
graph = tf.get_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

model = ConvNet([227, 227, 3], 2, **hp_d)
evaluator = Evaluator()
saver = tf.train.Saver()

sess = tf.Session(graph=graph, config=config)
saver.restore(sess, 'results/model.ckpt')  # restore learned weights, 학습된 파라미터 로드 및 복원
test_y_pred = model.predict(sess, test_set, **hp_d)
test_score = evaluator.score(test_set.labels, test_y_pred)

print('Test accuracy: {}'.format(test_score))
