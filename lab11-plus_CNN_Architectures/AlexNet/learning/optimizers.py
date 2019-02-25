import os
import time
from abc import abstractmethod
import tensorflow as tf
from learning.utils import plot_learning_curve


class Optimizer(object):
    """
    Base class for gradient-based optimization algorithm
    경사 하강 러닝 알고리즘 기반 optimizer의 베이스 클래스
    """

    def __init__(self, model, train_set, evaluator, val_set=None, **kwargs):
        """
        Optimizer initializer
        Optimizer 생성자
        :param model: ConvNet, the model to be learned
                               학습할 모델
        :param train_set: DataSet, training set to be used
                                   학습에 사용할 학습 데이터셋
        :param evaluator: Evaluator, for computing performance scores during training
                                     학습 수행 과정에서 성능 평가에 사용할 evaluator
        :param val_set: DataSet, validation set to be used, which can be None if not used
                                 검증 데이터셋, 주어지지 않는 경우 None으로 남겨둘 수 있음
        :param kwargs: dict, extra arguments containing training hyperparameters
                             학습 관련 하이퍼파라미터로 구성된 추가 인자
            - batch_size: int, batch size for each iteration
                               각 반복 회차에서의 미니 배치 크기
            - num_epochs: int, total number of epochs for training
                               총 epoch 수
            - init_learning_rate: float, initial learning rate
                                         학습률 초기값
        """
        self.model = model
        self.train_set = train_set
        self.evaluator = evaluator
        self.val_set = val_set

        # Training hyperparameters
        # 학습 관련 하이퍼파라미터
        self.batch_size = kwargs.pop('batch+size', 256)
        self.num_epochs = kwargs.pop('num_epochs', 320)
        self.init_learning_rate = kwargs.pop('init_learning_rate', 0.01)

        # Placeholder for current learning rate
        # 현 학습률 값의 Placeholder
        self.learning_rate_placeholder = tf.placeholder(tf.float32)
        self.optimize = self._optimize_op()

        self._reset()

    def _reset(self):
        """
        Reset some variables
        일부 변수 재설정
        """
        self.curr_epoch = 1
        # number of bad epochs, where the model is updated without improvement
        # 'bad epochs' 수: 성능 향상이 연속적으로 이루어지지 않는 epochs 수
        self.num_bad_epochs = 0
        # initialize best score with the worst one
        # 최저 성능 점수로, 현 최고 점수를 초기화함
        self.best_score = self.evaluator.worst_score
        # current learning rate
        # 현 학습률 값
        self.curr_learning_rate = self.init_learning_rate

    @abstractmethod
    def _optimize_op(self, **kwargs):
        """
        tf.train.Optimizer.minimize Op for a gradient update
        This should be implemented, and should not be called manually
        경사 하강 업데이트를 위한 tf.train.Optimizer.minimize Op.
        해당 함수를 추후 구현해야 하며, 외부에서 임의로 호출할 수 없음
        """
        pass

    @abstractmethod
    def _update_learning_rate(self, **kwargs):
        """
        Update current learning rate (if needed) on every epoch, by its own schedule
        This should be impelmented, and should not be called manually
        고유 학습률 스케줄링 방법에 따라, (필요한 경우) 매 epoch마다 현 학습률 값을 업데이트함
        해당 함수를 추후 구현해야 하며, 외부에서 임의로 호출할 수 없음
        """
        pass

    def _step(self, sess, **kwargs):
        """
        Make a single gradient update and return its results
        This should not be called manually
        경사 하강 업데이트를 1회 수행하며, 관련된 값을 반환함
        해당 함수를 외부에서 임의로 호출할 수 없음
        :param sess: tf.Session
        :param kwargs: dict, extra arguments containing training hyperparameters
                             학습 관련 하이퍼파라미터로 구성된 추가인자
            - augment_train: bool, value for the single iteration step
                                   학습 과정에서 데이터 증강을 수행할지 여부
        :return loss: float, loss value for the single iteration step
                             1회 반복 회차 결과 손실 함수값
                y_true: np.ndarray, true label from the training set
                                    학습 데이터셋의 실제 레이블
                y_pred: np.ndarray, predicted label from the model
                                    모델이 반환한 예측 레이블
        """
        augment_train = kwargs.pop('augment_train', True)

        # Sample a single batch
        X, y_true = self.train_set.next_batch(self.batch_size, shuffle=True,
                                              augment=augment_train, is_train=True)

        # Compute the loss and make update
        _, loss, y_pred = sess.run([self.optimize, self.model.loss, self.model.pred],
                                   feed_dict={self.model.X:X, self.model.y: y_true,
                                              self.model.is_train: True,
                                              self.learning_rate_placeholder: self.curr_learning_rate})

        return loss, y_true, y_pred

    def train(self, sess, save_dir='/tmp', details=False, verbose=True, **kwargs):
        """
        Run optimizer to train the model
        opimizer를 실행하고, 모델을 학습함
        :param sess: tf.Session
        :param save_dir: str, the directory to save the learned weights of the model
                              학습된 모데르이 파라미터들을 저장할 디렉토리 경로
        :param details: bool, whether to return detailed results
                              학습 결과 관련 구체적인 정보를, 학습 종료 후 반환할지 여부
        :param verbose: bool, whether to print detailed during training
                             학습 과정에서 구체적인 정보를 출력할지 여부
        :param kwargs: dict, extra arguments containing training hyperparameters
                             학습 관련 하이퍼파라미터로 구성된 추가 인자
        :return train_results: dict, containing detailed results of training
                                     구체적인 학습 결과를 담은 dict
        """
        saver = tf.train.Saver()
        # Initialize all weights
        # 전체 파라미터를 초기화함
        sess.run(tf.global_variables_initializer())

        # dictionary to contain training(, evaluation) results and details
        # 학습 (및 검증) 결과 관련 정보를 포함하는 dict
        train_results = dict()
        train_size = self.train_set.num_examples
        num_steps_per_epoch = train_size // self.batch_size
        num_steps = self.num_epochs * num_steps_per_epoch

        if verbose:
            print('Running training loop...')
            print('Number of training iterations: {}'.format(num_steps))

        step_losses, step_scores, eval_scores = [], [], []
        start_time = time.time()

        # Start training loop
        # 학습 루프를 실행함
        for i in range(num_steps):
            # Perform a gradient update from a single minibatch
            # 미니배치 하나로부터 경사 하강 업데이트를 1회 수행함
            step_loss, step_y_true, step_y_pred = self._step(sess, **kwargs)
            step_losses.append(step_loss)

            # Perform evaluation in the end of each epoch
            # 매 epoch의 말미에서, 성능 평가를 수행함
            if (i+1) % num_steps_per_epoch == 0:
                # Evaluate model with current minibatch, from training set
                # 학습 데이터셋으로부터 추출한 미니배치에 대하여 모델의 예측 성능을 평가함
                step_score = self.evaluator.score(step_y_true, step_y_pred)
                step_scores.append(step_score)

                # If validation set is initially given, use it for evaluation
                # 검증 데이터셋이 처음부터 주어진 경우, 이를 사용하여 모델 성능을 평가함
                if self.val_set is not None:
                    # Evaluate model with the validation set
                    # 검증 데이터셋을 사용하여 모델 성능을 평가함
                    eval_y_pred = self.model.predict(sess, self.val_set, verbose=False, **kwargs)
                    eval_score = self.evaluator.score(self.val_set.labels, eval_y_pred)
                    eval_scores.append(eval_score)

                    if verbose:
                        # Print intermediate results
                        # 중간 결과를 출력함
                        print('[epoch {}]\tloss: {:.6f} | Train score: {:.6f} |Eval score: {.6f} |lr: {:.6f}'\
                              .format(self.curr_epoch, step_loss, step_score, eval_score, self.curr_learning_rate))
                        # Plot intermediate results
                        # 중간 결과를 플롯팅함
                        plot_learning_curve(-1, step_losses, step_scores, eval_scores=eval_scores,
                                            model=self.evaluator.mode, img_dir=save_dir)
                    curr_score = eval_score

                # else, just use results from current minibatch for evaluation
                # 그렇지 않은 경우, 단순히 미니배치에 대한 결과를 사용하여 모델 성능을 평가함
                else:
                    if verbose:
                        # Print intermediate results
                        # 중간 결과를 출력함
                        print('[epoch {}]\tloss: {} | Train score: {:.6f} |lr: {:.6f}'\
                              .format(self.curr_epoch, step_loss, step_score, self.curr_learning_rate))
                        # Plot intermediate results
                        plot_learning_curve(-1, step_losses, step_scores, eval_scores=None,
                                            model=self.evaluator.mode, img_dir=save_dir)
                    curr_score = step_score

                # Keep track of the current best model,
                # by comparing current score and the best core
                # 현재의 성능 점수의 현재까지의 최고 성능 점수를 비교하고,
                # 최고 성능 점수가 갱신된 경우 해당 성능을 발휘한 모델의 파라미터들을 저장함
                if self.evaluator.is_better(curr_score, self.best_score, **kwargs):
                    self.best_score = curr_score
                    self.num_bad_epochs = 0
                    # save current weights
                    # 현재 모델의 파라미터들을 저장함
                    saver.save(sess, os.path.join(save_dir, 'model.ckpt'))
                    print('save ok')
                else:
                    self.num_bad_epochs += 1

                self._update_learning_rate(**kwargs)
                self.curr_epoch += 1

        if verbose:
            print('Total training time(sec): {}'.format(time.time() - start_time))
            print('Best {} score: {}'.format('evaluation' if eval else 'training',
                                             self.best_score))

        print('Done')

        if details:
            # Store training results in a dictionary
            train_results['step_losses'] = step_losses  # (num_iterations)
            train_results['step_scores'] = step_scores  # (num_epochs)
            if self.val_set is not None:
                train_results['eval_scores'] = eval_scores  # (num_epochs)

            return train_results


class MomentumOptimizer(Optimizer):
    """
    Gradient descent optimizer, with Momentum algorithm
    모멘텀 알고리즘을 포함한 경사 하강 optimizer 클래스
    """

    def _optimize_op(self, **kwargs):
        """
        경사 하강 업데이트를 위한 tf.train.MomentumOptimizer.minimize Op
        :param kwargs: dict, optimizer의 추가 인자
            - momentum: float, 모멘텀 계수
        :return tf.Operation
        """
        momentum = kwargs.pop('momentum', 0.9)

        update_vars = tf.trainable_variables()
        return tf.train.MomentumOptimizer(self.learning_rate_placeholder, momentum, use_nesterov=False)\
            .minimize(self.model.loss, var_list=update_vars)

    def _update_learning_rate(self, **kwargs):
        """
        Update current learning rate, when evaluation score plateaus
        성능 평가 점수 상에 개선이 없을 때, 현 학습률 값을 업데이트함
        :param kwargs: dict, extra arguments for learning rate scheduling
                             학습률 스케줄링을 위한 추가 인자
            - learning_rate_patience: int, number of epochs with no improvement
                                           after which learning rate will be reduced
                                           성능 향상이 연속적으로 이루어지지 않는 epoch 수가
                                           해당 값을 초과할 경우, 학습률 값을 감소시킴
            - learning_rate_decay: float, factor by which the learning rate will be updated
                                          학습률 업데이트 비율
            - eps: float, if the difference between new and old learning rate is smaller than eps,
                          the update is ignored
                          업데이트된 학습률 값과 기존 학습률 값 간의 차이가 해당 값보다 작을 경우,
                          학습률 업데이트를 취소함
        """
        learning_rate_patience = kwargs.pop('learning_rate_patience', 10)
        learning_rate_decay = kwargs.pop('learning_rate_decay', 0.1)
        eps = kwargs.pop('eps', 1e-8)

        if self.num_bad_epochs > learning_rate_patience:
            new_learning_rate = self.curr_learning_rate * learning_rate_decay
            # Decay learning rate only when the difference is higher than epsilon
            # 새 학습률 값과 기존 학습률 값 간의 차이가 eps보다 큰 경우에 한해서만 업데이트를 수행함
            if self.curr_learning_rate - new_learning_rate > eps:
                self.curr_learning_rate = new_learning_rate
            self.num_bad_epochs = 0
