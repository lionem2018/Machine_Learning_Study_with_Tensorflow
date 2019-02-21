from abc import abstractmethod, abstractproperty
from sklearn.metrics import accuracy_score


class Evaluator(object):
    """
    Base class for evaluation functions
    성능 평가를 위한 evaluator의 베이스 클래스(추상 클래스)
    """

    @abstractproperty
    def worst_score(self):
        """
        The worst performance score
        최저 성능 점수
        :return: float
        """
        pass

    @abstractproperty
    def mode(self):
        """
        The mode for performance score, either 'max' or 'min'
        e.g. 'max' for accuracy, AUC, precision and recall,
             and 'min' for error rate, FNR and FPR
        점수가 높아야 성능이 우수한지, 낮아야 성능이 우수한지 여부. 'max'와 'min' 중 하나
        e.g. 정확도, AUC, 정밀도, 재현율 등의 경우 'max',
             오류율, 미검률, 오검률 등의 경우 'min'
        :return: str.
        """
        pass

    @abstractmethod
    def score(self, y_true, y_pred):
        """
        Performacne metric for a given prediction
        This should be implemented
        실제로 사용할 성능 평가 지표
        해당 함수를 추후 구현해야 함
        지정한 성능 평가 척도에 의거하여 성능 점수를 계산하여 반환함
        :param y_true: np.ndarray, shape: (N, num_classes)
        :param y_pred: np.ndarray, shape: (N, num_classes)
        :return: float
        """
        pass

    @abstractmethod
    def is_better(self, curr, best, **kwargs):
        """
        Function to return whether current performance score is better than current best
        This should be implemented
        현재 주어진 성능 점수가 현재까지의 최고 성능 점수보다 우수한지 여부를 반환하는 함수
        해당 함수를 추후 구현해야 함
        :param curr: float, current performance to be evaluated, 평가 대상이 되는 현재 성능 점수
        :param best: float, current best performance, 현재까지의 최고 성능 점수
        :return: bool
        """
        pass


class AccuracyEvaluator(Evaluator):
    """
    Evaluator with accuracy metric
    정확도를 평가 척도로 사용하는 evaluator 클래스
    """

    @property
    def worst_score(self):
        """
        The worst performance score
        최저 성능 점수
        """
        return 0.0

    @property
    def mode(self):
        """
        The mode for performance score
        점수가 높아야 성능이 우수한지, 낮아야 성능이 우수한지 여부
        """
        return 'max'

    def score(self, y_true, y_pred):
        """
        Compute accuracy for a given prediction
        정확도에 기반한 성능 평가 점수
        """
        return accuracy_score(y_true.argmax(axis=1), y_pred.argmax(axis=1))

    def is_better(self, curr, best, **kwargs):
        """
        Return whether current performance score is better than current best,
        with consideration of the relative threshold to the given performance score
        상대적 문턱값을 고려하여, 현재 주어진 성능 점수가 현재까지의 최고 성능 점수보다
        우수한지 여부를 반환하는 함수
        즉, 두 성능 간의 단순 비교를 수행하는 것이 아니라,
        상대적 문턱값을 사용하여 현재 평가 성능이 최고 평가 성능보다 지정한 비율 이상으로 높은 경우에 한해 True 반환
        :param kwargs: dict, extra arguments, 추가 인자
            - score_threshold: float, relative threshold for measuring the new optimum,
                               to only focus on significant changes
                               새로운 최적값 결정을 위한 상대적 문턱값으로,
                               유의미한 차이가 발생했을 경우만을 반영하기 위함
        """
        score_threshold = kwargs.pop('score_threshold', 1e-4)
        relative_eps = 1.0 + score_threshold
        return curr > best * relative_eps
