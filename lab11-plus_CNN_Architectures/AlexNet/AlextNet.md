# Lab11-plus. CNN Architectures - (1)

## AlextNet

- [datasets/asirra.py](#datasets/asirra.py)
- [learning/evaluators.py](#learning/evaluators.py)
- [learning/optimizers.py](#learning/optimizers.py)
- [learning/utils.py](#learning/utils.py)
- [models/layers.py](#models/layers.py)
- [models/nn.py](#models/nn.py)
- [train.py](#train.py)
- [test.py](#test.py)

## datasets/asirra.py

    import os
    import numpy as np
    from skimage.io import imread
    from skimage.transform import resize
    
    
    def read_asirra_subset(subset_dir, one_hot=True, sample_size=None):
        """
        디스크로부터 Asirra Dogs vs. Cats 데이터셋을 로드하고,
        AlexNet을 학습하는 데 사용하기 위한 형태로 전처리를 수행함
        :param subset_dir: str, 원본 데이터셋이 저장된 디렉토리 경로
        :param one_hot: bool, one-hot 인코딩 형태ㅡ이 레이블을 반환할 것인지 여부
        :param sample_size: int, 전체 데이터셋을 모두 사용하지 않는 경우, 사용하고자 하는 샘플 이미지의 개수
        :return:  X_set: np.ndarray, shape: (N, H, W, C)
                   y_set: np.ndarray, shape: (N, num_channels) or (N, )
        """
        # 학습 + 검증 데이터셋을 읽어들임
        filename_list = os.listdir(subset_dir)
        set_size = len(filename_list)
    
        if sample_size is not None and sample_size < set_size:
            # sample_size가 명시된 경우, 원본 중 일부를 랜덤하게 샘플링
            # np.random.choice(<list> or <int>, <size>, <replace>, <p>): 배열 상에서의 데이터에서 size 만큼 랜덤으로 선택
            #                                                            정수가 첫번째 인자로 전달되었다면 arange(<int>)와 같은 효과
            #                                                            <p>는 배열로, 각 데이터가 선택될 수 있는 확률들을 나타냄
            filename_list = np.random.choice(filename_list, size=sample_size, replace=False)
            set_size = sample_size
        else:
            # 단순히 filename list의 순서를 랜덤하게 섞음
            np.random.shuffle(filename_list)
    
        # 데이터 array들을 메모리 공간에 미리 할당
        # X_set: 256*256 크기의 RGB 이미지를 담을 set_size 만큼의 배열
        # y_set: set_size 각각의 label(0 또는 1)을 담을 배열
        X_set = np.empty((set_size, 256, 256, 3), dtype=np.float32)   # (N, H, W, C)
        y_set = np.empty((set_size), dtype=np.uint8)                  # (N)
    
        # 랜덤하게 섞인 filename_list에서 file을 하나씩 가져옴
        for i, filename in enumerate(filename_list):
            if i % 1000 == 0:
                # 가져올 수(set_size)에서 총 몇 개를 가져왔는지 표시
                print('Reading subset data: {}/{}...'.format(i, set_size), end="\r")
            # file 이름의 맨 앞에 오는 문자열에 따라 label 변환
            # cat == 0, dog == 1
            label = filename.split('.')
            if label == 'cat':
                y = 0
            else:  # label == 'dog'
                y = 1
            # subset_dir + filename의 file_path 생성
            file_path = os.path.join(subset_dir, filename)
            # 해당 경로의 이미지를 불러와 256*256 크기로 resize (비율 유지)
            img = imread(file_path)  # shape: (H, W, 3), range: [0, 255]
            img = resize(img, (256, 256), mode='constant').astype(np.float32)  # (256, 256, 3), [0.0, 1.0]
            X_set[i] = img
            y_set[i] = y
    
        # 0 또는 1로 표현되어있는 label을 (1, 0) 또는 (0, 1) 형태로 변환
        if one_hot:
            # 모든 레이블들을 one-hot 인코딩 벡터들로 변환, shape: (N, num_classes)
            y_set_oh = np.zeros((set_size, 2), dtype=np.uint8)
            y_set_oh[np.arange(set_size), y_set] = 1
            y_set = y_set_oh
    
        print('\nDone')
    
        return X_set, y_set
    
    
    def random_crop_reflect(images, crop_1):
        """
        Perform random cropping and reflection from images
        원본 256*256 크기의 이미지로부터 crop_1*crop_1(여기서는 227*227) 크기의 patch를 랜덤한 위치에서 추출,
        50% 확률로 해당 패치에 대한 수평 방향으로의 대칭 변환
        1 이미지 -> 1 패치 (학습 단계에서 사용)
        :param images: np.ndarray, shape: (N, H, W, C)
        :param crop_1: int, a side length of crop region
        :return: np.ndarray, shape: (N, h, w, C)
        """
        # crop할 이미지들의 높이와 너비를 구함
        H, W = images.shape[1:3]
        augmented_images = []
        for image in images:  # image.shape: (H, W, C)
            # Randomly crop patch
            # 0부터 H-crop_1-1까지 중 하나를 crop할 시작점의 y 좌표로 잡음
            # 0부터 W-crop_1-1까지 중 하나를 crop할 시작점의 x 좌표로 잡음
            y = np.random.randint(H-crop_1)
            x = np.random.randint(W-crop_1)
            # 각 이미지를 crop_1 사이즈 만큼 crop
            image = image[y:y+crop_1, x:x+crop_1]  # (h, w, C)
    
            # Randomly reflect patch horizontally
            # 랜덤하게 이미지 뒤집기(좌우반전)
            reflect = bool(np.random.randint(2))
            if reflect:
                image = image[:, ::-1]
    
            augmented_images.append(image)
    
        return np.stack(augmented_images)  # shape: (N, h, w, C)
    
    
    def corner_center_crop_reflect(images, crop_1):
        """
        Perform 4 corners and center cropping and reflection from images,
        resulting in 10x augmented patches
        원본 256*256 크기 이미지에서의 좌측 상단, 우측 상단, 좌측 하단, 우측 하단, 중심 위치 각각으로부터
        총 5개의 crop_1*crop_1(여기서는 227*227) 패치를 추출하고, 이들 각각에 대해 수평 방향 대칭 변환
        1 이미지 -> 10 패치 (테스트 단계에서 사용)
        :param images: np.ndarray, shape: (N, H, W, C)
        :param crop_1: int, a side length of crop region
        :return: np.ndarray, shape: (N, 10, h, w, C)
        """
        H, W = images.shape[1:3]
        augmented_images = []
        for image in images:  # image.shape: (H, W, C)
            aug_image_orig = []
            # Crop image in 4 corners
            aug_image_orig.append(image[:crop_1, :crop_1])
            aug_image_orig.append(image[:crop_1, -crop_1:])
            aug_image_orig.append(image[-crop_1:, :crop_1])
            aug_image_orig.append(image[-crop_1:, -crop_1:])
            # Crop image in the center
            aug_image_orig.append(image[H//2-(crop_1//2):H//2+(crop_1-crop_1//2),
                                        W//2-(crop_1//2):W//2+(crop_1-crop_1//2)])
            aug_image_orig = np.stack(aug_image_orig)  # (5, h, w, C)
    
            # Flip augmented images and add it
            aug_image_flipped = aug_image_orig[:, :, ::-1]  # (5, h, w, C)
            # np.concatenate: 여러 배열을 한 번에 합침
            aug_image = np.concatenate((aug_image_orig, aug_image_flipped), axis=0)  # (10, h, w, C)
            augmented_images.append(aug_image)
    
        return np.stack(augmented_images)  # shape: (N, 10, h, w, C)
    
    
    def center_crop(images, crop_1):
        """
        Perform center cropping of images
        256*256 이미지의 중앙을 crop_1*crop_1(여기서는 227*227) 사이즈로 크롭
        1 이미지 -> 1 패치
        :param images: np.ndarray, shape: (N, H, W, C)
        :param crop_1: int, a side length of crop region
        :return: np.ndarray, shape: (N, h, w, C)
        """
        H, W = images.shape[1:3]
        cropped_images = []
        for image in images:  # image.shape: (H, W, C)
            # Crop image in the center
            cropped_images.append(image[H//2-(crop_1//2):H//2+(crop_1-crop_1//2),
                                        W//2-(crop_1//2):W//2+(crop_1-crop_1//2)])
    
        return np.stack(cropped_images)
    
    
    class DataSet(object):
        """
         데이터셋 요소를 클래스화
        """
        def __init__(self, images, labels=None):
            """
            새로운 DataSet 객체를 생성함
            :param images: np.ndarray, shape: (N, H, W, C)
            :param labels: np.ndarray, shape: (N, num_classes) or (N)
            """
            if labels is not None:
                assert images.shape[0] == labels.shape[0], (
                    'Number of examples mismatch, between images and labels'
                )
            self._num_examples = images.shape[0]
            self._images = images
            self._labels = labels
            self._indices = np.arange(self._num_examples, dtype=np.uint)
            self._reset()
    
        def _reset(self):
            """일부 변수를 재설정함"""
            self._epoch_completed = 0
            self._index_in_epoch = 0
    
        @property
        def images(self):
            return self._images
    
        @property
        def labels(self):
            return self._labels
    
        @property
        def num_examples(self):
            return self._num_examples
    
        def next_batch(self, batch_size, shuffle=True, augment=True, is_train=True, fake_data=False):
            """
            'batch_size' 개수 만큼의 이미지들을 현재 데이터셋으로부터 추출하여 미니배치 형태로 반환
            :param batch_size: int, 미니배치 크기
            :param shuffle: bool, 미니배치 추출에 앞서, 현재 데이터셋 내 이미지들의 순서를 랜덤하게 섞을 것인지 여부
            :param augment: bool, 미니배치를 추출할 때, 데이터 증강을 수행할 것인지 여부
            :param is_train: bool, 미니배치 추출을 위한 현재 상황(학습/예측)
            :param fake_data: bool, (디버깅 목적으로) 가짜 이미지 데이터를 생성할 것인지 여부
            :return: batch_images: np.ndarray, shape: (N, h, w, C) or (N, 10, h, w, C)
                     batch_labels: np.ndarray, shape: (N, num_classes) or (N)
            """
            if fake_data:
                fake_batch_images = np.random.random(size=(batch_size, 227, 227, 3))
                fake_batch_labels = np.zeros((batch_size, 2), dtype=np.uint8)
                fake_batch_labels[np.arange(batch_size), np.random.randint(2, size=batch_size)] = 1
                return fake_batch_images, fake_batch_labels
    
            start_index = self._index_in_epoch
    
            # 맨 첫번째 epoch에서는 전체 데이터셋을 랜덤하게 섞음
            if self._epoch_completed == 0 and start_index == 0 and shuffle:
                np.random.shuffle(self._indices)
    
            # 현재의 인덱스가 전체 이미지 수를 넘어간 경우, 다음 epoch을 진행
            if start_index + batch_size > self._num_examples:
                # 완료된 epoch 수를 1 증가
                self._epoch_completed += 1
                # 새로운 epoch에서 남은 이미지들을 가져옴
                rest_num_examples = self._num_examples - start_index
                indices_rest_part = self._indices[start_index:self._num_examples]
    
                # 하나의 epoch이 끝나면, 전체 데이터셋을 섞음
                if shuffle:
                    np.random.shuffle(self._indices)
    
                # 다음 epoch 시작
                start_index = 0
                self._index_in_epoch = batch_size - rest_num_examples
                end_index = self._index_in_epoch
                indices_new_part = self._indices[start_index:end_index]
    
                images_rest_part = self.images[indices_rest_part]
                images_new_part = self.images[indices_new_part]
                batch_images = np.concatenate((images_rest_part, images_new_part), axis = 0)
                if self.labels is not None:
                    labels_rest_part = self.labels[indices_rest_part]
                    labels_new_part = self.labels[indices_new_part]
                    batch_labels = np.concatenate((labels_rest_part, labels_new_part), axis=0)
    
                else:
                    batch_labels = None
    
            else:
                self._index_in_epoch += batch_size
                end_index = self._index_in_epoch
                indices = self._indices[start_index:end_index]
                batch_images = self.images[indices]
                if self.labels is not None:
                    batch_labels = self.labels[indices]
                else:
                    batch_labels = None
    
            if augment and is_train:
                # 학습 상황에서의 데이터 증강을 수행
                # 256*256 크기의 1 이미지 -> 227*227 크기의 크롭된 (추가로 50% 확률로 수평 대칭 변환 수행된) 1 패치
                batch_images = random_crop_reflect(batch_images, 227)
            elif augment and not is_train:
                # Perform data augmentation, for evaluation phase(10x)
                # 테스트 상황에서의 데이터 증강을 수행
                # 256*256 크기의 1 이미지 -> 227*227 크기의 크롭된 (추가로 수평 대칭 변환 수행된) 10 패치
                batch_images = corner_center_crop_reflect(batch_images, 227)
            else:
                # Don't perform data augmentation, generating center-cropped patches
                # 데이터 증강을 수행하지 않고 이미지만 크롭
                # 256*256 크기의 1 이미지 -> 227*227 크기의 중앙 크롭된 1 패치
                batch_images = center_crop(batch_images, 227)
    
            return batch_images, batch_labels

## learning/evaluators.py

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
            
## learning/optimizers.py

## learning/utils.py

## models/layers.py

[return]

Done

Training set stats:

(10000, 256, 256, 3)

0.0 1.0

0 10000

Validation set stats:

(2500, 256, 256, 3)

0.0 1.0

0 2500

conv1.shape [None, 55, 55, 96]

pool1.shape [None, 27, 27, 96]

conv2.shape [None, 27, 27, 256]

pool2.shape [None, 13, 13, 256]

conv3.shape [None, 13, 13, 384]

conv4.shape [None, 13, 13, 384]

conv5.shape [None, 13, 13, 256]

pool5.shape [None, 6, 6, 256]

drop6.shape [None, 4096]

drop7.shape [None, 4096]

Running training loop...

Number of training iterations: 11700

Total training time(sec): 23.094427347183228

Best evaluation score: 0.0

Done
