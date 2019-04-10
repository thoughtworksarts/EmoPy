import os, cv2
import numpy as np

from EmoPy.src.data_loader import _DataLoader
from EmoPy.src.face_detection import FaceDetector


class DirectoryDataLoader(_DataLoader):
    """
    DataLoader subclass loads image and label data from directory.

    :param target_emotion_map: Optional dict of target emotion label values/strings and their corresponding label vector index values.
    :param datapath: Location of image dataset.
    :param validation_split: Float percentage of data to use as validation set.
    :param out_channels: Number of image channels.
    :param time_delay: Number of images to load from each time series sample. Parameter must be provided to load time series data and unspecified if using static image data.
    """

    def __init__(self, target_emotion_map=None, datapath=None, validation_split=0.2, out_channels=1, time_delay=None,
                 faceDetector=FaceDetector()):
        self.datapath = datapath
        self.target_emotion_map = target_emotion_map
        self.out_channels = out_channels
        self.faceDetector = faceDetector
        super().__init__(validation_split, time_delay)

    def load_data(self):
        label_directories = [dir for dir in os.listdir(self.datapath) if not dir.startswith('.')]
        emotion_index_map = {emotion: index for index, emotion in enumerate(label_directories)}
        all_image_files = []
        labels = []
        for label_directory in label_directories:
            for root, directories, filenames in os.walk(self.datapath + '/' + label_directory):
                image_file_paths = [os.path.join(root, filename) for filename in filenames]

                all_image_files = all_image_files + image_file_paths
                labels = labels + [label_directory]*len(image_file_paths)

        images = [self._load_image(image_file) for image_file in all_image_files]

        vectorized_labels = self._vectorize_labels(emotion_index_map, labels)
        self._check_data_not_empty(images)
        return self._load_dataset(np.array(images), np.array(vectorized_labels), emotion_index_map)

    def _load_image(self, image_file):
        image = cv2.imread(image_file)
        cropped_image = self.faceDetector.crop_face(image, False)
        if cropped_image is None:
            return None
        return self._reshape(cropped_image)

    def _validate_arguments(self):
        self._check_directory_arguments()

    def _check_directory_arguments(self):
        """
        Validates arguments for loading from directories, including static image and time series directories.
        """
        if not os.path.isdir(self.datapath):
            raise (NotADirectoryError('Directory does not exist: %s' % self.datapath))
        if self.time_delay:
            if self.time_delay < 1:
                raise ValueError('Time step argument must be greater than 0, but gave: %i' % self.time_delay)
            if not isinstance(self.time_delay, int):
                raise ValueError('Time step argument must be an integer, but gave: %s' % str(self.time_delay))

