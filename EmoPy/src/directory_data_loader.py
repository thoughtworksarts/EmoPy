import os, cv2
import numpy as np

from EmoPy.src.data_loader import _DataLoader

class DirectoryDataLoader(_DataLoader):
    """
    DataLoader subclass loads image and label data from directory.

    :param target_emotion_map: Optional dict of target emotion label values/strings and their corresponding label vector index values.
    :param datapath: Location of image dataset.
    :param validation_split: Float percentage of data to use as validation set.
    :param out_channels: Number of image channels.
    :param time_delay: Number of images to load from each time series sample. Parameter must be provided to load time series data and unspecified if using static image data.
    """

    def __init__(self, target_emotion_map=None, datapath=None, validation_split=0.2, out_channels=1, time_delay=None):
        self.datapath = datapath
        self.target_emotion_map = target_emotion_map
        self.out_channels = out_channels
        super().__init__(validation_split, time_delay)

    def load_data(self):
        """
        Loads image and label data from specified directory path.

        :return: Dataset object containing image and label data.
        """
        images = list()
        labels = list()
        emotion_index_map = dict()
        label_directories = [dir for dir in os.listdir(self.datapath) if not dir.startswith('.')]
        for label_directory in label_directories:
            if self.target_emotion_map:
                if label_directory not in self.target_emotion_map.keys():    continue
            self._add_new_label_to_map(label_directory, emotion_index_map)
            label_directory_path = self.datapath + '/' + label_directory

            if self.time_delay:
                self._load_series_for_single_emotion_directory(images, label_directory, label_directory_path, labels)
            else:
                image_files = [image_file for image_file in os.listdir(label_directory_path) if not image_file.startswith('.')]
                self._load_images_from_directory_to_array(image_files, images, label_directory, label_directory_path, labels)

        vectorized_labels = self._vectorize_labels(emotion_index_map, labels)
        self._check_data_not_empty(images)
        return self._load_dataset(np.array(images), np.array(vectorized_labels), emotion_index_map)

    def _load_series_for_single_emotion_directory(self, images, label_directory, label_directory_path, labels):
        series_directories = [series_directory for series_directory in os.listdir(label_directory_path) if not series_directory.startswith('.')]
        for series_directory in series_directories:
            series_directory_path = label_directory_path + '/' + series_directory
            self._check_series_directory_size(series_directory_path)
            new_image_series = list()
            image_files = [image_file for image_file in os.listdir(series_directory_path) if not image_file.startswith('.')]
            self._load_images_from_directory_to_array(image_files, new_image_series, label_directory, series_directory_path, labels)
            new_image_series = self._apply_time_delay_to_series(images, new_image_series)
            images.append(new_image_series)
            labels.append(label_directory)

    def _apply_time_delay_to_series(self, images, new_image_series):
        start_idx = len(new_image_series) - self.time_delay
        end_idx = len(new_image_series)
        return new_image_series[start_idx:end_idx]

    def _load_images_from_directory_to_array(self, image_files, images, label, directory_path, labels):
        for image_file in image_files:
            images.append(self._load_image(image_file, directory_path))
            if not self.time_delay:
                labels.append(label)

    def _add_new_label_to_map(self, label_directory, label_index_map):
        new_label_index = len(label_index_map.keys())
        label_index_map[label_directory] = new_label_index

    def _load_image(self, image_file, directory_path):
        image_file_path = directory_path + '/' + image_file
        image = cv2.imread(image_file_path)
        image = self._reshape(image)
        return image

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


    def _check_series_directory_size(self, series_directory_path):
        image_files = [image_file for image_file in os.listdir(series_directory_path) if not image_file.startswith('.')]
        if len(image_files) < self.time_delay:
            raise ValueError('Time series sample found in path %s does not contain enough images for %s time steps.' % (
                series_directory_path, str(self.time_delay)))
