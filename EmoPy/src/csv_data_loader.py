from EmoPy.src.data_loader import _DataLoader
import csv, cv2
import numpy as np

class CSVDataLoader(_DataLoader):
    """
    DataLoader subclass loads image and label data from csv file.

    :param emotion_map: Dict of target emotion label values and their corresponding label vector index values.
    :param datapath: Location of image dataset.
    :param validation_split: Float percentage of data to use as validation set.
    :param image_dimensions: Dimensions of sample images (height, width).
    :param csv_label_col: Index of label value column in csv.
    :param csv_image_col: Index of image column in csv.
    :param out_channels: Number of image channels.
    """
    def __init__(self, target_emotion_map, datapath=None, validation_split=0.2, image_dimensions=None, csv_label_col=None, csv_image_col=None, out_channels=1):
        self.target_emotion_map = target_emotion_map
        self.datapath = datapath
        self.image_dimensions = image_dimensions
        self.csv_image_col = csv_image_col
        self.csv_label_col = csv_label_col
        self.out_channels = out_channels
        super().__init__(validation_split)

    def load_data(self):
        """
        Loads image and label data from specified csv file path.

        :return: Dataset object containing image and label data.
        """
        print('Extracting training data from csv...')
        images = list()
        labels = list()
        emotion_index_map = dict()
        with open(self.datapath) as csv_file:
            reader = csv.reader(csv_file, delimiter=',', quotechar='"')
            for row in reader:
                label_class = row[self.csv_label_col]
                if label_class not in self.target_emotion_map.keys():
                    continue
                label_class = self.target_emotion_map[label_class]
                if label_class not in emotion_index_map.keys():
                    emotion_index_map[label_class] = len(emotion_index_map.keys())
                labels.append(label_class)

                image = np.asarray([int(pixel) for pixel in row[self.csv_image_col].split(' ')], dtype=np.uint8).reshape(self.image_dimensions)
                image = self._reshape(image)
                images.append(image)

        vectorized_labels = self._vectorize_labels(emotion_index_map, labels)
        self._check_data_not_empty(images)

        return self._load_dataset(np.array(images), np.array(vectorized_labels), emotion_index_map)

    def _validate_arguments(self):
        if self.csv_image_col is None or self.csv_label_col is None:
            raise ValueError(
                'Must provide image and label indices to extract data from csv. csv_label_col and csv_image_col arguments not provided during DataLoader initialization.')

        if self.target_emotion_map is None:
            raise ValueError('Must supply target_labels when loading data from csv.')

        if self.image_dimensions is None:
            raise ValueError('Must provide image dimensions when loading data from csv.')

        # check received valid csv file
        with open(self.datapath) as csv_file:

            # check image and label indices are valid
            reader = csv.reader(csv_file, delimiter=',', quotechar='"')
            num_cols = len(next(reader))
            if self.csv_image_col >= num_cols:
                raise (ValueError('Csv column index for image is out of range: %i' % self.csv_image_col))
            if self.csv_label_col >= num_cols:
                raise (ValueError('Csv column index for label is out of range: %i' % self.csv_label_col))

            # check image dimensions
            pixels = next(reader)[self.csv_image_col].split(' ')
            if len(pixels) != self.image_dimensions[0] * self.image_dimensions[1]:
                raise ValueError('Invalid image dimensions: %s' % str(self.image_dimensions))
