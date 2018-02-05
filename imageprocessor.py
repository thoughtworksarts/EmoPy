import csv, cv2, datetime
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

class ImageProcessor:
    """
    Performs image dataset pre-processing such as resizing, augmenting the dataset, etc.

    :param images: List of raw image samples.
    :param target_dimensions: Final dimensions of training images.
    :param augment_data: If true, will augment data with ImageDataGenerator.
    :param rgb: True if raw images are in rgb.
    :param time_series: True if each sample is a series of images.
    """
    def __init__(self, images, target_dimensions=None, augment_data=False, rgb=False, time_series=False):
        self.images = images
        self.target_dimensions = target_dimensions
        self.augment_data = augment_data
        self.rgb = rgb
        self.time_series = time_series

    def process_training_data(self):
        """
        :return:  List of processed image data.
        """
        start = datetime.datetime.now()

        images = list()
        for raw_image in self.images:
            image = self._resize_image_sample(raw_image)
            if self.rgb:
                image = np.array([image, image, image]).reshape((self.target_dimensions[0], self.target_dimensions[1], 3))
            images.append(image)
        if self.augment_data:
            data_gen = ImageDataGenerator(rotation_range=180)
            data_gen.fit(images)   # TODO: functionality: send data_gen new image set to feature extractor
                                # TODO: functionality: ImDataGen input will be dependent on experimentation results for emotion subsets

        end = datetime.datetime.now()
        print('Training data extraction runtime - ' + str(end-start))

        return np.array(images)

    def _resize_image_sample(self, raw_image):
        """
        :param raw_image:
        :return: Resized image sample.
        """
        if not self.time_series:
            return cv2.resize(raw_image, self.target_dimensions, interpolation=cv2.INTER_LINEAR)
        if self.time_series:
            sample = list()
            for slice in raw_image:
                sample.append(cv2.resize(slice, self.target_dimensions, interpolation=cv2.INTER_LINEAR))
            return sample