from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

K.set_image_dim_ordering('th')


class DataGenerator:
    def __init__(self):
        self.config_augmentation()

    def config_augmentation(self, zca_whitening=False, rotation_range=90, shift_range=0.2, horizontal_flip=True):
        self.data_gen = ImageDataGenerator(featurewise_center=True,
                                           featurewise_std_normalization=True,
                                           zca_whitening=zca_whitening,
                                           rotation_range=rotation_range,
                                           width_shift_range=shift_range,
                                           height_shift_range=shift_range,
                                           horizontal_flip=horizontal_flip)

    def fit(self, images, labels):
        self.images = self._reshape(images)
        self.labels = labels
        self.data_gen.fit(self.images)
        return self

    def get_next_batch(self, batch_size=10):
        for images, labels in self.data_gen.flow(self.images, self.labels, batch_size=batch_size):
            samples, pixels, width, height = images.shape
            images = images.reshape(samples, width, height)
            return images, labels

    def generate(self, batch_size=10):
        return self.data_gen.flow(self.images, self.labels, batch_size=batch_size)

    def _reshape(self, images):
        images = images.astype('float32')
        samples, width, height = images.shape
        return images.reshape(samples, 1, width, height)
