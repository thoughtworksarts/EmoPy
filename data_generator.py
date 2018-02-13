from keras import backend as K

from library.image import ImageDataGenerator

K.set_image_dim_ordering('th')


class DataGenerator:
    def __init__(self, time_delay=None):
        self.config_augmentation(time_delay=time_delay)

    def config_augmentation(self, zca_whitening=False, rotation_angle=90, shift_range=0.2, horizontal_flip=True,
                            time_delay=None):
        self.data_gen = ImageDataGenerator(featurewise_center=True,
                                           featurewise_std_normalization=True,
                                           zca_whitening=zca_whitening,
                                           rotation_angle=rotation_angle,
                                           width_shift_range=shift_range,
                                           height_shift_range=shift_range,
                                           horizontal_flip=horizontal_flip,
                                           time_delay=time_delay)
        return self

    def fit(self, images, labels):
        self.images = images
        self.labels = labels
        self.data_gen.fit(self.images)
        return self

    def get_next_batch(self, batch_size=10):
        for images, labels in self.data_gen.flow(self.images, self.labels, batch_size=batch_size):
            samples, pixels, width, height = images.shape
            images = images.reshape(samples, width, height)
            return images, labels

    def generate(self, target_dimensions=None, batch_size=10):
        return self.data_gen.flow(self.images, self.labels, batch_size=batch_size, target_dimension=target_dimensions)
