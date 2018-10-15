from keras import backend as K

from EmoPy.library.image import ImageDataGenerator

K.set_image_dim_ordering('th')


class DataGenerator:
    def __init__(self, time_delay=None):
        self.time_delay = time_delay
        self.images = None
        self.labels = None
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
        self._validate(images, labels)
        self.images = images
        self.labels = labels
        self.data_gen.fit(self.images)
        return self

    def get_next_batch(self, batch_size=10, target_dimensions=None):
        self._check_model_has_been_fit()
        for images, labels in self.data_gen.flow(self.images, self.labels, batch_size=batch_size,
                                                 target_dimensions=target_dimensions):
            return images, labels

    def generate(self, target_dimensions=None, batch_size=10):
        self._check_model_has_been_fit()
        return self.data_gen.flow(self.images, self.labels, batch_size=batch_size, target_dimension=target_dimensions)

    def _validate(self, images, labels):
        if len(images) != len(labels):
            raise ValueError("Samples are not labeled properly")
        if images.ndim < 4:
            raise ValueError("Channel Axis should have value")
        if self.time_delay:
            if images.ndim != 5:
                raise ValueError("Time_delay parameter was set but Images say otherwise")
        if images.ndim == 5 and images.shape[1] != self.time_delay:
            raise ValueError("Images have time axis length {given} "
                             "but time_delay parameter was set to {set}"
                             .format(given=images.shape[1], set=self.time_delay))

    def _check_model_has_been_fit(self):
        if self.images is None or self.labels is None:
            raise ValueError("Model is not fit to any data set yet")
