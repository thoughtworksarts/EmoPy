
class Dataset():

    def __init__(self, train_images, test_images, train_labels, test_labels, emotion_index_map, time_delay=None):
        self._train_images = train_images
        self._test_images = test_images
        self._train_labels = train_labels
        self._test_labels = test_labels
        self._emotion_index_map = emotion_index_map
        self._time_delay = time_delay

    def get_training_data(self):
        return self._train_images, self._train_labels

    def get_test_data(self):
        return self._test_images, self._test_labels

    def get_emotion_index_map(self):
        return self._emotion_index_map

    def get_time_delay(self):
        return self._time_delay

    def num_test_images(self):
        return len(self._test_images)

    def num_train_images(self):
        return len(self._train_images)

    def num_images(self):
        return self.num_train_images() + self.num_test_images()

    def print_data_details(self):
        print('\nDATASET DETAILS')
        print('%d image samples' % (self.num_images()))
        print('%d training samples' % self.num_train_images())
        print('%d test samples\n' % self.num_test_images())