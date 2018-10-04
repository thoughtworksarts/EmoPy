import unittest

import numpy as np

from EmoPy.src.data_generator import DataGenerator


class DataGeneratorTest(unittest.TestCase):
    def test_should_produce_augmented_samples_given_batch_size(self):
        images = np.random.rand(20, 64, 64, 1)
        labels = np.random.rand(20)
        generator = DataGenerator().fit(images, labels)
        batch_size = 10
        batch, _ = generator.get_next_batch(batch_size)
        self.assertEqual(batch.shape[0], batch_size)
        self.assertEqual(images.shape[1:], batch.shape[1:])

    def test_should_resize_images_to_given_target_dimension(self):
        images = np.random.rand(20, 64, 64, 3)
        labels = np.random.rand(20)
        generator = DataGenerator().fit(images, labels)
        batch, _ = generator.get_next_batch(10, target_dimensions=(28, 28))
        self.assertEqual(batch.shape, (10, 28, 28, 3))

    def test_should_raise_error_when_labels_and_samples_are_mis_matched(self):
        images = np.random.rand(20, 64, 64)
        with self.assertRaises(ValueError) as e:
            DataGenerator().fit(images, [1])
        self.assertEqual("Samples are not labeled properly", str(e.exception))

    def test_should_raise_error_when_channel_axis_is_not_present(self):
        images = np.random.rand(20, 64, 64)
        labels = np.random.rand(20)
        with self.assertRaises(ValueError) as e:
            DataGenerator().fit(images, labels)
        self.assertEqual("Channel Axis should have vale", str(e.exception))

    def test_should_raise_error_when_time_delay_parameter_is_set_and_input_is_simple_images(self):
        images = np.random.rand(10, 64, 64, 3)
        labels = np.random.rand(10)
        with self.assertRaises(ValueError) as e:
            DataGenerator(time_delay=4).fit(images, labels)
        self.assertEqual("Time_delay parameter was set but Images say otherwise", str(e.exception))

    def test_should_raise_error_when_time_delay_was_not_set_and_input_is_time_series(self):
        images = np.random.rand(10, 2, 64, 64, 3)
        labels = np.random.rand(10)
        with self.assertRaises(ValueError) as e:
            DataGenerator().fit(images, labels)
        self.assertEqual("Images have time axis length 2 but time_delay parameter was set to None", str(e.exception))

    def test_should_raise_error_if_time_delay_is_not_matching_input_time_axis(self):
        images = np.random.rand(10, 4, 64, 64, 3)
        labels = np.random.rand(10)
        with self.assertRaises(ValueError) as e:
            DataGenerator(time_delay=5).fit(images, labels)
        self.assertEqual("Images have time axis length 4 but time_delay parameter was set to 5", str(e.exception))

    def test_should_raise_error_if_model_not_fit_to_data_yet(self):
        with self.assertRaises(ValueError) as a:
            DataGenerator(time_delay=5).get_next_batch()
        with self.assertRaises(ValueError) as b:
            DataGenerator(time_delay=5).generate()

        self.assertEqual("Model is not fit to any data set yet", str(a.exception))
        self.assertEqual("Model is not fit to any data set yet", str(b.exception))
