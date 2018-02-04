import unittest

from library.image import *


class SingleFrameTransformationTest(unittest.TestCase):
    def test_should_generate_identity_transform_matrix_for_default_configuration(self):
        image = np.zeros((64, 64, 3))
        generator = ImageDataGenerator(time_delay=None)
        transform_matrix = generator.get_random_transform_matrix(image)
        self.assertTrue(np.array_equal(transform_matrix, np.identity(3)))

    def test_should_generate_non_identity_transformation_matrix(self):
        image = np.zeros((64, 64, 1))
        generator = ImageDataGenerator(rotation_range=0.2)
        transformation_matrix = generator.get_random_transform_matrix(image)
        self.assertFalse(np.array_equal(transformation_matrix, np.identity(3)))

    def test_should_not_transform_when_transform_matrix_is_identity_matrix(self):
        image = np.random.rand(64, 64, 3)
        transform_matrix = np.identity(3)
        transformed_image = apply_transform(image, transform_matrix, channel_axis=2)
        self.assertTrue(np.array_equal(transformed_image, image))

    def test_should_transform_when_non_identity_transform_matrix_is_given(self):
        image = np.random.rand(64, 64, 3)
        transform_matrix = np.random.rand(3, 3)
        transformed_image = apply_transform(image, transform_matrix, channel_axis=2)
        self.assertFalse(np.array_equal(transformed_image, image))


class MultipleFrameTransformationTest(unittest.TestCase):
    def test_should_generate_transformation_matrix_with_time_delay(self):
        image = np.zeros((3, 64, 64, 1))
        generator = ImageDataGenerator(height_shift_range=0.2, time_delay=3)
        transform_matrix = generator.get_random_transform_matrix(image)
        self.assertFalse(np.array_equal(transform_matrix, np.identity(3)))

    def test_should_transform_each_frame_identically_with_time_delay(self):
        sample = np.random.rand(3, 64, 64, 3)
        transform_matrix = np.random.rand(3, 3)
        transformed_sample = apply_transform(sample, transform_matrix, channel_axis=3)
        transformed_frames = [apply_transform(frame, transform_matrix, channel_axis=2) for frame in sample]
        transformed_frames = np.stack(transformed_frames, axis=0)

        self.assertFalse(np.array_equal(transformed_sample, sample))
        self.assertTrue(np.array_equal(transformed_sample, transformed_frames))
