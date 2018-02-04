import unittest

from library.image import *


class TransformationTest(unittest.TestCase):

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

    def test_should_generate_transformation_matrix_with_time_delay(self):
        image = np.zeros((3, 64, 64, 1))
        generator = ImageDataGenerator(rotation_range=0.2)
        transform_matrix = generator.get_random_transform_matrix(image)
        self.assertFalse(np.array_equal(transform_matrix, np.identity(3)))

    def test_should_not_transform_when_transform_matrix_is_identity_matrix(self):
        image = np.ones((64, 64, 3))
        transform_matrix = np.identity(3)
        transformed_image = apply_transform(image, transform_matrix, channel_axis=2)
        self.assertTrue(np.array_equal(transformed_image, image))

    def test_should_transform_when_non_identity_transform_matrix_is_given(self):
        image = np.random.rand(64, 64, 3)
        transform_matrix = np.random.rand(3, 3)
        transformed_image = apply_transform(image, transform_matrix, channel_axis=2)
        self.assertFalse(np.array_equal(transformed_image, image))
