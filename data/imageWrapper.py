import cv2
import math
import random

import numpy as np


class ImageWrapper:

    def __init__(self, image, root_directory):
        self.image = cv2.imread(image,1)
        self.root_directory = root_directory
        self.height, self.width, self.channels = image.shape
        self.name = self.root_directory.split('/')[-1]

    def get_name_path(self, process):
        name_list = self.name.split('.')
        name_list.insert(1, process)
        new_name = self.root_directory + '_'.join(name_list[:2]) + '.' + name_list[2]
        return new_name

    def crop_image(self):
        print("Cropping image to 480x480...")
        cropped_image = self.image[0:480, 120:600]
        return cropped_image

    def should_activate(self):
        PROB_THRESHHOLD = 0.5
        random_number = random.uniform(0.1,1.0)
        print("Probability is %f..." % random_number)
        if(random_number > PROB_THRESHHOLD):
            print("Activating...")
            return True
        print("Did not activate...")
        return False

    def flip_image(self):
        print("Flipping...")
        if self.should_activate():
            flipped_image = cv2.flip(self.image, 1)
            image_name = self.get_name_path('flip')
            cv2.imwrite(image_name, flipped_image)

    def rotate_image(self):
        print("Rotating...")
        rows, cols, channels = self.image.shape
        if self.should_activate():
            rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), 90, 1)
            rotated_image = cv2.warpAffine(self.image, rotation_matrix, (cols, rows))
            image_name = self.get_name_path('rotate')
            cv2.imwrite(image_name, rotated_image)

    def add_noise(self):
        print("Adding noise...")
        rgb_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb_img.shape
        print("H: %d\tW: %d\tC:%d" % (height, width, channels))
        noise = np.random.randint(0,50, (height, width))
        jitter = np.zeros_like(rgb_img)
        jitter[:,:,1] = noise

        if self.should_activate(self):
            image_name = self.get_name_path('noisy')
            noisy_image = cv2.add(self.image, jitter)
            combined = np.vstack((self.image[:math.ceil(height/2),:,:], noisy_image[math.ceil(height/2):,:,:]))
            cv2.imwrite(image_name,combined)



