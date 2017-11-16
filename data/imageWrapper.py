import cv2
import math
import os
import random

import numpy as np

from PIL import Image, ImageEnhance

class ImageWrapper:

    def __init__(self, image, root_directory):
        self.image = cv2.imread(image,1)
        self.root_directory = root_directory
        self.height, self.width, self.channels = self.image.shape
        self.name = image.split('/')[-1]

    def get_name_path(self, process):
        name_list = self.name.split('.')
        name_list.insert(1, process)
        process_directory = self.root_directory + "/" + process
        new_name =  process_directory + "/" + '_'.join(name_list[:2]) + '.' + name_list[2]
        if not os.path.exists(process_directory):
            os.makedirs(process_directory)
        return new_name

    def get_crop_measurements(self, height, width):
        if self.width > width:
            width_difference = self.width - width
            width_adjustment = width_difference/2
        else:
            width_adjustment = 0

        if self.height > height:
            height_difference = self.height - height
            height_adjustment = height_difference/2
        else:
            height_adjustment = 0

        return int(width_adjustment), int(height_adjustment)

    def crop(self, height, width):
        print("Cropping image to %dx%d..." % (height, width))
        width_adjustment, height_adjustment = self.get_crop_measurements(height, width)
        width_left = 0 + width_adjustment
        width_right = self.width - width_adjustment
        height_top = 0 + height_adjustment
        height_bottom = self.height - height_adjustment
        cropped_image = self.image[height_top:height_bottom, width_left:width_right]
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

    def flip(self):
        print("Flipping...")
        flipped_image = cv2.flip(self.image, 1)
        image_name = self.get_name_path('flip')
        cv2.imwrite(image_name, flipped_image)

    def rotate(self, degrees_rotation=90):
        print("Rotating...")
        rows, cols, channels = self.image.shape
        rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), degrees_rotation, 1)
        rotated_image = cv2.warpAffine(self.image, rotation_matrix, (cols, rows))
        image_name = self.get_name_path('rotate')
        print(image_name)
        cv2.imwrite(image_name, rotated_image)

    def add_noise(self):
        print("Adding noise...")
        rgb_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb_img.shape
        noise = np.random.randint(0,50, (height, width))
        jitter = np.zeros_like(rgb_img)
        jitter[:,:,1] = noise
        image_name = self.get_name_path('noisy')
        noisy_image = cv2.add(self.image, jitter)
        combined = np.vstack((self.image[:math.ceil(height/2),:,:], noisy_image[math.ceil(height/2):,:,:]))
        print(image_name)
        cv2.imwrite(image_name,combined)

    def brighten(self, max_brighten=13):
        print("Brightening...")
        image_for_brightening = Image.fromarray(self.image)
        enhancer = ImageEnhance.Brightness(image_for_brightening)
        factor = random.randint(0, max_brighten)/4.0
        brightened = enhancer.enhance(factor)
        image_name = self.get_name_path('brighten')
        print(type(brightened))
        cv2.imwrite(image_name,brightened)




