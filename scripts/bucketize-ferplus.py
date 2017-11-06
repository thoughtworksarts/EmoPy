import csv
import os
import shutil
from PIL import Image
import argparse

class Label:
    __emotions = {2: 'neutral',
                  3: 'happiness',
                  4: 'surprise',
                  5: 'sadness',
                  6: 'anger',
                  7: 'disgust',
                  8: 'fear',
                  9: 'contempt',
                  10: 'unknown',
                  11: 'NF'}

    def __init__(self, label_row):
        self.image = label_row[0]
        self.__label_counter = {}
        for col_index in range(2, 12):
            col_emotion = self.__emotions[col_index]
            self.__label_counter[col_emotion] = label_row[col_index]

        self.dominant = max(self.__label_counter, key=self.__label_counter.get)


def prepare_buckets(buckets_root_path, emotions):
    print("Clearing out any existing contents in buckets")
    if os.path.exists(buckets_root_path):
        shutil.rmtree(buckets_root_path)

    os.mkdir(buckets_root_path)
    for emotion in emotions.values():
        os.mkdir(get_bucket_dir(buckets_root_path, emotion))

    print("Created emotion buckets in " + buckets_root_path)

def get_bucket_dir(buckets_root_path, emotion):
    return buckets_root_path + "/" + emotion


def png_to_jpg(source, target):
    target = target.replace('.png', '.jpg')
    image = Image.open(source)
    image.save(target)

def bucketize(buckets_root_path, training_path):
    emotions = {2: 'neutral',  # dupe
                3: 'happiness',
                4: 'surprise',
                5: 'sadness',
                6: 'anger',
                7: 'disgust',
                8: 'fear',
                9: 'contempt',
                10: 'unknown',
                11: 'NF'}
    prepare_buckets(buckets_root_path, emotions)

    training_labels_path = '{}/label.csv'.format(training_path)
    with open(training_labels_path, 'r') as csvfile:
        training_label_rows = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in training_label_rows:
            label = Label(row)
            source = '{}/{}'.format(training_path, label.image)
            target = '{}/{}/{}'.format(buckets_root_path, label.dominant, label.image)
            print('.', end='', flush=True)
            png_to_jpg(source, target)

    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""Turns FERPlus data to a form ready to be used to train
                                                    retrain Inception-V3 on TensorFlow 
                                                    (details: https://www.tensorflow.org/tutorials/image_retraining). 

                                                    ***NOTE***: This assumes you've run generate_training_data.py prior 
                                                    to this to extract *.png files out of the original FER data from Kaggle.
                                                    For more info, follow this: https://github.com/Microsoft/FERPlus#training-data""")
    parser.add_argument("-training",
                        "--training_path",
                        type=str,
                        help="""full path to directory containing FERPlus training data 
                        (a bunch of *.png files and label.csv), e.g. <full_path>/FERPlus/data/FER2013Train""",
                        required=True)
    parser.add_argument("-buckets",
                        "--buckets_root_path",
                        type=str,
                        help="where you'd like the destination buckets to be",
                        required=True)

    args = parser.parse_args()
    bucketize(args.buckets_root_path, args.training_path)