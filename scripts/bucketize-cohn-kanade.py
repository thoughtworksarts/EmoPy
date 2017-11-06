import os
import itertools
import shutil
import math
from PIL import Image

def prepare_buckets(buckets_root_path, emotions):
    # start with a clean slate
    if os.path.exists(buckets_root_path):
        shutil.rmtree(buckets_root_path)

    os.mkdir(buckets_root_path)
    for emotion in emotions.values():
        os.mkdir(get_bucket_dir(buckets_root_path, emotion))

    print("Created emotion buckets in " + buckets_root_path)

def get_bucket_dir(buckets_root_path, emotion):
    return buckets_root_path + "/" + emotion

def is_label(path):
    if len(path) == 0:
        return False

    return ".txt" in path[0]

def get_label_paths(cohn_kanade_emotion_path):
    return itertools.filterfalse(lambda path: not is_label(path[2]), os.walk(cohn_kanade_emotion_path))

def corresponding_images_dir(cohn_kanade_emotion_path, label_file):
    components = label_file.split("_")
    subject = components[0]
    sequence = components[1]

    return cohn_kanade_emotion_path + "/" + subject + "/" + sequence


def get_emotion(emotions, label_path):
    label_full_path = label_path[0] + "/" + label_path[2][0]
    f = open(label_full_path, 'r')
    label_value = int(float(f.read().strip()))
    return emotions[label_value]


def png_to_jpg(source, target):
    target = target.replace('.png', '.jpg')
    image = Image.open(source)
    image.save(target)

def bucketize(cohn_kanade_image_path, cohn_kanade_emotion_path, buckets_root_path):
    emotions = {0: 'neutral', 1: 'anger', 2: 'contempt', 3: 'disgust', 4: 'fear', 5: 'happy', 6: 'sad', 7: 'surprise'}
    prepare_buckets(buckets_root_path, emotions)

    label_paths = get_label_paths(cohn_kanade_emotion_path)

    for label_path in label_paths:
        label_file = label_path[2][0]
        image_files = os.listdir(corresponding_images_dir(cohn_kanade_image_path, label_file))
        last_half_image_files = image_files[math.floor(len(image_files)/2):]

        bucket_dir = get_bucket_dir(buckets_root_path, get_emotion(emotions, label_path))

        for image_file in last_half_image_files:
            source = corresponding_images_dir(cohn_kanade_image_path, label_file) + "/" + image_file
            target = bucket_dir + "/" + image_file
            print("Putting " + source + " to " + target)
            png_to_jpg(source, target)

    print("Done!")


cohn_kanade_image_path = '/Users/stania/Work/TWNY/karen-palmer/data/cohn-kanade/cohn-kanade-images'
cohn_kanade_emotion_path = '/Users/stania/Work/TWNY/karen-palmer/data/cohn-kanade/Emotion'
buckets_root_path = '/Users/stania/Work/TWNY/karen-palmer/data/cohn-kanade/tensorflow-training-buckets'

bucketize(cohn_kanade_image_path, cohn_kanade_emotion_path, buckets_root_path)
