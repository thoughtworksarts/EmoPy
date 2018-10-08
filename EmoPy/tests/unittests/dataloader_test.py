import sys

import pytest

#from EmoPy.src.data_loader import DataLoader

from EmoPy.src.csv_data_loader import CSVDataLoader
from EmoPy.src.directory_data_loader import DirectoryDataLoader

from pkg_resources import resource_filename

valid_csv_file_path = resource_filename('EmoPy.examples','image_data/sample.csv')
valid_image_dimensions = (48, 48)
csv_label_col = 0
csv_image_col = 1
valid_target_labels = [0, 1, 2, 3, 4, 5, 6]
## replaces target emotion map, but is a list, not a map.
## what is target emotion map meant to look like?


def test_load_csv_data():
    invalid_csv_file_path = 'invalid_csv_file_path'
    channels = 1
    invalid_image_dimensions = (50, 77)
    invalid_target_labels = [8, 9, 10]

    # should raise error when not given csv column indices for images and labels
    with pytest.raises(ValueError):
        CSVDataLoader(target_emotion_map=valid_target_labels, datapath=valid_csv_file_path,
                   image_dimensions=valid_image_dimensions, csv_label_col=csv_label_col)


    # should raise error when given invalid csv file path
    with pytest.raises(FileNotFoundError):
        CSVDataLoader(target_emotion_map=valid_target_labels, datapath=invalid_csv_file_path,
                   image_dimensions=valid_image_dimensions, csv_label_col=csv_label_col, csv_image_col=csv_image_col)

    # should raise error when given invalid csv column indices
    with pytest.raises(ValueError):
        CSVDataLoader(target_emotion_map=valid_target_labels, datapath=valid_csv_file_path,
                   image_dimensions=valid_image_dimensions, csv_label_col=csv_label_col, csv_image_col=10)

    # should raise error when given empty target_labels list
    with pytest.raises(ValueError):
        CSVDataLoader(datapath=valid_csv_file_path, image_dimensions=valid_image_dimensions,
                   csv_label_col=csv_label_col, csv_image_col=csv_image_col,target_emotion_map=None)

    # should raise error when not given image dimensions
    with pytest.raises(ValueError):
        CSVDataLoader(target_emotion_map=valid_target_labels, datapath=valid_csv_file_path,
                   csv_label_col=csv_label_col, csv_image_col=csv_image_col)

    # should raise error when given invalid image dimensions
    with pytest.raises(ValueError):
        CSVDataLoader(target_emotion_map=valid_target_labels, datapath=valid_csv_file_path,
                   image_dimensions=invalid_image_dimensions, csv_label_col=csv_label_col, csv_image_col=csv_image_col)
'''
    # should raise error if no image samples found in csv file
    with pytest.raises(AssertionError):
        data_loader = CSVDataLoader(target_emotion_map=invalid_target_labels, datapath=valid_csv_file_path,
                                 image_dimensions=valid_image_dimensions, csv_label_col=csv_label_col,
                                 csv_image_col=csv_image_col)
        data_loader.load_data()

    data_loader = DataLoader(from_csv=True, target_labels=valid_target_labels, datapath=valid_csv_file_path,
                             image_dimensions=valid_image_dimensions, csv_label_col=csv_label_col,
                             csv_image_col=csv_image_col)
    images, labels = data_loader.load_data()
    # should return non-empty image and label arrays when given valid arguments
    assert len(images) > 0 and len(labels) > 0
    # should return same number of labels and images when given valid arguments
    assert len(images) == len(labels)
    # should reshape the images to given valid image_dimensions
    assert list(images.shape[1:]) == list(valid_image_dimensions) + [channels]
'''

def test_load_directory_data():

    invalid_directory_path = 'invalid_directory_path'
    valid_dummy_directory = resource_filename('EmoPy','tests/unittests/resources/dummy_data_directory')
    empty_dummy_directory = resource_filename('EmoPy','tests/unittests/resources/dummy_empty_data_directory')
    channels = 1

    # should raise error when receives an invalid directory path
    with pytest.raises(NotADirectoryError):
        DirectoryDataLoader(datapath=invalid_directory_path)

    # should raise error when tries to load empty directory
    data_loader = DirectoryDataLoader(datapath=empty_dummy_directory)
    with pytest.raises(AssertionError):
        data_loader.load_data()
'''
    # should assign an image's parent directory name as its label
    data_loader = DataLoader(from_csv=False, datapath=valid_dummy_directory)
    images, labels, label_index_map = data_loader.load_data()
    label_count = len(label_index_map.keys())
    label = [0] * label_count
    label[label_index_map['happiness']] = 1
    assert label == labels[0]

    data_loader = DataLoader(from_csv=False, datapath=valid_dummy_directory)
    images, labels, label_index_map = data_loader.load_data()
    # should return non-empty image and label arrays when given valid arguments
    assert len(images) > 0 and len(labels) > 0
    # should return same number of labels and images when given valid arguments
    assert len(images) == len(labels)
    # should reshape image to contain channel_axis in channel_last format
    assert images.shape[-1] == channels
    '''


def test_load_time_series_directory_data():
    '''
    invalid_directory_path = 'invalid_directory_path'
    valid_dummy_directory = resource_filename('EmoPy','tests/unittests/resources/dummy_time_series_data_directory')
    empty_dummy_directory = resource_filename('EmoPy','tests/unittests/resources/dummy_empty_data_directory')
    valid_time_steps = 4
    channels = 1

    # should raise error when receives an invalid directory path
    with pytest.raises(NotADirectoryError):
        DataLoader(from_csv=False, datapath=invalid_directory_path, time_steps=4)

    # should raise error when tries to load empty directory
    data_loader = DataLoader(from_csv=False, datapath=empty_dummy_directory, time_steps=4)
    with pytest.raises(AssertionError):
        data_loader.load_data()

    # should raise error when given time_step argument that is less than 1
    with pytest.raises(ValueError):
        DataLoader(from_csv=False, datapath=valid_dummy_directory, time_steps=-4)

    # should raise error when given time_step argument that not an integer
    with pytest.raises(ValueError):
        DataLoader(from_csv=False, datapath=valid_dummy_directory, time_steps=4.7)

    # should raise error when tries to load time series sample
    # containing a quantity of images less than the time_steps argument
    with pytest.raises(ValueError):
        data_loader = DataLoader(from_csv=False, datapath=valid_dummy_directory, time_steps=10)
        data_loader.load_data()

    # should assign an image's parent directory name as its label
    data_loader = DataLoader(from_csv=False, datapath=valid_dummy_directory, time_steps=valid_time_steps)
    samples, labels, label_index_map = data_loader.load_data()
    label_count = len(label_index_map.keys())
    label = [0] * label_count
    label[label_index_map['happiness']] = 1
    assert label == labels[0]

    data_loader = DataLoader(from_csv=False, datapath=valid_dummy_directory, time_steps=valid_time_steps)
    samples, labels, label_index_map = data_loader.load_data()
    # should return non-empty image and label arrays when given valid arguments
    assert len(samples) > 0 and len(labels) > 0
    # should return same number of labels and images when given valid arguments
    assert len(samples) == len(labels)
    # should reshape image to contain channel_axis in channel_last format
    assert samples.shape[1] == valid_time_steps
    # should reshape image to contain channel_axis in channel_last format
    assert samples.shape[-1] == channels
    '''


def test_should_generate_images_based_on_out_channels_parameter():
    '''
    with pytest.raises(ValueError) as e:
        DataLoader(out_channels=0)
    assert "Out put channel should be either 3(RGB) or 1(Grey) but got 0" == str(e.value)

    # Should generate images with single channel
    channels = 1
    data_loader = DataLoader(from_csv=True, target_labels=valid_target_labels, datapath=valid_csv_file_path,
                             image_dimensions=valid_image_dimensions, csv_label_col=csv_label_col,
                             csv_image_col=csv_image_col, out_channels=channels)
    images, labels = data_loader.load_data()
    assert list(images.shape[1:]) == list(valid_image_dimensions) + [channels]

    # Should generate images with 3 channel
    channels = 3
    data_loader = DataLoader(from_csv=True, target_labels=valid_target_labels, datapath=valid_csv_file_path,
                             image_dimensions=valid_image_dimensions, csv_label_col=csv_label_col,
                             csv_image_col=csv_image_col, out_channels=channels)
    images, labels = data_loader.load_data()
    assert list(images.shape[1:]) == list(valid_image_dimensions) + [channels]
    '''

if __name__ == '__main__':
    pytest.main([__file__])
