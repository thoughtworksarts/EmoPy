import pytest
import sys
sys.path.append('../../')
from dataloader import DataLoader

def test_load_csv_data():

    valid_csv_file_path =  '../../examples/image_data/sample.csv'
    invalid_csv_file_path = 'invalid_csv_file_path'
    valid_image_dimensions = (48,48)
    invalid_image_dimensions = (50,77)
    csv_label_col = 0
    csv_image_col = 1
    valid_target_labels = [0,1,2,3,4,5,6]
    invalid_target_labels = [8,9,10]

    # should receive csv column indices for images and labels
    with pytest.raises(ValueError):
       DataLoader(from_csv=True, target_labels=valid_target_labels, datapath=valid_csv_file_path, image_dimensions=valid_image_dimensions, csv_label_col=csv_label_col)

    # should receive valid csv file path
    with pytest.raises(FileNotFoundError):
        DataLoader(from_csv=True, target_labels=valid_target_labels, datapath=invalid_csv_file_path, image_dimensions=valid_image_dimensions, csv_label_col=csv_label_col, csv_image_col=csv_image_col)

    # should receive valid csv column indices
    with pytest.raises(ValueError):
        DataLoader(from_csv=True, target_labels=valid_target_labels, datapath=valid_csv_file_path, image_dimensions=valid_image_dimensions, csv_label_col=csv_label_col, csv_image_col=10)

    # should receive non-empty target_labels list
    with pytest.raises(ValueError):
        DataLoader(from_csv=True, datapath=valid_csv_file_path, image_dimensions=valid_image_dimensions, csv_label_col=csv_label_col, csv_image_col=csv_image_col)

    # should receive image dimensions
    with pytest.raises(ValueError):
        DataLoader(from_csv=True, target_labels=valid_target_labels, datapath=valid_csv_file_path, csv_label_col=csv_label_col, csv_image_col=csv_image_col)

    # should receive valid image dimensions
    with pytest.raises(ValueError):
        DataLoader(from_csv=True, target_labels=valid_target_labels, datapath=valid_csv_file_path, image_dimensions=invalid_image_dimensions, csv_label_col=csv_label_col, csv_image_col=csv_image_col)

    # should raise error if no image samples found in csv file
    with pytest.raises(AssertionError):
        data_loader = DataLoader(from_csv=True, target_labels=invalid_target_labels, datapath=valid_csv_file_path, image_dimensions=valid_image_dimensions, csv_label_col=csv_label_col, csv_image_col=csv_image_col)
        data_loader.get_data()

    data_loader = DataLoader(from_csv=True, target_labels=valid_target_labels, datapath=valid_csv_file_path, image_dimensions=valid_image_dimensions, csv_label_col=csv_label_col, csv_image_col=csv_image_col)
    images, labels = data_loader.get_data()
    # should return non-empty image and label arrays
    assert len(images) > 0 and len(labels) > 0
    # should return same number of labels and images
    assert len(images) == len(labels)

def test_load_directory_data():

    valid_directory_path =  '../../examples/image_data/sample_image_directory.csv'
    invalid_directory_path = 'invalid_directory_path'
    dummy_datapath = '../resources/dummy_data_directory'

    # should receive a valid directory path
    with pytest.raises(FileNotFoundError):
        DataLoader(from_csv=False, datapath=invalid_directory_path)

    # should assign an image's parent directory name as its label
    data_loader = DataLoader(from_csv=False, datapath=dummy_datapath)
    images, labels, label_map = data_loader.get_data()
    label_count = len(label_map.keys())
    label = [0]*label_count
    label[label_map['happiness']] = 1
    assert label == labels[0]

if __name__ == '__main__':
    pytest.main([__file__])