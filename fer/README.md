# FERPython

This repository contains deep neural net classes designed to solve Facial Expression Recognition (FER) problems. These classes can be trained on user-supplied datasets containing images of human faces at peak emotions.

Our goal is create deep neural net classes that are easily adaptable to perform well with various datasets, lighting conditions, and subsets of the classic 6 emotion labels (i.e. happiness, anger, fear, surprise, calm/neutral, sadness)

We want the help of the open-source community in experimenting with these neural networks to improve performance. Additionally, we want to include an easy-to-use top-layer class (FERModel) that can be used to train and predict by simply supplying an image dataset.

## Installation

You will need to install Python 3.6.3. To install all additional dependencies run the following command:

```
pip install -r requirements.txt
```

## Usage

The fastest way to get started is by using the FERModel class, which chooses and initializes a neural net for you based on the set of emotions you want to classify. Simply supply a list of target emotions and an image training set. The training set can be given as a numpy list of images and a numpy list of corresponding labels or a local path to a csv file. If supplying a csv file, you must also provide the dimensions of the image contained in the csv and the image and label column indices.

Example using a csv file:

```python
from fermodel import FERModel

target_emotions = ['anger', 'fear', 'calm', 'sad', 'happy', 'surprise']
csv_file_path = "<your local csv file path>"
model = FERModel(target_emotions, csv_data_path=csv_file_path, raw_dimensions=(48,48), csv_image_col=1, csv_label_col=0, verbose=True)
model.train()
```

## Contributing

1. Fork it!
2. Create your feature branch: `git checkout -b my-new-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin my-new-feature`
5. Submit a pull request :D

## Credits

TODO: Write credits

## License

TODO: Write license