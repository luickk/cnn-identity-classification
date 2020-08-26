Face Identity Classification
===================

The project aims to recognizes and identify faces in pictures by using opencv to detect faces and a keras CNN to identify them.

----------

## Installation

	git clone https://github.com/luickk/cnn-face-recognition

## Training

Install listed dependiencies

> - numpy
> - Keras
> - cv2
> - Pillow

Then create the data dir

	mkdir data

Add your data to `data/data/class/imgFiles` <br>
Used dataset: https://www.kaggle.com/dansbecker/5-celebrity-faces-dataset

	python3 train.py

## Testing

Fill in trained model uid and test-img path beforehand

	python predict.py


Results
-------------------

![img_classified](media/2.png)

Original:

![img_org](media/1.jpg)
