Face Classification
===================

The project aims to recognizes and class faces in pictures by using opencv to detect faces and a keras CNN to recognize them.

----------

## Installation

	git clone https://github.com/luickk/cnn-face-recognition

## Training

Install listed dependiencies


	mkdir data
	mkdir data/train
	mkdir data/valid
	mkdir data/trainedModels

Add your data (data/class/imgFiles). Used dataset: https://www.kaggle.com/dansbecker/5-celebrity-faces-dataset


	python3 train.py

## Testing

Fill in trained model uid and test-img path beforehand

	python predict.py


Dependencies
-------------------

> - numpy
> - Keras
> - cv2
> - Pillow

Results
-------------------

![img_classified](media/2.png)

Original:

![img_org](media/1.jpg)
