Face Classification
===================

The project aims to recognizes and class faces in pictures by using opencv to detect faces and a keras CNN to recognize them.

----------

## Installation

	git clone https://github.com/luickk/cnn-face-recognition

## Training

1. Install listed dependiencies

2.

	mkdir data
	mkdir data/train
	mkdir data/valid
	mkdir data/trainedModels

3. Add your data (data/class/imgFiles)

4.	

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
