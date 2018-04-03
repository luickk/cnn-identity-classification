Face Recognition and detection
===================

The python program detects and recognizes faces by using opencv to detect faces and a in Keras built CNN to recognize them.

----------

#### <i class="icon-down-big"></i> Installation

	> - Clone Repository
	> - Install Dependencies

#### <i class="icon-ccw"></i> Training
  Change training dir, default empty. Put your faces grouped in folders with appropriate class name here!<br>
  Models are saved in **pretrained_models/** with model id as folder name <br>

	> - python main.py -m train

#### <i class="icon-right-big"></i> Testing

	> - python main.py -u <model ID from trained model here> -i <file name> -m label



Dependencies
-------------------

> - numpy
> - Keras
> - cv2
> - Pillow

Example
-------------------
As you can see the face was recognized and assigned to the respective class.
![img_classified](https://image.ibb.co/koLkbc/Unbenannt.png)

Original:

![img_org](http://cdn1.spiegel.de/images/image-1015094-860_poster_16x9-hejm-1015094.jpg)
