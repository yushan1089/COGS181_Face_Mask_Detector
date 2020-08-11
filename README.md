# Face Mask Detector (COGS 181 Final Project)

This is my deep learning course (COGS 181) final project. The idea was to develop a model using deep learning knowledge to help fight against COVID-19. Then I and my classmate Jiemin Tang developed this Face Mask Detector. 

## Documentation and presentation of this project

The paper of this project : https://yushan1089.github.io/file/181.pdf

The presentation ppt of this project: https://yushan1089.github.io/file/181.pptx


## Requirement

There are several things need to be installed to run the code of this project.

The `TensorFlow` module is required, due to the difference of installation between windows and macOS, I would simply refer a tutorial here: https://www.tensorflow.org/install

Also need `OpenCV` , `keras`, `imutils` package. Those could be done through `pip install`. 

Webcam or other camera are preferred when using the code.

## Instructions
This is a python project using Tensorflow and keras. If you want to use this project there are several ways to check out.

1. Directly using the trained model to test mask or not. Then simply run the  following code

> python LiveMaskWebcam_new.py

2. Wish to trained your own Mask Detector Model then feel free to run the two notebooks in the directory.

    The difference between CNN design and VGG-16 model design for the model is discussed in the paper above. Feel Free to check out.

## Finally, wish everyone stay well, stay safe.