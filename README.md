# Handwritten_digit_recognition

## What is it?
This is a basic project using Machine Learning to identify handwritten numerical
digits, using the MNIST database.

The main idea of this code is to allow the user to write on the computer a digit
with a mouse and then see the program trying to guess (almost) always the digit.

There are three main modules in this repository: 
* the machine training, where it is built a model with the MNIST database;
* the pre-processing image module, so the image becomes the most next to the 
image in the used database
* the GUI, where the user can draw their digits.

## How to use it?
This code is open-source, so you can see every code we wrote and edit it in any 
way you desire. In the current state, you can use this code without any big 
modifications, and here are two ways you can do it:

### Just executing it
Just execute the file gui_digit_recognizer.py and see the magic happening in front
of your own eyes.
The program will open a canvas where you can draw any single-digit number and it
will try to guess it (getting it right almost always). 

### Building your model
* To train your model, you will execute the machine_training.py file in the 
src folder, you can edit any value you want. One variable we recommend you try 
to change is the epoch variable, so you can control how many times the training 
will occur. Generally, how much more times, the more accuracy you get.

* Now, you just need to execute the gui_digit_recognizer.py file, just like in 
the previous section.

## Sources
* [Base of the GUI interface.](https://data-flair.training/blogs/python-deep-learning-project-handwritten-digit-recognition/)
* [The model was based in this site.](https://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/)
* [Important ideas related to preprocessing were developed thanks to this site.](https://medium.com/@o.kroeger/tensorflow-mnist-and-your-own-handwritten-digits-4d1cd32bbab4)