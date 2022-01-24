from keras.models import load_model
from tkinter import *
import tkinter as tk
from PIL import ImageOps, Image
import numpy as np
from os import listdir

model = load_model('mnist2.h5')

def predict_digit(img):
    # resize image to 28x28 pixels
    img = img.resize((28,28))
    # convert rgb to grayscale
    img = img.convert('L')
    img = ImageOps.invert(img)
    img.show()
    img = np.array(img)
    #reshaping to support our model input and normalizing
    img = img.reshape(1,28,28,1)
    img = img/255.0
    #predicting the class
    res = model.predict([img])[0]
    return np.argmax(res), max(res)

for i in listdir('digits'):
    correct_digit = i.split('.')[0]
    print(f"Number {correct_digit}")

    digit, acc = predict_digit(Image.open('digits/' + i, mode='r'))
    print(f"This digit is {digit} and the accuracy is {acc}!\n")

    input()