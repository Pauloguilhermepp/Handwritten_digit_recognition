import tkinter as tk
from keras.models import load_model
from PIL import Image
import numpy as np
from preprocessing import adjust_colors, resize_image, adjust_center

# Class to represent the GUI 
class App(tk.Tk):
    def __init__(self, model_path):
        tk.Tk.__init__(self)

        self.x = self.y = 0
        self.model = load_model(model_path)

        # Creating elements
        self.canvas = tk.Canvas(self, width=300, height=300, bg="white", cursor="cross")
        self.label = tk.Label(self, text="Thinking...", font=("Helvetica", 48))
        self.classify_btn = tk.Button(
            self, text="Recognize", command=self.classify_handwriting
        )
        self.button_clear = tk.Button(self, text="Clear", command=self.clear_all)

        # Grid structure
        self.canvas.grid(
            row=0,
            column=0,
            pady=2,
            sticky=tk.W,
        )
        self.label.grid(row=0, column=1, pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)

        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def predict_digit(self, img):
        # Convert rgb to grayscale
        img = adjust_colors(img)

        # Resize image to 28x28 pixels
        img = resize_image(img)

        # Reshaping to support our model input and normalizing
        img = adjust_center(img)
        img = img.reshape(1, 28, 28, 1)
        img = img / 255.0

        # Predicting the image class
        res = self.model.predict([img])[0]
        return np.argmax(res), max(res)

    # Cleaning the screen
    def clear_all(self):
        self.canvas.delete("all")

    def classify_handwriting(self):
        self.canvas.postscript(file="img.eps")
        # use PIL to convert to PNG
        img = Image.open("img.eps")

        digit, acc = self.predict_digit(img)
        self.label.configure(text=str(digit) + ", " + str(int(acc * 100)) + "%")

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        radius = 8

        self.canvas.create_oval(
            self.x - radius,
            self.y - radius,
            self.x + radius,
            self.y + radius,
            fill="black",
        )


def main():
    App("Models/mnist2.h5")
    tk.mainloop()


if __name__ == "__main__":
    main()
