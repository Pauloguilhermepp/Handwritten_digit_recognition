import math
from PIL import ImageOps
from scipy import ndimage
import numpy as np
import cv2


def adjust_colors(img):
    img = img.convert("L")
    img = ImageOps.invert(img)

    return img

def centering_img(img):
    while np.sum(img[0]) == 0:
        img = img[1:]

    while np.sum(img[:, 0]) == 0:
        img = np.delete(img, 0, 1)

    while np.sum(img[-1]) == 0:
        img = img[:-1]

    while np.sum(img[:, -1]) == 0:
        img = np.delete(img, -1, 1)
    
    return img


def resize_image(img):
    img = img.resize((28, 28))
    img = np.array(img)

    img = centering_img(img)

    rows, cols = img.shape

    if rows > cols:
        factor = 20.0 / rows
        rows = 20
        cols = int(round(cols * factor))
        img = cv2.resize(img, (cols, rows))
    else:
        factor = 20.0 / cols
        cols = 20
        rows = int(round(rows * factor))
        img = cv2.resize(img, (cols, rows))

    cols_padding = (
        int(math.ceil((28 - cols) / 2.0)),
        int(math.floor((28 - cols) / 2.0)),
    )
    rows_padding = (
        int(math.ceil((28 - rows) / 2.0)),
        int(math.floor((28 - rows) / 2.0)),
    )
    img = np.lib.pad(img, (rows_padding, cols_padding), "constant")

    return img


def get_best_shift(img):
    cy, cx = ndimage.measurements.center_of_mass(img)

    rows, cols = img.shape
    shiftx = np.round(cols / 2.0 - cx).astype(int)
    shifty = np.round(rows / 2.0 - cy).astype(int)

    return shiftx, shifty


def shift(img, sx, sy):
    rows, cols = img.shape
    M = np.float32([[1, 0, sx], [0, 1, sy]])
    shifted = cv2.warpAffine(img, M, (cols, rows))
    return shifted


def adjust_center(img):
    shiftx, shifty = get_best_shift(img)
    shifted = shift(img, shiftx, shifty)
    img = shifted

    return img
