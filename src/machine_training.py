# Simple CNN for the MNIST Dataset
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils


def load_data_base():
    # load data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # reshape to be [samples][width][height][channels]
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype("float32")
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype("float32")

    # normalize inputs from 0-255 to 0-1
    x_train = x_train / 255
    x_test = x_test / 255

    return [(x_train, y_train), (x_test, y_test)]


# define a simple CNN model
def baseline_model(num_classes):
    # create model
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation="relu"))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(num_classes, activation="softmax"))

    # Compile model
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    return model


def main():
    (x_train, y_train), (x_test, y_test) = load_data_base()

    # one hot encode outputs
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]

    # build the model
    model = baseline_model(num_classes)

    # Set input for training model
    epochs_number = 10
    batch_number = 200

    # Fit the model
    model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=epochs_number,
        batch_size=batch_number,
    )

    # Final evaluation of the model
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("CNN Error: %.2f%%" % (100 - scores[1] * 100))

    # Saving model
    model.save("mnist.h5")


if __name__ == "__main__":
    main()
