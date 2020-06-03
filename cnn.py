import os

from keras.callbacks import TensorBoard
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
from keras.utils.np_utils import to_categorical
import random, shutil
from keras.models import Sequential
from keras.layers import Dropout, Conv2D, Flatten, Dense, MaxPooling2D, BatchNormalization
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator


class ConvModel:
    BS = 32
    TS = (24, 24)
    train_datagen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
                                       horizontal_flip=True,
                                       zoom_range=0.2, rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=(24, 24, 1), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), input_shape=(24, 24, 1), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), input_shape=(24, 24, 1), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(1024, activation="relu"))

    model.add(Dense(512, activation="relu"))

    model.add(Dense(2, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

    model.summary()

    train_gen = train_datagen.flow_from_directory("data/train", target_size=TS, shuffle=True, color_mode="grayscale",
                                                  class_mode="categorical", batch_size=BS)

    test_gen = test_datagen.flow_from_directory("data/test", target_size=TS, shuffle=True, color_mode="grayscale",
                                                class_mode="categorical", batch_size=BS)

   # tb = TensorBoard(log_dir="DrowsyDetect/logs")
    model_history = model.fit_generator(train_gen, epochs=15, validation_data=test_gen)

    loss, acc = model.evaluate_generator(test_gen, len(test_gen.classes))
    print("Loss = ", loss)
    print("Accuracy = ", acc)

    model.save('models/newcnn.h5', overwrite=True)


ConvModel()

# img = image.load_img('data/test/Closed/X-130.jpg', target_size=(24, 24))
# image_pred = image.img_to_array(img)
# image_pred = np.expand_dims(image_pred, axis=0)
# image_pred /= 255.
# plt.imshow(image_pred[0])
# plt.show()
# print(image_pred.shape)
