from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization


def paper_architecture(num_classes, input_shape, batch_normalisation=False):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(4, 4), strides=(2, 2), activation='relu', input_shape=input_shape))
    if batch_normalisation:
        model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(4, 4), strides=(2, 2)))
    model.add(Conv2D(64, kernel_size=(4, 4), strides=(2, 2), activation='relu', input_shape=input_shape))
    if batch_normalisation:
        model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(4, 4), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(10 * num_classes, activation='relu'))
    if batch_normalisation:
        model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(5 * num_classes, activation='relu'))
    if batch_normalisation:
        model.add(BatchNormalization())
    model.add(Dense(num_classes, activation='softmax'))
    return model
