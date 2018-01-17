import keras 
from utils import get_data, serialize_data, next_batch
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

root_dir = "/home/anne/src/dog_identification/"  
train_zip = root_dir + "data/train.zip"
valid_zip = root_dir + "data/valid.zip"
training_filename = root_dir + "data/train.p"
validation_filename = root_dir + "data/valid.p"
labels_filename = root_dir + "data/labels.csv.zip"

num_validation = 0.3
batch_size = 250
epochs = 1000
n_breeds = 10

nwidth = 60
nheight = 60


def train():
    serialize_data(train_zip, labels_filename, training_filename, img_size=nwidth)
    x, y, breed_dict = get_data(training_filename, labels_filename, top_breeds=n_breeds)
    y_hot = keras.utils.to_categorical(y)
    print y_hot.shape
    x_train, x_val, y_train, y_val = train_test_split(x, y_hot, test_size=num_validation, random_state=6)

    nb_train_samples = x_train.shape[0]
    nb_val_samples = x_val.shape[0]

    train_gen = ImageDataGenerator(
        rotation_range=20,
        shear_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)
    train_gen.fit(x_train)
    train_flow = train_gen.flow(x_train, y_train, batch_size=batch_size)

    val_gen = ImageDataGenerator()
    val_gen.fit(x_val)
    val_flow =val_gen.flow(x_val, y_val, batch_size=batch_size)

    # Create the model
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(nwidth, nheight,3), data_format='channels_last'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (4, 4)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3)))
    model.add(Dropout(0.3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2))) 
    
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(n_breeds, activation='softmax'))

    print("Compiling model")
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

    print("Starting generator")
    model.fit_generator(
        train_flow,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=val_flow,
        validation_steps=nb_val_samples // batch_size)

if __name__ == "__main__":
    train()