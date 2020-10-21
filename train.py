import keras.backend as K
from keras.layers import *
from keras.losses import binary_crossentropy, categorical_crossentropy
from keras.models import *
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras

"""
We explicitly redefine the Squeezent architecture since Keras has no predefined Squeezent
"""


def squeezenet_fire_module(input, input_channel_small=16, input_channel_large=64):

    channel_axis = 3

    input = Conv2D(input_channel_small, (1, 1), padding="valid")(input)
    input = Activation("relu")(input)

    input_branch_1 = Conv2D(input_channel_large, (1, 1),
                            padding="valid")(input)
    input_branch_1 = Activation("relu")(input_branch_1)

    input_branch_2 = Conv2D(input_channel_large, (3, 3), padding="same")(input)
    input_branch_2 = Activation("relu")(input_branch_2)

    input = concatenate([input_branch_1, input_branch_2], axis=channel_axis)

    return input


def SqueezeNet(input_shape=(224, 224, 3)):

    image_input = Input(shape=input_shape)

    network = Conv2D(64, (3, 3), strides=(2, 2), padding="valid")(image_input)
    network = Activation("relu")(network)
    network = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(network)

    network = squeezenet_fire_module(
        input=network, input_channel_small=16, input_channel_large=64)
    network = squeezenet_fire_module(
        input=network, input_channel_small=16, input_channel_large=64)
    network = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(network)

    network = squeezenet_fire_module(
        input=network, input_channel_small=32, input_channel_large=128)
    network = squeezenet_fire_module(
        input=network, input_channel_small=32, input_channel_large=128)
    network = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(network)

    network = squeezenet_fire_module(
        input=network, input_channel_small=48, input_channel_large=192)
    network = squeezenet_fire_module(
        input=network, input_channel_small=48, input_channel_large=192)
    network = squeezenet_fire_module(
        input=network, input_channel_small=64, input_channel_large=256)
    network = squeezenet_fire_module(
        input=network, input_channel_small=64, input_channel_large=256)

    # Remove layers like Dropout and BatchNormalization, they are only needed in training
    #network = Dropout(0.5)(network)

    # network = Conv2D(64, kernel_size=(
    #    1, 1),  name="last_conv")(network)
    #network = Activation("relu")(network)
    network = Conv2D(2, kernel_size=(
        1, 1),  name="last_conv")(network)
    network = Activation("relu")(network)

    # network = Conv2D(2, kernel_size=(
    #    1, 1), padding="valid", name="last_conv2")(network)
    #network = Activation("relu")(network)
    network = GlobalAvgPool2D()(network)
    network = Activation('softmax')(network)
    input_image = image_input
    model = Model(inputs=input_image, outputs=network)

    return model


def train():
    from argparse import ArgumentParser
    import os
    par = ArgumentParser()
    par.add_argument('--root', type=str, help='dataset root')
    par.add_argument('--epochs', type=int, default=50)
    arg = par.parse_args()
    model = SqueezeNet()

    image_size = (224, 224)
    batch_size = 32
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        brightness_range=[0.8, 1.2],
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        os.path.join(arg.root, 'Train'),
        target_size=(224, 224),
        batch_size=32,

    )

    validation_generator = test_datagen.flow_from_directory(
        os.path.join(arg.root, 'Validation'),
        target_size=(224, 224),
        batch_size=32)

    # if want to use SGD, first define sgd, then set optimizer=sgd

    # select loss\optimizer\
    epochs = arg.epochs
    ada = Adam(lr=6e-5, decay=5e-4)
    from keras.callbacks import ModelCheckpoint
    callbacks = [
        ModelCheckpoint("sq_{epoch}.h5"),
    ]
    model.compile(
        optimizer=ada,
        loss=categorical_crossentropy,
        metrics=["categorical_accuracy"],
    )
    model.fit(
        train_generator, epochs=epochs, callbacks=callbacks, validation_data=validation_generator,
    )


if __name__ == '__main__':
    train()
