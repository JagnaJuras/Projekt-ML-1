import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
from tensorflow.keras.layers import Conv2D,Add,MaxPooling2D, Dense, BatchNormalization,Input,Flatten, Dropout,GlobalMaxPooling2D,Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# print the directory
print(os.listdir('../input/FishImgDataset'))

train = '../input/FishImgDataset/train'
validation = '../input/FishImgDataset/val'
test = '../input/FishImgDataset/test'

# DATA AUGMENTATION
train_datagen = ImageDataGenerator(rescale=1./255,    # convert to target values between 0 and 1 for faster training
                                   shear_range=0.2,   # for randomly applying shearing transformations
                                   zoom_range=0.2,    # for randomly zooming inside pictures
                                   horizontal_flip=True)    # for randomly flipping half of the images horizontally
# initialize train generator
valid_datagen = ImageDataGenerator(rescale=1.0/255.)    # initialize validation generator
test_datagen = ImageDataGenerator(rescale=1.0/255.)     # initialize validation generator

# flow_from_directory it takes the path to a directory & generates batches of augmented data.
# target_size the dimensions to which all images found will be resized
# batch_size the size of the batches of data (default: 32)
# class_mode determines the type of label arrays that are returned ("categorical" for 2D labels,"binary" for 1D binary
# labels,"sparse" for 1D integer labels,and "input" for identical image)
# shuffle use whether to shuffle the data (default: True) If set to False, sorts the data in alphanumeric order.

train_generator = train_datagen.flow_from_directory(train,
                                                    target_size=(224, 224),
                                                    batch_size=32,
                                                    class_mode='categorical')
validation_generator = valid_datagen.flow_from_directory(validation,
                                                         target_size=(224, 224),
                                                         batch_size=32,
                                                         class_mode='categorical')
test_generator = test_datagen.flow_from_directory(test,
                                                  target_size=(224, 224),
                                                  batch_size=32,
                                                  class_mode='categorical',
                                                  shuffle=False)

# Load Model
inception = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# inception.summary()

inception.trainable = True
for layer in inception.layers[:197]:
    layer.trainable = False

# for idx, layer in enumerate(inception.layers):
#     print(f' {idx}:  {layer.name}: trainable = {layer.trainable}')

# get the last layer
last_layer = inception.get_layer('mixed7')
layer_output = last_layer.output

n_categories = len(os.listdir('../input/FishImgDataset/train'))     # number of categories
print(n_categories)
# x  = BatchNormalization()(layer_output)
xs = Flatten()(layer_output)
xs = Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(xs)
xs = Dropout(0.4)(xs)
xs = Dense(n_categories, activation='softmax')(xs)

model = Model(inputs=inception.inputs, outputs=xs)

# Set the training parameters
model.compile(optimizer=RMSprop(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

epochs = 10


def scheduler(epoch, lr):
    if epoch < epochs:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


callback = tf.keras.callbacks.LearningRateScheduler(scheduler)


history = model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=epochs,
            callbacks=[callback]
)


plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()
plt.plot(history.history['accuracy'], label='acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()

model_name = 'FishModelClassifier_V6.h5'
model.save(model_name, save_format='h5')
model.save_weights('model_weights_V6.weights.h5')

results = model.evaluate(test_generator, verbose=0)


print("    Test Loss: {:.5f}".format(results[0]))
print("Test Accuracy: {:.2f}%".format(results[1] * 100))
