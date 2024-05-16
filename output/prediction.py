from keras.models import load_model
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from sklearn.metrics import classification_report, confusion_matrix
import itertools

model = load_model('FishModelClassifier_V6.h5', compile=False)
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

train_generator = train_datagen.flow_from_directory(train, target_size=(224, 224), batch_size=32, class_mode='categorical')
validation_generator = valid_datagen.flow_from_directory(validation, target_size=(224, 224), batch_size=32, class_mode='categorical')
test_generator = test_datagen.flow_from_directory(test, target_size=(224, 224), batch_size=32, class_mode='categorical', shuffle=False)

class_name = ['Bangus', 'Big Head Carp', 'Black Spotted Barb', 'Catfish', 'Climbing Perch', 'Fourfinger Threadfin',
              'Freshwater Eel', 'Glass Perchlet', 'Goby', 'Gold Fish', 'Gourami', 'Grass Carp', 'Green Spotted Puffer',
              'Indian Carp', 'Indo-Pacific Tarpon', 'Jaguar Gapote', 'Janitor Fish', 'Knifefish',
              'Long-Snouted Pipefish',
              'Mosquito Fish', 'Mudfish', 'Mullet', 'Pangasius', 'Perch', 'Scat Fish', 'Silver Barb', 'Silver Carp',
              'Silver Perch', 'Snakehead', 'Tenpounder', 'Tilapia']


fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(25, 25))
dic = {i: ax for i, ax in enumerate(axes.flat)}

for i, (images, labels) in enumerate(test_generator):
    if i * 4 >= len(dic):  # Ensure we don't exceed the number of axes
        break

    preds = model.predict(images)
    for j in range(len(images)):
        if i * 4 + j >= len(dic):  # Ensure we don't exceed the number of axes
            break

        label = np.argmax(labels[j])
        pred = np.argmax(preds[j])
        image = images[j]

        dic[i * 4 + j].set_title(
            "real label: " + str(class_name[label]) + " v.s " + "predicted label: " + str(class_name[pred]))
        dic[i * 4 + j].imshow(image)

plt.tight_layout()
plt.show()


def predict(path):
    img = load_img(path, target_size=(224, 224, 3))  # convert image size and declare the kernel
    img = img_to_array(img)  # convert the image to an image array
    img = img / 255  # rgb is 255
    img = np.expand_dims(img, [0])
    answer = model.predict(img)
    x = list(np.argsort(answer[0])[::-1][:5])

    for i in x:
        print("{className}: {predVal:.2f}%".format(className=class_name[i], predVal=float(answer[0][i]) * 100))

    y_class = answer.argmax(axis=-1)
    y = " ".join(str(x) for x in y_class)
    y = int(y)
    res = class_name[y]

    return res


def get_random_image():
    dataset_dir = '../input/FishImgDataset/test/'
    fish_species = os.listdir(dataset_dir)
    random_species = random.choice(fish_species)

    images_dir = os.path.join(dataset_dir, random_species)
    images = os.listdir(images_dir)
    random_image = random.choice(images)

    image_path = os.path.join(images_dir, random_image)
    return image_path


# Now, you can use get_random_image() function to get a random image path
random_image_path = get_random_image()
print("Random image path:", random_image_path)
print("Predicted class:", predict(random_image_path))

# Display the random image
random_img = load_img(random_image_path, target_size=(224, 224, 3))
plt.imshow(random_img, aspect="auto")
plt.show()

# -------------------------

# Get list of layers from model
layer_outputs = [layer.output for layer in model.layers[1:]]
# Create a visualization model
visualize_model = tf.keras.models.Model(inputs = model.input, outputs = layer_outputs)

# Load image for prediction
img=load_img('../input/FishImgDataset/test/Mudfish/Mudfish 013.jpg',target_size=(224,224))
# Convert image to array
x = img_to_array(img)
# Print shape of array
x.shape

# Reshape image for passing it to prediction
x=x.reshape((1,224,224,3))
print(x.shape)
# Rescale the image
x = x /255

# Get all layers feature maps for image
feature_maps=visualize_model.predict(x)
print(len(feature_maps))
# Show names of layers available in model
layer_names = [layer.name for layer in model.layers]

# Plotting the graph
for layer_names, feature_maps in zip(layer_names, feature_maps):
    if len(feature_maps.shape) == 4:
        channels = feature_maps.shape[-1]
        size = feature_maps.shape[1]
        display_grid = np.zeros((size, size * channels))
        for i in range(channels):
            x = feature_maps[0, :, :, i]
            x_mean = x.mean()
            x_std = x.std()
            if x_std != 0:
                x -= x_mean
                x /= x_std
                x *= 64
                x += 128
                x = np.clip(x, 0, 255).astype('uint8')
            else:
                x = np.zeros_like(x).astype('uint8')  # Handle division by zero
            # We'll tile each filter into this big horizontal grid
            display_grid[:, i * size : (i + 1) * size] = x

        scale = 20. / channels
        plt.figure(figsize=(scale * channels, scale))
        plt.title(layer_names)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        plt.close()  # Close the figure after displaying it


def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap='rainbow'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(20,20))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, decimals=2)
        cm[np.isnan(cm)] = 0.0
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "white")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png')

#Print the Target names

#shuffle=False

target_names = []
for key in train_generator.class_indices:
    target_names.append(key)

# print(target_names)

#Confution Matrix

Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
cm = confusion_matrix(test_generator.classes, y_pred)
plot_confusion_matrix(cm, target_names, title='Confusion Matrix')

#Print Classification Report
print('Classification Report')
print(classification_report(test_generator.classes, y_pred, target_names=target_names))