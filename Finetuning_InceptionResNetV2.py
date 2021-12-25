import tensorflow as tf
tf.__version__
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input

batchSize = 32
num_classes = 3

train_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)
'''
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True)
'''
train_it = train_datagen.flow_from_directory('./train_generator/', class_mode='categorical', target_size = (200, 200), batch_size = batchSize)

test_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)
test_it = test_datagen.flow_from_directory('./val_generator/', class_mode='categorical', target_size = (200, 200), batch_size = batchSize)

InceptionResnetV2_model = InceptionResNetV2(weights = 'imagenet', include_top = False, input_shape = (200, 200, 3))

InceptionResnetV2_model.summary()

# Freeze four convolution blocks 31 - 8 ver
for layer in InceptionResnetV2_model.layers:
    layer.trainable = False

# Check frozen correct layers
for i, layer in enumerate(InceptionResnetV2_model.layers):
    print(i, layer.name, layer.trainable)

# 1 - 9 ver
x = InceptionResnetV2_model.output
x = Flatten()(x)
x = Dense(num_classes, activation = 'softmax')(x)
transfer_model = Model(inputs = InceptionResnetV2_model.input, outputs = x)

transfer_model.summary()

from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from tensorflow.keras import optimizers

import os
from datetime import date
import time

today = date.today()
d4 = today.strftime("%b-%d-%Y")

t = time.localtime()
current_time = time.strftime("%H-%M-%S", t)

folder_name = d4 + '-' + current_time
folder_path = "./modelStorage/inceptionresnetv2/" + folder_name + "/"
os.makedirs(folder_path)

filepath = folder_path + "weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
loggpath = folder_path + "Result.csv"
csv_logger = CSVLogger(loggpath, append=True)
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
transfer_model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(lr=0.00001), metrics=["accuracy"])

"""
start_epoch = 0
total_epoch = start_epoch + 15
history = transfer_model.fit(train_it, 
                            epochs = total_epoch, 
                            steps_per_epoch = train_it.samples/batchSize,
                            validation_data = test_it,
                            validation_steps= test_it.samples/batchSize,
                            callbacks=[checkpoint, csv_logger])
"""

# Fine tuning pretrained model

modelName = 'weights-improvement-30-0.42.hdf5'
start_epoch = int(modelName.split("-")[2])
total_epoch = start_epoch + 20

transfer_model.load_weights('./modelStorage/inceptionresnetv2/Dec-21-2021-01-26-27/' + modelName)
history = transfer_model.fit(train_it, 
                            epochs = total_epoch, 
                            steps_per_epoch = train_it.samples/batchSize,
                            validation_data = test_it,
                            validation_steps= test_it.samples/batchSize,
                            callbacks=[checkpoint, csv_logger],
                             initial_epoch = start_epoch)


import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

epochs = range(start_epoch,total_epoch)

loss_train = history.history['loss']
loss_val = history.history['val_loss']

ax = plt.figure().gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
figurePath = folder_path + "loss_figure.PNG"
plt.savefig(figurePath)
plt.show()

acc_train = history.history['accuracy']
acc_val = history.history['val_accuracy']

ax = plt.figure().gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

plt.plot(epochs, acc_train, 'g', label='Training accuracy')
plt.plot(epochs, acc_val, 'b', label='validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
figurePath = folder_path + "acc_figure.PNG"
plt.savefig(figurePath)
plt.show()

