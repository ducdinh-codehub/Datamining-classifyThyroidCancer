#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
tf.__version__
gpus = tf.config.experimental.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(gpus[0], True)
from tensorflow.python.keras.applications.nasnet import NASNetLarge
from tensorflow.python.keras.applications.nasnet import preprocess_input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adadelta
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


batchSize = 32
num_classes = 3

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
"""shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True)"""
train_it = train_datagen.flow_from_directory('./train_generator/', class_mode='categorical', target_size = (331, 331), batch_size = batchSize)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_it = test_datagen.flow_from_directory('./val_generator/', class_mode='categorical', target_size = (331, 331), batch_size = batchSize)

nasnet_model = NASNetLarge(weights = 'imagenet', include_top = False, input_shape = (331, 331, 3))

nasnet_model.summary()

# Freeze four convolution blocks
for layer in nasnet_model.layers:
    layer.trainable = False

# Check frozen correct layers
for i, layer in enumerate(nasnet_model.layers):
    print(i, layer.name, layer.trainable)

x = nasnet_model.output
x = Flatten()(x)
#x = Dense(64, activation = 'relu')(x)
x = Dense(128, activation = 'relu')(x)
#x = Dense(256, activation = 'relu')(x)
x = Dense(num_classes, activation = 'softmax')(x)
transfer_model = Model(inputs = nasnet_model.input, outputs = x)
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
folder_path = "./modelStorage/nasnetlarge/" + folder_name + "/"

os.makedirs(folder_path)

filepath = folder_path + "weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
loggpath = folder_path + "result.csv"
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

print("Improving model performance")
modelName = 'weights-improvement-23-0.41.hdf5'
start_epoch = int(modelName.split("-")[2])
total_epoch = start_epoch + 20
transfer_model.load_weights('./modelStorage/nasnetlarge/Dec-21-2021-22-09-54/' + modelName)

# In[ ]:
history = transfer_model.fit(train_it, 
                            epochs = total_epoch,
                            steps_per_epoch = train_it.samples/batchSize,
                            validation_data = test_it,
                            validation_steps= test_it.samples/batchSize,
                            callbacks=[checkpoint, csv_logger],
                            initial_epoch = start_epoch)

# In[ ]:
epochs = range(start_epoch, total_epoch)

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




