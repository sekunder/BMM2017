import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.optimizers import Adam, Adadelta
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.callbacks import TensorBoard
from time import gmtime, strftime
import datetime
import os
import sys
import numpy as np
import json

from stimulus import build_dataset

################################################################################
### Misc metadata
################################################################################
# datestring = strftime("%Y%m%d%H%M%S", gmtime())
datestring = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
model_file_format = "h5"

print "============================== BEGIN TRAINING =============================="
print "Training network on %s" % datestring


model_output_dir = os.path.join(os.getcwd(), "model", datestring)
if not os.path.isdir("model"):
	print "Creating directory:", os.path.join(os.getcwd(), "model")
	os.mkdir("model")
if not os.path.isdir(os.path.join("model",datestring)):
	print "Creating directory:", os.path.join(os.getcwd(), "model", datestring)
	os.mkdir(os.path.join("model",datestring))

################################################################################
### Data loading - MNIST
################################################################################

print "Loading MNIST data for training"
batch_size = 128
num_classes = 10
epochs = 10 # FOR TESTING, only 1 epoch.

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# set shapes appropriately
stim_shape = (64,64)
channels = 1 # luminance only
bg_noise = None
print "Embedding images into %d x %d canvases" % stim_shape
if bg_noise is not None:
	print "* bg_noise is " + bg_noise
else:
	print "* no background noise"

# reminder to self - adding tuples together concatenates them in python
if K.image_data_format() == 'channels_first':
	x_train = x_train.reshape(x_train.shape[0], channels, x_train.shape[1], x_train.shape[2])
	x_test = x_test.reshape(x_test.shape[0], channels, x_test.shape[1], x_test.shape[2])
	input_shape = (channels,) + stim_shape
else:
	x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], channels)
	x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], channels)
	input_shape = stim_shape + (channels,)

# convert to floating point values
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# expand canvas and place digits randomly
print "Before enlarging: x_train.shape:", x_train.shape
x_train, obj_pos_train = build_dataset(x_train, img_size=stim_shape, obj_box=(0,0,64,64), data_format=K.image_data_format(), bg_noise=bg_noise)
x_test, obj_pos_test = build_dataset(x_test, img_size=stim_shape, data_format=K.image_data_format(), bg_noise=bg_noise)
print "After englarging: x_train.shape:", x_train.shape
obj_pos_train_filename = "obj_pos_train_" + datestring
obj_pos_test_filename = "obj_pos_test_" + datestring
print "Saving training object positions:", os.path.join(model_output_dir, obj_pos_train_filename)
print "Saving testing object positions: ", os.path.join(model_output_dir, obj_pos_test_filename)
np.save(os.path.join(model_output_dir, obj_pos_train_filename), obj_pos_train)
np.save(os.path.join(model_output_dir, obj_pos_test_filename), obj_pos_test)

# convert class vectors to one-hot class matrices
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)



################################################################################
### Network definition
################################################################################

n_filter1 = 16;  n_filter2 = 32;  n_filter3 = 64
k_size1 = (3,3); k_size2 = (3,3); k_size3 = (3,3)
stride1 = (1,1); stride2 = (1,1); stride3 = (1,1)
dropout_rate = 0.5
n_dense = 1024

model = Sequential([
	Conv2D(n_filter1, kernel_size=k_size1, strides=stride1, padding='same', activation='relu', input_shape=input_shape),
	MaxPooling2D(), # fun fact, the default value for pool size is (2,2)
	Conv2D(n_filter2, k_size2, strides=stride2, padding='same', activation='relu'),
	MaxPooling2D(),
	Conv2D(n_filter3, k_size3, strides=stride3, padding='same', activation='relu'),
	MaxPooling2D(),
	Dropout(dropout_rate),
	Flatten(),
	Dense(n_dense),
	Dense(num_classes),
	Activation('softmax')
])

# model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer=Adadelta(), loss='categorical_crossentropy', metrics=['accuracy'])

print "--- MODEL SUMMARY ---"
print model.summary()
print "Extra info: [dropout rate: %f]" % dropout_rate

################################################################################
### Callbacks
################################################################################

callbacks = [
	TensorBoard(log_dir="logs",histogram_freq=1, batch_size=batch_size, write_images=True)
]

################################################################################
### Fit the model
################################################################################

print "Training network [batch size: %d, epochs: %d]..." % (batch_size, epochs)
sys.stdout.flush() # otherwise, buffer doesn't flush until training is complete, making it difficult to tell if it's running.

model.fit(x_train, y_train,
		  batch_size=batch_size, epochs=epochs,
		  validation_data=(x_test, y_test),
		  callbacks=callbacks,
		  verbose=2)

weights_filename = os.path.join(model_output_dir, "weights_" + datestring + "." + model_file_format)
model_filename = os.path.join(model_output_dir, "model_" + datestring + "." + model_file_format)
# print "Attempting to save weights to ", os.path.join(model_output_dir, datestring + "weights")
# model.save_weights(os.path.join(model_output_dir, datestring + ".h5"))
print "Saving model:", model_filename
model.save(model_filename)

score = model.evaluate(x_test, y_test, verbose=2)
print ""
print "Test loss:", score[0]
print "Test accuracy (%):", score[1] * 100

print "Saving metadata:", os.path.join(model_output_dir, "metadata_" + datestring + ".json")
md_file = open(os.path.join(model_output_dir, "metadata_" + datestring + ".json"), mode='w')
metadata = {"bg_noise":bg_noise}
json.dump(metadata, md_file)
md_file.close()

print "------------------------------  END TRAINING  ------------------------------"



