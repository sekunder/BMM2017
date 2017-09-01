"""Script to train linear regression and/or linear decoder on output of each layer of a trained network"""

from keras.utils import to_categorical
from keras.datasets import mnist
import keras.backend as K
import numpy as np
from numpy.random import rand
from numpy import reshape
import keras
import sys
import os
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import json

from stimulus import build_dataset
from util import diff_norm

################################################################################
### Set up, load model
################################################################################

print "============================== BEGIN ANALYSIS =============================="

if len(sys.argv) == 1:
	datestring = "20170829151236"
else:
	datestring = sys.argv[1]

print "* Processing model " + datestring

if not os.path.isdir("analysis"):
	print "  Creating directory:", os.path.join(os.getcwd(), "analysis")
	os.mkdir("analysis")
if not os.path.isdir(os.path.join("analysis", datestring)):
	print "  Creating directory:", os.path.join(os.getcwd(), "analysis", datestring)
	os.mkdir(os.path.join("analysis",datestring))

model = keras.models.load_model(os.path.join(os.getcwd(),"model",datestring,"model_" + datestring + ".h5"))
obj_pos_test = np.load(os.path.join(os.getcwd(),"model",datestring,"obj_pos_test_" + datestring + ".npy"))
obj_pos_train = np.load(os.path.join(os.getcwd(),"model",datestring,"obj_pos_train_" + datestring + ".npy"))

if os.path.isfile(os.path.join(os.getcwd(), "model", datestring, "metadata_" + datestring + ".json")):
	md_file = open(os.path.join(os.getcwd(), "model", datestring, "metadata_" + datestring + ".json"),'r')
	metadata = json.load(md_file)
else:
	metadata = {}

# for k,v in metadata:
# 	# for now, just hack it together: I know that bg_noise is the only thing in there. I can worry about improving this for general use later.
# 	pass
bg_noise = metadata.get('bg_noise')

################################################################################
### Load mnist data
################################################################################

print "* Loading MNIST data"
(x_train, y_train), (x_test, y_test) = mnist.load_data()
stim_shape = (64,64)
channels = 1 # luminance only
train_samples = x_train.shape[0]
test_samples = x_test.shape[0]

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

x_test, obj_pos_test_check = build_dataset(x_test, img_size=stim_shape, obj_pos=obj_pos_test, data_format=K.image_data_format(), bg_noise=bg_noise)
x_train, obj_pos_train_check = build_dataset(x_train, img_size=stim_shape, obj_pos=obj_pos_train, bg_noise=bg_noise)

if np.alltrue(obj_pos_test == obj_pos_test_check) and np.alltrue(obj_pos_train == obj_pos_train_check):
	print "  Test and train stimuli successfully loaded"
else:
	print "**** WARNING! Failed to properly load test stimuli! Aborting... ****"
	exit()

del obj_pos_test_check, obj_pos_train_check

# Goals:
# 1. Run model on every test image, and compare predicted digit vs. real digit. create a confusion matrix, save it
# 2. Grab output from each layer, train a linear regression (output -> obj_position)
# 3. Grab output from each layer, train a linear decoder (e.g. perceptron, SVM) to separate, say, 7
# 4. Create modified network with ``attention'' module(s):
#		a. if softmax output is ambiguous, focus attention on position
#		b. if softmax output is ambiguous, focus attention on 7

################################################################################
### confusion matrix
################################################################################

print "* Computing confusion matrix"
sys.stdout.flush()

base_model_prediction = model.predict(x_test) # return will be shape (n_tests, 10)
base_model_confusion_matrix = np.matmul(base_model_prediction.T, to_categorical(y_test)) / np.sum(to_categorical(y_test), 0)

# display the matrix cause why not
plt.matshow(base_model_confusion_matrix)
plt.colorbar()
plt.title("Base network confusion matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks(np.arange(10),np.arange(10))
plt.yticks(np.arange(10),np.arange(10))
plt.savefig(os.path.join(os.getcwd(), "analysis", datestring,"base_confusion_matrix.png"))


################################################################################
### linear regression
################################################################################

print "* Linear regressions"


print "  Examining layer: input (raw pixels)"
print "    layer output: 64 x 64 x 1 image"
r_pix = (rand(train_samples) < 1./6.)
n_pix_sample = np.count_nonzero(r_pix)
print "    Using subsample of %d training samples to fit linear regression" % n_pix_sample
pix_train = reshape(x_train[r_pix], [n_pix_sample,-1])
pix_lr = LinearRegression(normalize=True)
pix_lr.fit(pix_train, obj_pos_train[r_pix])
obj_pos_train_pix_predict = pix_lr.predict(pix_train)
obj_pos_train_pix_err = diff_norm(obj_pos_train_pix_predict, obj_pos_train[r_pix])
print "    min, max | mean, std of position error on training data: %8.4f\t%8.4f | %8.4f\t%8.4f" % (obj_pos_train_pix_err.min(), obj_pos_train_pix_err.max(), obj_pos_train_pix_err.mean(), obj_pos_train_pix_err.std())
del pix_train
pix_test = reshape(x_test, [x_test.shape[0],-1])
obj_pos_test_pix_predict = pix_lr.predict(pix_test)
obj_pos_test_pix_err = diff_norm(obj_pos_test_pix_predict, obj_pos_test)
print "    min, max | mean, std of position error on testing data:  %8.4f\t%8.4f | %8.4f\t%8.4f" % (obj_pos_test_pix_err.min(), obj_pos_test_pix_err.max(), obj_pos_test_pix_err.mean(), obj_pos_test_pix_err.std())
print "    Saving regression coefficients to", os.path.join(os.getcwd(), "analysis", datestring, "linreg_coeff_pixels.npz")
np.savez(os.path.join(os.getcwd(), "analysis", datestring, "linreg_coeff_pixels"), coef=pix_lr.coef_, intercept=pix_lr.intercept_)

sys.stdout.flush()

pool_and_dense_layers = [l for l in model.layers
						 if (type(l) is keras.layers.MaxPooling2D or type(l) is keras.layers.Dense)]
#						 and np.prod(l.output.shape.as_list()[1:]) <= 4096]
for l in pool_and_dense_layers:
	print "  Examining layer:", l.name
	print "    Layer output:", l.output
	f = K.function([model.input, K.learning_phase()], [l.output])

	# x_train is too big for my poor little laptop. So for now, I'm going to take a random subsample for fitting
	r_l = (rand(train_samples) < 1. / 6.)
	n_subsample = np.count_nonzero(r_l)
	print "    Using subsample of %d training samples to fit linear regression" % n_subsample

	l_train = reshape(f([x_train[r_l], False])[0], [n_subsample, -1])

	print "    Fitting linear regression on layer output vs. object position"
	lr = LinearRegression()
	lr.fit(l_train, obj_pos_train[r_l])
	obj_pos_train_predict = lr.predict(l_train)
	obj_pos_train_err = diff_norm(obj_pos_train_predict, obj_pos_train[r_l])
	print "    min, max | mean, std of position error on training data: %8.4f\t%8.4f | %8.4f\t%8.4f" % (obj_pos_train_err.min(), obj_pos_train_err.max(), obj_pos_train_err.mean(), obj_pos_train_err.std())

	# again, my computer is wimpy, so I'll suggest the garbage collector get over here
	del l_train

	l_test = reshape(f([x_test, False])[0], [x_test.shape[0], -1])
	obj_pos_test_predict = lr.predict(l_test)
	obj_pos_test_err = diff_norm(obj_pos_test_predict, obj_pos_test)
	print "    min, max | mean, std of position error on testing data:  %8.4f\t%8.4f | %8.4f\t%8.4f" % (obj_pos_test_err.min(), obj_pos_test_err.max(), obj_pos_test_err.mean(), obj_pos_test_err.std())
	print "    Saving regression coefficients to", os.path.join(os.getcwd(), "analysis", datestring, "linreg_coeff_" + str(l.name) + ".npz")
	np.savez(os.path.join(os.getcwd(), "analysis", datestring, "linreg_coeff_" + str(l.name)), coef=lr.coef_, intercept=lr.intercept_)
	print ""
	sys.stdout.flush()
	

	# produce a boxplot with error values grouped by digit
	# errs_by_digit = [obj_pos_test_err[y_test == d] for d in range(10)]



################################################################################
### linear decoder
################################################################################

print "------------------------------  END ANALYSIS  ------------------------------"
