"""Script to use trained linear regressions to focus spatial attention"""

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
from numpy import random as rand
from numpy.random import RandomState
from scipy.stats import entropy

from stimulus import build_dataset, stim_pairs
from util import diff_norm


# if len(sys.argv) == 1:
# 	datestring = "20170829151236"
# else:
# 	datestring = sys.argv[1]
datestring = "20170829151236"

################################################################################
### Set up, load model
################################################################################

print "============================== BEGIN ATTENTION =============================="

print "* Processing model " + datestring

if not os.path.isdir("attention"):
	print "  Creating directory:", os.path.join(os.getcwd(), "attention")
	os.mkdir("attention")
if not os.path.isdir(os.path.join("attention", datestring)):
	print "  Creating directory:", os.path.join(os.getcwd(), "attention", datestring)
	os.mkdir(os.path.join("attention",datestring))

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

train_double, train_double_centers, train_double_ids = stim_pairs(x_train)
test_double, test_double_centers, test_double_ids = stim_pairs(x_test)

train_data = build_dataset(x_train, obj_pos=obj_pos_train)[0]
test_data = build_dataset(x_test, obj_pos=obj_pos_test, bg_noise=bg_noise)[0]

################################################################################
### test network performance on double stimuli
################################################################################

dd_softmax = model.predict(test_double)
dd_predict = np.tile(np.argmax(dd_softmax, axis=1), (2, 1)).T
dd_true = y_test[test_double_ids]

dd_acc = float(test_double.shape[0] - np.count_nonzero(np.prod(dd_predict - dd_true, 1))) / float(test_double.shape[0])
print "Base network accuracy on two-digit stimuli (correctly identifies one digit): %4.2f" % (dd_acc * 100.)

dd_acc_matrix = np.zeros((10,10))
for i in range(10):
	for j in range(10):
		ij_trials = [k for k in range(dd_true.shape[0]) if dd_true[k,0] == i and dd_true[k,1] == j]
		correct_trials = [k for k in ij_trials if i in dd_predict[k] or j in dd_predict[k]]
		dd_acc_matrix[i,j] = float(len(correct_trials)) / float(len(ij_trials))

dd_acc_matrix = (dd_acc_matrix + dd_acc_matrix.T) / 2.
plt.matshow(dd_acc_matrix)
plt.colorbar()
plt.xticks(range(10))
plt.yticks(range(10))
plt.title("Accuracy for pairs of digits")
plt.savefig(os.path.join(os.getcwd(), "attention", datestring, "pair_accuracy_no_att.png"))



dd_entropy = [entropy(dd_softmax[i]) for i in range(dd_softmax.shape[0])]
dd_entropy_matrix = np.zeros((10, 10))
for i in range(10):
	for j in range(10):
		entropies = [dd_entropy[k] for k in range(len(dd_entropy)) if dd_true[k, 0] == i and dd_true[k, 1] == j]
		dd_entropy_matrix[i, j] += np.mean(entropies)

dd_entropy_matrix = (dd_entropy_matrix + dd_entropy_matrix.T) / 2.
plt.matshow(dd_entropy_matrix)
plt.colorbar()
plt.xticks(range(10), range(10))
plt.yticks(range(10), range(10))
plt.title("Softmax Entropy for pairs of digits")
plt.savefig(os.path.join(os.getcwd(), "attention", datestring, "pair_entropy_no_att.png"))


################################################################################
### load linear regressions
################################################################################

# first things first: Let's check that the position predictor is linear
# (why on earth should it be)

# let's start with just dense_1
layer_name = "dense_1"
dense_1 = model.layers[-3] # This is true for all the different architectures I used
dense_2 = model.layers[-2]
output_layer = model.layers[-1]
dense_1_npz = np.load(os.path.join(os.getcwd(), "analysis", datestring, "linreg_coeff_" + layer_name + ".npz"))
LR_coef = dense_1_npz['coef']
LR_intercept = dense_1_npz['intercept']

def predict(X, coef=LR_coef, intercept=LR_intercept):
	return np.matmul(X, coef.T) + intercept

f_1 = K.function([model.input, K.learning_phase()], [dense_1.output])
f_2 = K.function([dense_2.input], [output_layer.output])

f_1_test = f_1([test_data, False])[0]
obj_pos_test_predict = predict(f_1_test)
obj_pos_test_err = diff_norm(obj_pos_test_predict, obj_pos_test)
print "Test data, object position prediction. Mean, std:", obj_pos_test_err.mean(), obj_pos_test_err.std()

test_double_first = build_dataset(x_test[test_double_ids[:,0]], obj_pos=test_double_centers[:,0,:])[0]
test_double_second = build_dataset(x_test[test_double_ids[:,1]], obj_pos=test_double_centers[:,1,:])[0]
obj_pos_test_first_predict = predict(f_1([test_double_first, False])[0])
obj_pos_test_first_err = diff_norm(obj_pos_test_first_predict, test_double_centers[:, 0, :])
obj_pos_test_second_predict = predict(f_1([test_double_second, False])[0])
obj_pos_test_second_err = diff_norm(obj_pos_test_second_predict, test_double_centers[:, 1, :])
print "Double stimulus, 1st object, prediction. mean, std:", obj_pos_test_first_err.mean(), obj_pos_test_first_err.std()
print "Double stimulus, 2nd object, prediction. mean, std:", obj_pos_test_second_err.mean(), obj_pos_test_second_err.std()

f_1_double = f_1([test_double, False])[0]
test_double_predict = predict(f_1_double)
test_double_err_from_mid = diff_norm(test_double_predict, test_double_centers.sum(1)/2)
print "Double stimulus, prediction difference from midpoint. mean, std:", test_double_err_from_mid.mean(), test_double_err_from_mid.std()

################################################################################
### Try modifying attention
################################################################################

# first, a simple task: take f_1_double, and modify it as follows:
# take the predicted coordinates [xhat, yhat] and multiply by [-entropy, 0]
# where entropy is the entropy of the softmax prediction for that trial.
# Add this to f_1_double. Then, feed this to f_2 and compare the predictions.

for rho in [10.0**i for i in range(0,-10,-1)]:
	# rho = 0.1
	# lamnu = np.zeros_like(test_double_predict)
	# lamnu[:,0] = np.array(dd_entropy)
	# lamnu[:,1] = np.array(dd_entropy)
	lamnu = np.ones_like(test_double_predict)
	Lprime = f_1_double + rho * np.matmul(lamnu * test_double_predict, LR_coef)

	att_softmax = f_2([Lprime])[0]
	att_predict = np.tile(np.argmax(att_softmax, axis=1), (2,1)).T
	att_acc = float(test_double.shape[0] - np.count_nonzero(np.prod(att_predict - dd_true, 1))) / float(test_double.shape[0])
	print "With attention, rho = %f:  accuracy is %4.2f %%" % (rho, att_acc * 100.)
	print "Compare with no attention: accuracy is %4.2f %%" % (dd_acc * 100.)



print "------------------------------  END ATTENTION  ------------------------------"
