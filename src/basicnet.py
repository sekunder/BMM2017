# A basic image classification network.

import tensorflow as tf

# images are w x h grayscale images (c = 1 channel, luminance only)
w = 64
h = 64
c = 1

# object icons will be 24x24
obj_h = 24
obj_w = 24

n_classes = 5				# number of distinct classes
ker_sz1 = 5					# size of kernel for first convolution
n_ker1 = 32					# number of kernels for first convolution
pool_sz1 = 2; pool_st1 = 2	# size and stride for first pooling layer
ker_sz2, n_ker2 = 5, 64		# kernels for 2nd conv
pool_sz2 = 2; pool_st2 = 2	# 2nd max pool
n_full3 = 1024				# number of output units in fully connected layer

# Helper functions for creating my CNN
def init_weights(shape):
    """ Random initialize weight variable """
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def init_bias(shape):
    """ Initialize bias variable to zero """
    return tf.Variable(tf.zeros(shape))

###############################################################
# CNN CONFIGURATION
###############################################################
dropout = 0.5
learning_rate = 0.001
training_iters = 500
batch_size = 100
step_display = 10
step_save = 500
path_save = 'models/mnist_cnn'


###############################################################
# CNN NETWORK DEFINITION
###############################################################

# let's collect all the weights I'll need here
weights = {
	'wc1': init_weights([ker_sz1, ker_sz1, c, n_ker1]),
	'wc2': init_weights([ker_sz2, ker_sz2, n_ker1, n_ker2]),
	'wf3': init_weights([w * h * n_ker2 / (pool_st1 * pool_st2), n_full3]),
	'wo': init_weights([n_full3, n_classes])
}
biases = {
	'bc1': init_bias(n_ker1),
	'bc2': init_bias(n_ker2),
	'bf3': init_bias(n_full3),
	'bo': init_bias(n_classes)
}

input_layer = tf.placeholder(tf.float32, [None, h, w, c], name="input_img")
correct_labels = tf.placeholder(tf.int64, None, name="output_class")
keep_dropout = tf.placeholder(tf.float32)


# first layer Convolution + ReLU + pooling
conv1 = tf.nn.conv2d(input_layer, weights['wc1'], strides=[1, 1, 1, 1], padding='SAME')
relu1 = tf.nn.relu(tf.nn.bias_add(conv1, biases['bc1']))
pool1 = tf.nn.max_pool(relu1, ksize=[1,pool_sz1,pool_sz1,1],strides=[1,pool_st1,pool_st1,1], padding='SAME')

# second layer Conv + ReLU + pooling
conv2 = tf.nn.conv2d(pool1, weights['wc2'], strides=[1,1,1,1], padding='SAME')
relu2 = tf.nn.relu(tf.nn.bias_add(conv2, biases['bc2']))
pool2 = tf.nn.max_pool(relu2, ksize=[1,pool_sz2,pool_sz2,1], strides=[1,pool_st2,pool_st2,1], padding='SAME')

# fully connected + relu
temp3 = tf.reshape(pool2, [-1, weights['wf3'].get_shape().as_list()[0]])
fc3 = tf.nn.relu(tf.add(tf.matmul(temp3, weights['wf3']), biases['bf3']))
do3 = tf.nn.dropout(fc3, keep_dropout)

# the output layer of the network
logits = tf.add(tf.matmul(do3, weights['wo']), biases['bo'])

###############################################################
# CNN LOSS AND TRAINING FUNCTIONS
###############################################################

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=correct_labels))
train_optimizer = tf.train.AdamOptimizer().minimize(loss)

correct_pred = tf.equal(tf.argmax(logits,1), correct_labels)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

###############################################################
# INITIALIZE SESSION
###############################################################

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Define saver to write out model checkpoint
saver = tf.train.Saver()

# Configure summary writer to write out logs
LOG_DIR = "logs"
summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
merged = tf.summary.merge_all()

step = 0

while step < training_iters:
	images_batch,labels_batch = load_batch()
	#TODO implement loader
	summary, _ = sess.run([merged, train_optimizer], feed_dict={input_layer: images_batch, correct_labels: labels_batch, keep_dropout: dropout})

	# display loss and accuracy every so often
	if step % step_display == 0:
		l, acc = sess.run([loss, accuracy], feed_dict={input_layer: images_batch, correct_labels:labels_batch, keep_dropout: 1.0})
		print("Iter %d, minibatch loss = %0.6f, training accuracy = %0.4f" % (step, l, acc))

	step += 1

	if step % step_save == 0:
		saver.save(sess, path_save, global_step=step)
		print("Model saved at iter %d" % (step))

	summary_writer.add_summary(summary, step)