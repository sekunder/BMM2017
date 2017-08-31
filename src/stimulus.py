from PIL import Image, ImageDraw
import numpy as np
from numpy import random as rand
from numpy.random import RandomState
import os

class StimulusGenerator:
	"""Class for generating stimuli consisting of an object image placed on a uniform gray background"""

	# obj_image, obj_label
	# obj_x
	# obj_y
	# stim_x
	# stim_y

	def __init__(self, obj_file="img/stim_obj/png/16px/cat-16px.png", width=32, height=32, obj_label="cat", obj_name="cat1-16"):
		self.obj_image = Image.open(obj_file)
		self.obj_x, self.obj_y = self.obj_image.size
		self.obj_label = obj_label
		self.obj_name = obj_name
		self.stim_x = width
		self.stim_y = height

	def get_image_box(self):
		return (0,0,self.stim_x, self.stim_y)

	def generate_grayscale(self, bg=127, obj_box=None, obj_placement="random"):
		"""Generate an image with specified background color and the object's bounding box somewhere inside the specified obj_box (coordinates of obj_box are (L,U,R,D))"""
		# first create a blank gray canvas
		img_out = Image.new('L', (self.stim_x, self.stim_y))
		img_draw = ImageDraw.Draw(img_out)
		img_draw.rectangle([0, 0, self.stim_x, self.stim_y], fill=bg)
		# now generate a random coordinate for the upper left corner uniformly
		# so that the bounding box is still fully contained in the resulting image
		if obj_placement == "random":
			if obj_box is None:
				obj_x_min, obj_y_min = 0, 0
				obj_x_max, obj_y_max = self.stim_x, self.stim_y
			else:
				obj_x_min, obj_y_min, obj_x_max, obj_y_max = obj_box
			x = rand.randint(obj_x_min, obj_x_max-self.obj_x)
			y = rand.randint(obj_y_min, obj_y_max-self.obj_y)
		else:
			x,y = obj_placement
		# x, y = rand.randint(self.stim_x - self.obj_x), rand.randint(self.stim_y - self.obj_y)
		if 'A' in self.obj_image.getbands():
			msk = self.obj_image.split()[-1]
		else:
			msk = None
		img_out.paste(self.obj_image, (x, y), msk)

		return img_out

def generate_stimulus_batch(n_stimuli=1, sg=StimulusGenerator(), bg=127, obj_placement=None, output_dir="img/stim", fname_template="LABEL_OBJNAME_OBJX_OBJY_W_H", format="png"):
	"""Create a bunch of stimuli using the given stimulus generator"""
	# outline
	# 1. Check existence of output dir
	# 2. process object placement params
	# 3. for i in range(n_stimuli), produce stimulus and save file

	if not os.path.isdir(output_dir):
		os.mkdir(output_dir)

	obj_box = list(sg.get_image_box())
	if obj_placement is not None:
		tks = obj_placement.split()
		if "upper" in tks:
			obj_box[3] /= 2
		elif "lower" in tks:
			obj_box[1] = obj_box[3]/2
		if "left" in tks:
			obj_box[2] /= 2
		elif "right" in tks:
			obj_box[0] = obj_box[2]/2

	for n in range(n_stimuli):
		img = sg.generate_grayscale(bg, obj_box)

def build_dataset(data, img_size=(64,64), obj_pos='random', obj_box=(0,0,64,64), data_format='channels_last', bg_noise=None, rand_seed=1700):
	"""Take a given data set (e.g. a stack of images) and embed them in a larger canvas."""
	n_batch = data.shape[0]
	RandomState(rand_seed) # should ensure that positions/bg noise are identical each time
	if data_format == 'channels_last':
		obj_w,obj_h = data.shape[1:3]
		# obj_w,obj_h = (28, 28) # hardcoding this to check why shapes are weird
		w_idx, h_idx = 1,2
	else:
		print("Channels first, aborting")
		return None
		# obj_w,obj_h = data.shape[2:4]
		# w_idx, h_idx = 2,3

	new_data = np.zeros((n_batch,)+img_size+(1,), dtype=data.dtype)
	obj_centers = np.zeros((n_batch,2))
	# place objects in larger canvas, either at random or using specified positions
	for batch in range(n_batch):
		# if obj_pos == 'random':
		if type(obj_pos) is str:
			# place object at random
			x = obj_box[0] + rand.randint(obj_box[2] - obj_w)
			y = obj_box[1] + rand.randint(obj_box[3] - obj_h)
		elif len(obj_pos) > 2:
			# assume obj_pos is an array with shape (n_batch,2)
			x = obj_pos[batch,0] - obj_w/2
			y = obj_pos[batch,1] - obj_h/2
		else:
			# assume obj_pos is a tuple with two elements
			x,y = obj_pos
		new_data[batch,x:(x+obj_w),y:(y+obj_h),0] = data[batch,:,:,0]
		obj_centers[batch,:] = [x + obj_w/2, y+obj_h/2]
	# check if nosie should be added
	if bg_noise is not None:
		max_val = data.max()
		if bg_noise == 'uniform':
			# generate pixel values using continuous uniform distribution
			bg = max_val * rand.random(new_data.shape).astype(new_data.dtype)
			# set each pixel to be max(current value, bg value)
			new_data[new_data < bg] = bg[new_data < bg]
		if bg_noise == 'normal':
			# generate pixel values using normal distribution with mean 0.1 and variance 0.01 of max_val
			bg = 0.1 * max_val + 0.1 * rand.standard_normal(new_data.shape).astype(new_data.dtype)
			new_data[new_data < bg] = bg[new_data < bg]

	return new_data, obj_centers

def stim_pairs(data, img_size=(64,64), obj_1_pos='random', obj_2_pos='random', bg_noise=None, rand_seed=1700):
	"""Produce images with pairs of digits. Digits will be placed in two separate quadrants"""
	n_batch = data.shape[0]
	RandomState(rand_seed)
	# assume channels last
	obj_w, obj_h = data.shape[1:3]

	new_data = np.zeros((n_batch,) + img_size + (1,), dtype=data.dtype)
	obj_ids = np.zeros((n_batch/2, 2), dtype='int')
	obj_centers = np.zeros((n_batch/2, 2, 2))
	for idx in range(0, n_batch, 2):
		# for now assume random placement
		q1,q2 = rand.choice(4, 2, replace=False)
		x_1 = 32 * ((q1 & 0x2) >> 1) + rand.randint(img_size[0] - obj_w)
		y_1 = 32 * (q1 & 0x1) + rand.randint(img_size[1] - obj_h)
		x_2 = 32 * ((q2 & 0x2) >> 1) + rand.randint(img_size[0] - obj_w)
		y_2 = 32 * (q2 & 0x1) + rand.randint(img_size[1] - obj_h)
		new_data[idx/2,x_1:(x_1 + obj_w), y_1:(y_1 + obj_h),0] = data[idx,:,:,0]
		new_data[idx/2,x_2:(x_2 + obj_w), y_2:(y_2 + obj_h),0] = data[idx+1,:,:,0]
		obj_centers[idx/2,0,:] = [x_1 + obj_w/2, y_1 + obj_h/2]
		obj_centers[idx/2,1,:] = [x_2 + obj_w/2, y_2 + obj_h/2]
		obj_ids[idx/2, :] = [idx, idx+1]

	return new_data, obj_centers, obj_ids
