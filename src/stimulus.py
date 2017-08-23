from PIL import Image, ImageDraw
from numpy import random as rand


class StimulusGenerator:
	"""Class for generating stimuli consisting of an object image placed on a uniform gray background"""

	# obj_image
	# obj_x
	# obj_y
	# stim_x
	# stim_y

	def __init__(self, obj_file="img/stim_obj/png/16px/cat-16px.png", width=32, height=32):
		self.obj_image = Image.open(obj_file)
		self.obj_x, self.obj_y = self.obj_image.size
		self.stim_x = width
		self.stim_y = height

	def generate_grayscale(self):
		# first create a blank gray canvas
		img_out = Image.new('L', (self.stim_x, self.stim_y))
		img_draw = ImageDraw.Draw(img_out)
		img_draw.rectangle([0, 0, self.stim_x, self.stim_y], fill=127)
		# now generate a random coordinate for the upper left corner uniformly
		# so that the bounding box is still fully contained in the resulting image
		x, y = rand.randint(self.stim_x - self.obj_x), rand.randint(self.stim_y - self.obj_y)
		if 'A' in self.obj_image.getbands():
			msk = self.obj_image.split()[-1]
		else:
			msk = None
		img_out.paste(self.obj_image, (x, y), msk)
		return img_out

	# def generate_binary(self, tile_sz=1, bg="random"):
	# 	"""Generates a binary stimulus, using luminence values as probability of on/off"""
	# 	img_out = Image.new('1', (self.stim_x, self.stim_y))
	# 	bin_obj =
	# 	return img_out
