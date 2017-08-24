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

	def generate_grayscale(self, bg=127, obj_box=None):
		"""Generate an image with specified background color and the object's bounding box somewhere inside the specified obj_box (coordinates of obj_box are (L,U,R,D)"""
		# first create a blank gray canvas
		img_out = Image.new('L', (self.stim_x, self.stim_y))
		img_draw = ImageDraw.Draw(img_out)
		img_draw.rectangle([0, 0, self.stim_x, self.stim_y], fill=bg)
		# now generate a random coordinate for the upper left corner uniformly
		# so that the bounding box is still fully contained in the resulting image
		if not obj_box:
			obj_x_min, obj_y_min = 0, 0
			obj_x_max, obj_y_max = self.stim_x, self.stim_y
		else:
			obj_x_min, obj_y_min, obj_x_max, obj_y_max = obj_box
		x = rand.randint(obj_x_min, obj_x_max-self.obj_x)
		y = rand.randint(obj_y_min, obj_y_max-self.obj_y)
		# x, y = rand.randint(self.stim_x - self.obj_x), rand.randint(self.stim_y - self.obj_y)
		if 'A' in self.obj_image.getbands():
			msk = self.obj_image.split()[-1]
		else:
			msk = None
		img_out.paste(self.obj_image, (x, y), msk)

		return img_out

def generate_stimulus_batch(n_stimuli=100, sg=StimulusGenerator(), obj_placement=None, output_dir="img/stim"):
	"""Create a bunch of stimuli using the given stimulus generator"""
	# outline
	# 1. Check existence of output dir
	# 2. process object placement params
	# 3. 