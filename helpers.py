import numpy as np
import matplotlib.pyplot as plt
def greyscale(rgb):
	"""converts rgb image to greyscale"""
	if len(rgb.shape) == 3:
		return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

	else:
		print 'Current image does not have 3 color dimensions.'
		return rgb

def default_plot(images, color=False):
	"""plot up to 4 plt images based on tuple of images"""
	color_mapping = "gray" if not color else None
	for i, img in enumerate(images):
		if i < 5:
			plt.figure(i)
			plt.imshow(img, cmap="gray")
		else:
			print "can't show more than 4 images"
	plt.show()	



def sobel_fun():
	"""throwaway function for sobel + watershed"""
	# SOBEL + Watershed
	elevation_map = sobel(ballz)
	markers = np.zeros_like(ballz)
	markers[ballz < 40] = 1
	markers[ballz > 110] = 2
	segmentation = watershed(elevation_map, markers)
	default_plot(ballz, markers)

