# DONE: use coin trick to segment out single ballz

# TODO: (get dominant color) -> e.g. use histogram; find peak;
# 		... mask anything around that peak; blacken off table edge.
# TODO: try opencv hough -> http://www.pyimagesearch.com/2014/07/21/detecting-circles-images-using-opencv-hough-circles/
# TODO: color separation
# TODO: try template matching algos

"""links:
https://stackoverflow.com/questions/40717587/detecting-balls-on-a-pool-table
file:///Users/franzr/Downloads/Weatherford_Pool_Table_Cue_Guide.pdf
http://www.pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/
http://www.pyimagesearch.com/2015/09/21/opencv-track-object-movement/
"""

# ----first try----------------
import numpy as np 
import matplotlib.pyplot as plt
from skimage import io, filters, color, draw
# -- coin example imports:
# (http://scikit-image.org/docs/dev/user_guide/tutorial_segmentation.html)
from skimage.feature import canny
from scipy import ndimage as ndi
from skimage.filters import sobel
from skimage.morphology import watershed
# -- custom imports
from helpers import greyscale, default_plot
# -- hough example imports:
# (http://scikit-image.org/docs/0.8.0/auto_examples/plot_circular_hough_transform.html)
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import peak_local_max
from skimage.util import img_as_ubyte
# -- other seg experiments:
from skimage.segmentation import quickshift
# -- erosion/dilation
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.morphology import disk
# -- histogram
from skimage.exposure import histogram


# Light blue is roughly rgb(56,186,199)
LIGHT_BLUE = (56,186,199)
# outline: import png, get thresholding shit and hough transform going
# import screen shot for now

def edge_fun(img, filled=False):
	"""throwaway canny function, by default do not attempt to fill edges"""
	edges = canny(img/255., sigma=2)
	if filled:
		return ndi.binary_fill_holes(edges)
	else:
		return edges

def eroder(img, disk_size):
	selem = disk(disk_size)
	eroded = erosion(img, selem)
	return eroded

def closer(img, disk_size):
	selem = disk(disk_size)
	closed = closing(img, selem)
	return closed

def color_histo(img):
	"""returns histogram of each color channel in RGB img"""
	# check if rgb(a):
	if img.shape[2] in (3,4):
		channels = (img[::1],img[::2],img[::3])
	elif img.shape[2] == 1: 
		channels = img[::1]
	# return channels:
	else:
		print "weird number of color channels going on: ", img.shape
	return (histogram(chan) for chan in channels)

def remove_color(colored_pic, rgb_color, tolerance=0.6):
	"""blanks out any color in image within tolerance-percentage of color given"""
	# surprisingly, a high tolerance works best for the training pic...
	img = colored_pic.copy()
	# create color tolerance limits based on rgb color
	rlims,glims,blims = ((rgb_color[i]*(1.0-tolerance),rgb_color[i]*(1+tolerance)) for i in range(3))
	# set to black where within tolerance limits
	# rgb stored as [[(255,255,255), (0,0,0)], [(100,100,100), (5,5,5)], etc...]
	# -- this works suprisingly well as first guess:
	# rimg[rimg[:,:,0]>60]=255
	# -- but need to find 
	print rlims, glims, blims
	img[((img[:, :, 0]>rlims[0]) & (img[:, :, 0]<rlims[1])) & 
	((img[:, :, 1]>glims[0]) & (img[:, :, 1]<glims[1])) &
	((img[:, :, 2]>blims[0]) & (img[:, :, 2]<blims[1]))] = 255 
	return img

def hough_fun(img):
	"""fit circles to segmented image based on plausible range of radii"""
	# based on screen shot, I'm guessing about 25px radius for now
	hough_radii = np.arange(28,45)
	hough_res = hough_circle(img, hough_radii)
	blank = img.copy()
	blank[::] = 0
	"""
	accum, cx, cy, rad = hough_circle_peaks(hough_res, hough_radii)
	for i, ac in enumerate(np.argsort(accum)[::-1][:10]):
		center_x = cx[i]
		center_y = cy[i]
		radius = rad[i]
		cx, cy = draw.circle_perimeter(center_y, center_x, radius)
		blank[cy, cx] = 255
	return blank
	"""
	# if can't import hough_circle_peaks, try to replicate:
	centers = []
	accums = []
	radii = []
	for radius, h in zip(hough_radii, hough_res):
	# For each radius, extract, say, 3 circles
		peaks = peak_local_max(h, num_peaks=2)
		centers.extend(peaks - hough_radii.max())
		accums.extend(h[peaks[:, 0], peaks[:, 1]])
		radii.extend([radius, radius])
	for idx in np.argsort(accums)[::-1][:25]:
		center_x, center_y = centers[idx]
		radius = radii[idx]
		cx, cy = draw.circle_perimeter(center_y, center_x, radius)
		blank[cy, cx] = 255
	return blank

def quick_fun(img):
	"""throwaway function for trying quickshit segmentation"""
	return quickshift(img, convert2lab=True)

# fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))


def main():
	ballz = greyscale(io.imread("sc_cropped.png"))
	color_ballz = io.imread("sc_cropped.png")
	print color_ballz[0,0,0]
	# r,g,b = color_histo(color_ballz)
	# plt.plot(r[1],r[0],g[1],g[0], b[1], b[0])
	# plt.show()
	# result = edge_fun(ballz)
	# default_plot([edge_fun(ballz), result])
	uncolored = remove_color(color_ballz, LIGHT_BLUE)
	cannied = edge_fun(greyscale(uncolored))
	houghed = hough_fun(cannied)
	default_plot([color_ballz, uncolored, cannied, houghed], color=True)

if __name__ == "__main__":
	main()
	



