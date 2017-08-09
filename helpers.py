import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import sobel, threshold_otsu, threshold_mean
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage import io
from scipy import ndimage as ndi
from skimage import util 
import cv2

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
			plt.axis("off")
		else:
			print "can't show more than 4 images"
	plt.show()	

def cv_watershed_markers(filename, wanna_plot=False):
	"""for now, just return markers of certain ballz given by dist transform and thresh"""
	img = cv2.imread(filename)
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	# noise removal
	kernel = np.ones((3,3),np.uint8)
	opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
	# sure background area
	sure_bg = cv2.dilate(opening,kernel,iterations=3)
	# Finding sure foreground area
	dist_transform = cv2.distanceTransform(opening,cv2.cv.CV_DIST_L2,5)
	ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
	if wanna_plot:
		cv2.imshow("porn", sure_fg)
		cv2.waitKey(0)
	else:
		return sure_fg
	"""
	# Finding unknown region
	sure_fg = np.uint8(sure_fg)
	unknown = cv2.subtract(sure_bg,sure_fg)

	# Marker labelling
	ret, markers = cv2.connectedComponents(sure_fg)
	# Add one to all labels so that sure background is not 0, but 1
	markers = markers+1
	# Now, mark the region of unknown with zero
	markers[unknown==255] = 0
	# finally
	markers = cv2.watershed(img,markers)
	img[markers == -1] = [255,0,0]
	"""

def stupid_watershed(img):
	#img = util.invert(doggy)
	distance = ndi.distance_transform_edt(img)
	local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)),
							labels=img)

	markers = ndi.label(local_maxi)[0]
	labels = watershed(-distance, markers, mask=img)

	fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True,
							 subplot_kw={'adjustable': 'box-forced'})
	ax = axes.ravel()

	ax[0].imshow(img, cmap=plt.cm.gray, interpolation='nearest')
	ax[0].set_title('Overlapping objects')
	ax[1].imshow(-distance, cmap=plt.cm.gray, interpolation='nearest')
	ax[1].set_title('Distances')
	ax[2].imshow(labels, cmap=plt.cm.spectral, interpolation='nearest')
	ax[2].set_title('Separated objects')

	for a in ax:
		a.set_axis_off()

	fig.tight_layout()
	plt.show()

ballz = greyscale(io.imread("Figures/blob_test.png"))
# simba = cv_watershed_markers("Figures/blob_test.png")






