import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage import io, color, filters as filters
from skimage.future import graph
from skimage.draw import circle
from scipy import ndimage
from math import pi
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage.measure import regionprops, label
from skimage.segmentation import slic, mark_boundaries
# from helpers import cv_watershed_markers

def circularity(area, perimeter):
	"""circularity = 4pi(area/perimeter^2). """
	# decide a threshold such as 0.9, to check if the shape is circular. For perfect circle circularity == 1.
	return 4*pi*area/(perimeter**2)

def draw_bboxes(regions, ax):
	"""draw scaled bounding box of a region"""
	for region in regions:
		# take regions with large enough areas
		if region.area >= 100:
			# draw rectangle around segmented area
			minr, minc, maxr, maxc = region.bbox
			rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
									  fill=False, edgecolor='red', linewidth=0.5)
			ax.add_patch(rect)

def draw_circle(ax, center, radius=30):
	"""draw fixed width circle around x,y coords"""
	circ = mpatches.Circle(center, radius, fill=False, edgecolor='red', linewidth=0.5)
	ax.add_patch(circ)

"""https://stackoverflow.com/questions/28242274/count-number-of-objects-using-watershed-algorithm-scikit-image"""
def bilbo():
	#"Figures/blob_test.png"
	original = io.imread("Figures/triangle.png")
	image = color.rgb2gray(original)
	# image = color.rgb2gray(io.imread("mickey.png"))
	image = image < filters.threshold_otsu(image)

	distance = ndimage.distance_transform_edt(image)

	# Here's one way to measure the number of coins directly
	# from the distance map
	# THIS LINE IS SUPER IMPORTANT: KEY TO FINDING BALL CENTERS!
	coin_centres = (distance > 0.7 * distance.max())
	print 'Number of coins (method 1):', np.max(label(coin_centres))

	# Or you can proceed with the watershed labeling
	local_maxi = peak_local_max(distance, min_distance=10, indices=False, footprint=np.ones((3, 3)),
								labels=image)


	markers, num_features = ndimage.label(local_maxi)
	labels = watershed(-distance, markers, mask=image, compactness=100)

	# ...but then you have to clean up the tiny intersections between coins
	regions = regionprops(labels)
	# regions = [r for r in regions if r.area > 1000]
	print circularity(regions[0].area, regions[0].perimeter)
	print 'Number of coins (method 2):', len(regions) - 1

	# -- new attempt based on fixed radius circles and only IMPORTANT LINE
	# assign here:
	reduced_centers = label(coin_centres)
	# plot here:
	fig, axes = plt.subplots(ncols=2, figsize=(8, 2.7))
	ax0, ax1 = axes
	ax0.imshow(original, cmap=plt.cm.gray, interpolation='nearest')
	ax0.set_title('Original')
	for region in regionprops(reduced_centers):
		(y,x) = region.centroid
		# draw_circle(ax1, region.centroid)
		circ = mpatches.Circle((x,y), radius=25, fill=False, edgecolor='red', linewidth=0.5)
		ax0.add_patch(circ)
	ax1.imshow(reduced_centers, cmap=plt.cm.jet, interpolation='nearest')
	ax1.set_title('Reduced to certain center')
	for region in regionprops(reduced_centers):
		(y,x) = region.centroid
		# draw_circle(ax1, region.centroid)
		circ = mpatches.Circle((x,y), radius=25, fill=False, edgecolor='red', linewidth=0.5)
		ax1.add_patch(circ)

	"""
	fig, axes = plt.subplots(ncols=4, figsize=(8, 2.7))
	ax0, ax1, ax2, ax3= axes

	ax0.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
	ax0.set_title('Overlapping objects')
	ax1.imshow(-distance, cmap=plt.cm.jet, interpolation='nearest')
	ax1.set_title('Distances')
	ax2.set_title("labels")
	ax2.imshow(labels, cmap=plt.cm.spectral, interpolation='nearest')
	# now try to show bounding boxes:
	ax3.set_title('Separated objects')
	ax3.imshow(labels, cmap=plt.cm.spectral, interpolation='nearest')
	draw_bboxes(regions, ax3)
	"""
	for ax in axes:
		ax.axis('off')

	fig.subplots_adjust(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0,
						right=1)
	plt.show()


bilbo()	


# -- https://vcansimplify.wordpress.com/category/python-2/

# -- fit circle with cost function: https://stackoverflow.com/questions/28281742/fitting-a-circle-to-a-binary-image


# boundaries = mark_boundaries(labels, labels, color=(255,0,100))