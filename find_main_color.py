"""find main color of pool table under various lightings"""
from skimage import io, filters, color, draw


def main():
	color_ballz = io.imread("sc_cropped.png")

main()



# some starting point articles:
# http://www.pyimagesearch.com/2014/05/26/opencv-python-k-means-color-clustering/