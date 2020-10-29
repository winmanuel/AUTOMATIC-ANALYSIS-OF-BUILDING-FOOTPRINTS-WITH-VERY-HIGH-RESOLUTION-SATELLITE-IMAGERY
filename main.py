""" Bottom-up microsimulation of object-detection
Authors: Godwin Emmanuel, Ervin Wirth
 No rights reserved """

import numpy
from sklearn.cluster import KMeans
import gdal
import numpy as np
import matplotlib.pyplot as plt
import os

workspace_path = os.path.dirname(__file__) 
 
# raster_path = r"C:\Users\user\Desktop\remote sensing\main file\AUTOMATIC-ANALYSIS-OF-BUILDING-FOOTPRINTS-" \
#             r"firstTrialYola22.tif"

raster_path = workspace_path + "/finalimage2.tif"

dataset = gdal.Open(raster_path, gdal.GA_ReadOnly)
numpy_array = dataset.ReadAsArray().astype(numpy.float)
nbands = dataset.RasterCount

# Getting Dataset Information
print("Driver: {}/{}".format(dataset.GetDriver().ShortName, dataset.GetDriver().LongName))
print("Size is {} x {} x {}".format(dataset.RasterXSize,
                                    dataset.RasterYSize,
                                    dataset.RasterCount))

# create an empty array, each column of the empty array will hold one band of data from the image
# loop through each band in the image and add to the data array
data = np.empty((dataset.RasterXSize*dataset.RasterYSize, nbands))
for i in range(1, nbands+1):
    band = dataset.GetRasterBand(i).ReadAsArray()
    data[:, i-1] = band.flatten()
print(data.shape)

# -------------------------------------
#              Histogram for band 1
# Read data
data1 = dataset.GetRasterBand(1).ReadAsArray()
array_band1 = numpy.array(data1)
# Clean zeros from array, transform to vector
nonzero_vector_band1 = array_band1[numpy.nonzero(array_band1)]
plt.hist(nonzero_vector_band1, bins=100)
plt.title('Red band')
plt.ylabel('Values')
plt.show()

#              Histogram for band 2
# Read data
data2 = dataset.GetRasterBand(2).ReadAsArray()
array_band2 = numpy.array(data2)
# Clean zeros from array, transform to vector
nonzero_vector_band2 = array_band2[numpy.nonzero(array_band2)]
plt.hist(nonzero_vector_band2, bins=100)
plt.title('Green band')
plt.ylabel('Values')
plt.show()

#              Histogram for band 3
# Read data
data3 = dataset.GetRasterBand(3).ReadAsArray()
array_band3 = numpy.array(data3)
# Clean zeros from array, transform to vector
nonzero_vector_band3 = array_band3[numpy.nonzero(array_band3)]
plt.hist(nonzero_vector_band1, bins=100)
plt.title('Blue band')
plt.ylabel('Values')
plt.show()


# set up the kmeans classification, fit, and predict
km = KMeans(n_clusters=4)
km.fit(data)
km.predict(data)

# format the predicted classes to the shape of the original image
out_data = km.labels_.reshape((dataset.RasterYSize, dataset.RasterXSize))
print(out_data.shape)

# displaying the output
plt.figure(figsize=(20, 20))
plt.imshow(out_data, cmap="hsv")
plt.show()
