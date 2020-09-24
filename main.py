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
#             r"WITH-VERY-HIGH-RESOLUTION-SATELLITE-IMAGERY\PL_PS_20200723T0742_ALL_Tile_0_0_qKSm9prB.tif"

raster_path = workspace_path + "/firstTrialYola2.tif"
              
dataset = gdal.Open(raster_path, gdal.GA_ReadOnly)
numpy_array = dataset.ReadAsArray().astype(numpy.float)
nbands = dataset.RasterCount

# Getting Dataset Information
print("Driver: {}/{}".format(dataset.GetDriver().ShortName, dataset.GetDriver().LongName))
print("Size is {} x {} x {}".format(dataset.RasterXSize,
                                    dataset.RasterYSize,
                                    dataset.RasterCount))


#              Histogram for band 1
# Read data
data1a = dataset.GetRasterBand(1).ReadAsArray()
array_band1a = numpy.array(data1a)
# Clean zeros from array, transform to vector
nonzero_vector_band1 = array_band1a[numpy.nonzero(array_band1a)]
# plt.hist(nonzero_vector_band1, bins=107500)
# plt.show()

# create an empty array, each column of the empty array will hold one band of data from the image
# loop through each band in the image nad add to the data array
data = np.empty((dataset.RasterXSize*dataset.RasterYSize, nbands))
for i in range(1, nbands+1):
    band = dataset.GetRasterBand(i).ReadAsArray()
    data[:, i-1] = band.flatten()
print(data.shape)

# set up the kmeans classification, fit, and predict
km = KMeans(n_clusters=6)
km.fit(data)
km.predict(data)

# format the predicted classes to the shape of the original image
out_data = km.labels_.reshape((dataset.RasterYSize, dataset.RasterXSize))
print(out_data.shape)

# displaying the output
plt.figure(figsize=(20, 20))
plt.imshow(out_data, cmap="hsv")
plt.show()
