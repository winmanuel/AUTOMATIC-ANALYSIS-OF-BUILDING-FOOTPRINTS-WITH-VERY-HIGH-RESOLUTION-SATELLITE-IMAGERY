"""this is a test code"""
from sklearn.cluster import KMeans
import gdal
import numpy as np
import matplotlib.pyplot as plt
import os

workspace_path = os.path.dirname(__file__)


raster_path = workspace_path + "/PL_PS_20200723T0742_ALL_Tile_0_0_qKSm9prB.tif"

dataset = gdal.Open(raster_path, gdal.GA_ReadOnly)
numpy_array = dataset.ReadAsArray().astype(np.float)
nbands = dataset.RasterCount
# Getting Dataset Information
print("Driver: {}/{}".format(dataset.GetDriver().ShortName, dataset.GetDriver().LongName))
print("Size is {} x {} x {}".format(dataset.RasterXSize,
                                    dataset.RasterYSize,
                                    dataset.RasterCount))

# create an empty array, each column of the empty array will hold one band of data from the image
# loop through each band in the image nad add to the data array
data = np.empty((dataset.RasterXSize*dataset.RasterYSize, nbands))
for i in range(1, nbands+1):
    band = dataset.GetRasterBand(i).ReadAsArray()
    data[:, i-1] = band.flatten()
print(data.shape)

# set up the kmeans classification, fit, and predict
km = KMeans(n_clusters=8)
km.fit(data)
km.predict(data)

# format the predicted classes to the shape of the original image
out_dat = km.labels_.reshape((dataset.RasterYSize, dataset.RasterXSize))
print(out_dat.shape)

# displaying the output
plt.figure(figsize=(20, 20))
plt.imshow(out_dat, cmap="hsv")
plt.show()
