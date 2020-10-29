""" Bottom-up microsimulation of object-detection
Authors: Godwin Emmanuel, Ervin Wirth
 No rights reserved """

import numpy
from sklearn.cluster import KMeans
import gdal
import numpy as np
import matplotlib.pyplot as plt
import os
# from sklearn.metrics import silhouette_score

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


# create an empty array, each column of the empty array will hold one band of data from the image
# loop through each band in the image and add to the data array
data = np.empty((dataset.RasterXSize*dataset.RasterYSize, nbands))
for i in range(1, nbands+1):
    band = dataset.GetRasterBand(i).ReadAsArray()
    data[:, i-1] = band.flatten()
print(data.shape)
data.flatten()
print(data)


# fetch data cleans and ravels rasters. returns raveled raster
def fetch_data(raster_file, title="", null_value=-10):
    fname = raster_file
    print(fname, 0)

    # Using gdal to import raster, and convert it to numpy array
    datset = gdal.Open(raster_path, gdal.GA_ReadOnly)
    data1a = datset.GetRasterBand(1).ReadAsArray()
    j = numpy.array(data1a)

    # Data cleansing, remove the null value, in this case it's -10
    clean = lambda x: np.nan if x == null_value else x
    cleann = np.vectorize(clean)
    z = np.ravel(j)  # ravelling to make a x or y axis
    y = cleann(z)
    retval = y[~np.isnan(y)]

    # some visualization to describe the data
    plt.figure(figsize=(7, 3))
    plt.hist(retval, color='black', bins=32)
    plt.title(title)

    # return raveled data
    return retval


# path to each raster data
x_data_path = workspace_path + "/LC08_L1TP_015030_20180708_20180717_01_T1_B2.tif"
y_data_path = workspace_path + "/LC08_L1TP_015030_20180708_20180717_01_T1_B3.tif"
# raveling the data
y_data = fetch_data(y_data_path)
X_data = fetch_data(x_data_path)


plt.rcParams["font.family"] = "Times New Roman"
# Declaring the figure, and hiding the ticks' labels
fig, ax = plt.subplots(figsize=(15,8))
ax.set_yticklabels([])
ax.set_xticklabels([])
# Actually Plotting the data
plt.scatter(X_data, y_data, s=0.1, c='blue')
# Making the graph pretty and informative!
plt.title("Raster Data Scatter Plot", fontsize=28)
plt.xlabel("X-Axis Raster", fontsize=22)
plt.ylabel("Y-Axis Raster", fontsize=22)
plt.show()