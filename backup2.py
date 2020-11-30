""" Bottom-up microsimulation of object-detection
Authors: Godwin Emmanuel
 No rights reserved """

import numpy
from sklearn.cluster import KMeans
import gdal
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn import cluster
from osgeo import gdal, gdal_array

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

#              Histogram for band 1
# Read data
data1a = dataset.GetRasterBand(1).ReadAsArray()
array_band1a = numpy.array(data1a)
# Clean zeros from array, transform to vector
nonzero_vector_band1 = array_band1a[numpy.nonzero(array_band1a)]
# plt.hist(nonzero_vector_band1, bins=107500)
# plt.show()
img_ds = gdal.Open(raster_path, gdal.GA_ReadOnly)


img = np.zeros((img_ds.RasterYSize, img_ds.RasterXSize, img_ds.RasterCount),
               gdal_array.GDALTypeCodeToNumericTypeCode(img_ds.GetRasterBand(1).DataType))

for b in range(img.shape[2]):
    img[:, :, b] = img_ds.GetRasterBand(b + 1).ReadAsArray()

new_shape = (img.shape[0] * img.shape[1], img.shape[2])

X = img[:, :, :3].reshape(new_shape)

print(X.shape)
# create an empty array, each column of the empty array will hold one band of data from the image
# loop through each band in the image and add to the data array
data = np.empty((dataset.RasterXSize * dataset.RasterYSize, nbands))
for i in range(1, nbands + 1):
    band = dataset.GetRasterBand(i).ReadAsArray()
    data[:, i - 1] = band.flatten()
print(data.shape)
data.flatten()

# set up the kmeans classification, fit, and predict
# km = KMeans(n_clusters=6)
# km.fit(data)
# km.predict(data)
"""

score = []

for cluster in range(1, 10):
    kmeans = KMeans(n_clusters=cluster)
    kmeans.fit(data)
    score.append(kmeans.inertia_)

plt.plot(range(1, 10), score)
plt.title('The Elbow Method')
plt.xlabel('no of clusters')
plt.ylabel('wcss')
plt.show()


# --------------------

# A list holds the silhouette coefficients for each k
silhouette_coefficients = []

# Notice you start at 2 clusters for silhouette coefficientko
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, init="k-means++", random_state=10)
    kmeans.fit(data)
    score = silhouette_score(data, kmeans.labels_)
    silhouette_coefficients.append(score)

plt.style.use("fivethirtyeight")
plt.plot(range(2, 11), silhouette_coefficients)
plt.xticks(range(2, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Coefficient")
plt.show()"""

# -----------------------
"""# format the predicted classes to the shape of the original image
out_data = kmeans.labels_.reshape((dataset.RasterYSize, dataset.RasterXSize))
print(out_data.shape)

# displaying the output
plt.figure(figsize=(20, 20))
plt.imshow(out_data, cmap="hsv")
plt.show()"""
score = []
for cluster in range(1, 10):
    kmeans = KMeans(n_clusters=cluster)
    kmeans.fit(data)
    score.append(kmeans.inertia_)

plt.plot(range(1, 10), score)
plt.title('The Elbow Method')
plt.xlabel('no of clusters')
plt.ylabel('wcss')
plt.show()

# --------------------------------------

score = []
for cluster in range(1, 10):
    kmeans = KMeans(n_clusters=cluster)
    kmeans.fit(data)
    score.append(kmeans.inertia_)

plt.plot(range(1, 10), score)
plt.title('The Elbow Method')
plt.xlabel('no of clusters')
plt.ylabel('wcss')
plt.show()

# --------------------------------------
# Run the Kmeans algorithm and get the index of data points clusters
sse = []
list_k = list(range(1, 10))

for k in list_k:
    km = KMeans(n_clusters=k)
    km.fit(data)
    sse.append(km.inertia_)

# Plot sse against k
plt.figure(figsize=(6, 6))
plt.plot(list_k, sse, '-o')
plt.xlabel(r'Number of clusters *k*')
plt.ylabel('Sum of squared distance')
plt.show()
