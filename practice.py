"""k-means second trials"""
import numpy as np
from osgeo import gdal, gdal_array
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


raster_path = r"C:\Users\user\Desktop\remote sensing\main file\AUTOMATIC-ANALYSIS-OF-BUILDING-FOOTPRINTS-" \
            r"WITH-VERY-HIGH-RESOLUTION-SATELLITE-IMAGERY\PL_PS_20200723T0742_ALL_Tile_0_0_qKSm9prB.tif"
# Read in raster image
img_ds = gdal.Open(raster_path, gdal.GA_ReadOnly)


img = np.zeros((img_ds.RasterYSize, img_ds.RasterXSize, img_ds.RasterCount),
               gdal_array.GDALTypeCodeToNumericTypeCode(img_ds.GetRasterBand(1).DataType))

for b in range(img.shape[2]):
    img[:, :, b] = img_ds.GetRasterBand(b + 1).ReadAsArray()

new_shape = (img.shape[0] * img.shape[1], img.shape[2])
print(img.shape)

print(new_shape)

X = img[:, :, :4].reshape(new_shape)

print(X.shape)

km = KMeans(n_clusters=8)
km.fit(X)
km.predict(X)

X_cluster = km.labels_
X_cluster = X_cluster.reshape(img[:, :, 0].shape)

print(len(X_cluster))


plt.figure(figsize=(20, 20))
plt.imshow(X_cluster, cmap="hsv")

plt.show()

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