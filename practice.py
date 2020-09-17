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
