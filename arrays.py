"""The first step is to read data from the image into python using gdal and numpy. This is done by creating
a gdal Dataset with gdal.Open(), then reading data from each of the four bands in the  image (red, green,
blue, and near-infrared). The code below give the specifics of the process."""

import numpy as np
import gdal
from skimage import exposure
from skimage.segmentation import slic
import time

# loading image
s_image = "PL_PS_20200723T0742_ALL_Tile_0_0_qKSm9prB.tif"


driverTiff = gdal.GetDriverByName('GTiff')
# open s_image dataset
s_image_ds = gdal.Open(s_image)
# getting number of bands
nbands = s_image_ds.RasterCount
band_data = []


# appending each raster band to the list
for i in range(1, nbands + 1):
    band = s_image_ds.GetRasterBand(i).ReadAsArray()
    band_data.append(band)
band_data = np.dstack(band_data)
print(band_data)

"""
# ########

img = exposure.rescale_intensity(band_data)

# do segmentation multiple options with quickshift and slic
seg_start = time.time()
segments = slic(img, n_segments=500000, compactness=0.1)
print('segments complete', time.time() - seg_start)

# save segments to raster
segments_fn = 'C:/Users/user/Desktop/remote sensing/main file/AUTOMATIC-ANALYSIS-OF-BUILDING-FOOTPRINTS-WITH-VERY-HIGH-RESOLUTION-SATELLITE-IMAGERY/segments.tif'
segments_ds = driverTiff.Create(segments_fn, s_image_ds.RasterXSize, s_image_ds.RasterYSize,
                                1, gdal.GDT_Float32)
segments_ds.SetGeoTransform(s_image_ds.GetGeoTransform())
segments_ds.SetProjection(s_image_ds.GetProjectionRef())
segments_ds.GetRasterBand(1).WriteArray(segments)
segments_ds = None"""
