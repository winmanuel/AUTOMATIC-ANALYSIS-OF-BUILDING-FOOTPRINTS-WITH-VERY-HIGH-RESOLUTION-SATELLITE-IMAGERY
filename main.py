import gdal
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

""" osgeo has to be installed fisrt
Download a proper GDAL wheel file (.whl) from here:
 https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal
 After install it with pip: pip install path_to_wheelfile.whl
 Then the 'osgeo import' shall work"""

raster_path = r"C:\Users\user\Desktop\remote sensing\main file\AUTOMATIC-ANALYSIS-OF-BUILDING-FOOTPRINTS-" \
              r"WITH-VERY-HIGH-RESOLUTION-SATELLITE-IMAGERY\PL_PS_20200723T0742_ALL_Tile_0_0_qKSm9prB.tif"
dataset = gdal.Open(raster_path, gdal.GA_ReadOnly)
numpy_array = dataset.ReadAsArray().astype(np.float)

# Getting Dataset Information
print("Driver: {}/{}".format(dataset.GetDriver().ShortName, dataset.GetDriver().LongName))
print("Size is {} x {} x {}".format(dataset.RasterXSize,
                                    dataset.RasterYSize,
                                    dataset.RasterCount))
print("Projection is {}".format(dataset.GetProjection()))
geotransform = dataset.GetGeoTransform()
if geotransform:
    print("Origin = ({}, {})".format(geotransform[0], geotransform[3]))
    print("Pixel Size = ({}, {})".format(geotransform[1], geotransform[5]))

# print(numpy_array)
# for i in range(len(numpy_array)-1): #a1 is the array in question
#   plt.hist2d(numpy_array[i,:,0],numpy_array[i,0,:])
#  plt.show()

# iterating through the first dimension
# numpy_array = np.arange(4 * 380 * 370).reshape(4, 380, 370)

data = dataset.GetRasterBand(1).ReadAsArray()
array_band1 = np.array(data)
plt.hist(array_band1)
plt.show()


"""fig = plt.figure()
ax = fig.gca(projection='3d')

for x in range(len(numpy_array[:, 0, 0])):
    for y in range(len(numpy_array[0, :, 0])):
        for z in range(len(numpy_array[0, 0, :])):
            ax.scatter(x, y, z, c=tuple([numpy_array[x, y, z], numpy_array[x, y, z], numpy_array[x, y, z], 1]))
plt.show()"""



"""print("------------------------------------------------")
band = dataset.GetRasterBand(1)
print("Band Type={}".format(gdal.GetDataTypeName(band.DataType)))

min = band.GetMinimum()
max = band.GetMaximum()
if not min or not max:
    (min, max) = band.ComputeRasterMinMax(True)
print("Min={:.3f}, Max={:.3f}".format(min, max))

if band.GetOverviewCount() > 0:
    print("Band has {} overviews".format(band.GetOverviewCount()))

if band.GetRasterColorTable():
    print("Band has a color table with {} entries".format(band.GetRasterColorTable().GetCount()))
     
data = np.random.randn(1000)
plt.hist(data, 5)
# plt.show()
print(data.itemsize) """
# Try to some statistics first, e.g. with numpy
