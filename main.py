# Bottom-up microsimulation of object-detection
# Authors: Godwin Emmanuel, Ervin Wirth
# No rights reserved

import gdal
import numpy
from matplotlib import pyplot as plt
import os

workspace_path = os.path.dirname(__file__) 
 
# raster_path = r"C:\Users\user\Desktop\remote sensing\main file\AUTOMATIC-ANALYSIS-OF-BUILDING-FOOTPRINTS-" \
#             r"WITH-VERY-HIGH-RESOLUTION-SATELLITE-IMAGERY\PL_PS_20200723T0742_ALL_Tile_0_0_qKSm9prB.tif"

raster_path = workspace_path + "/PL_PS_20200723T0742_ALL_Tile_0_0_qKSm9prB.tif"
              
dataset = gdal.Open(raster_path, gdal.GA_ReadOnly)
numpy_array = dataset.ReadAsArray().astype(numpy.float)

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



                    ### Histogram for band 1
# Read data
data1 = dataset.GetRasterBand(1).ReadAsArray()
array_band1 = numpy.array(data1)
# Clean zeros from array, transform to vector
nonzero_vector_band1 = array_band1[numpy.nonzero(array_band1)]
plt.hist(nonzero_vector_band1, bins=100)
plt.show()

                    ### Start the analysis

data1 = dataset.GetRasterBand(1).ReadAsArray()
array_band1 = numpy.array(data1)
data2 = dataset.GetRasterBand(2).ReadAsArray()
array_band2 = numpy.array(data2)
data3 = dataset.GetRasterBand(3).ReadAsArray()
array_band3 = numpy.array(data3)
data4 = dataset.GetRasterBand(4).ReadAsArray()
array_band4 = numpy.array(data4)

# this should be created by using indices
segmentids_initial_test = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]

for i in range(array_band1.shape[0]):
  print(i)
  for j in range(array_band1.shape[0]):
    print(j)
    test_connections(i, j)

def test_connections(i, j ):
  # test east edge
  if test_similarity(i, j, i + 1, j):
    # rewrite segmentids
  # test south-east edge
  if test_similarity(i, j, i + 1, j + 1):
    # rewrite segmentids
  # test south edge
  if test_similarity(i, j, i, j + 1):
    # rewrite segmentids

def test_similarity(c1_x, c1_y, c2_x, c2_y):
  diff_band1 = array_band1[c1_x][c1_y] - array_band1[c2_x][c2_y]
  diff_band2 = array_band2[c1_x][c1_y] - array_band2[c2_x][c2_y]
  # ... You should continue the coding
  
#numpy.savetxt(workspace_path + "/test1.txt", new_array1)