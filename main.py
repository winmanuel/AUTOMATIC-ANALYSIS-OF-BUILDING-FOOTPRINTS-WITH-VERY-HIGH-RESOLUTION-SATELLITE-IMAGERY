## Bottom-up microsimulation of object-detection
## Authors: Godwin Emmanuel, Ervin Wirth
## No rights reserved

"""this is a test code"""
import gdal
import numpy
from matplotlib import pyplot as plt
#import altair as alt
#from vega_datasets import data
#import pandas as pd
import os

""" osgeo has to be installed first
Download a proper GDAL wheel file (.whl) from here:
 https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal
 After install it with pip: pip install path_to_wheelfile.whl
 Then the 'osgeo import' shall work"""

# pip install altair_viewer
# Needed if other renderer hadn't set up
alt.renderers.enable('altair_viewer')
 
workspace_path = os.path.dirname(__file__) 
 
raster_path = r"C:\Users\user\Desktop\remote sensing\main file\AUTOMATIC-ANALYSIS-OF-BUILDING-FOOTPRINTS-" \
              r"WITH-VERY-HIGH-RESOLUTION-SATELLITE-IMAGERY\PL_PS_20200723T0742_ALL_Tile_0_0_qKSm9prB.tif"

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


# for band 1
data1 = dataset.GetRasterBand(1).ReadAsArray()
array_band1 = numpy.array(data1)
# plt.hist(array_band1)
# plt.show()

# for band 2
data2 = dataset.GetRasterBand(2).ReadAsArray()
array_band2 = numpy.array(data2)
# plt.hist(array_band2)
# plt.show()

# for band 3
data3 = dataset.GetRasterBand(3).ReadAsArray()
array_band3 = numpy.array(data3)
# plt.hist(array_band3)
# plt.show()

# for band 4
data4 = dataset.GetRasterBand(4).ReadAsArray()
array_band4 = numpy.array(data4)
# plt.plot(array_band4)
# plt.show()

#data = pd.DataFrame(array_band4)
data_in_row = numpy.matrix.flatten(array_band4)
plt.hist(data_in_row, 100)
plt.figure()
plt.show()

#data = {'band4': data_in_row}
#df = pd.DataFrame(data) 

#alt.Chart(df).mark_bar().encode(alt.X("band4", bin=alt.Bin(extent=[0, 1], step=0.05)), y='count()',)

# alt.Chart(source).mark_bar().encode(
    # alt.X("IMDB_Rating:Q", bin=True),
    # y='count()',
# )

print("------------------------------------------------")
"""band = dataset.GetRasterBand(1)
print("Band Type={}".format(gdal.GetDataTypeName(band.DataType)))


min = band.GetMinimum()
max = band.GetMaximum()
if not min or not max:
    (min, max) = band.ComputeRasterMinMax(True)
print("Min={:.3f}, Max={:.3f}".format(min, max))

if band.GetOverviewCount() > 0:
    print("Band has {} overviews".format(band.GetOverviewCount()))

if band.GetRasterColorTable():
    print("Band has a color table with {} entries".format(band.GetRasterColorTable().GetCount()))"""

# Try to some statistics first, e.g. with numpy
