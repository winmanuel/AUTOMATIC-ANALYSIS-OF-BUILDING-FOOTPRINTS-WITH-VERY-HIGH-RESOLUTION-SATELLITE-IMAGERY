from osgeo import gdal, ogr, osr
import numpy as np

## osgeo has to be installed fisrt
## Download a proper GDAL wheel file (.whl) from here:
## https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal
## After install it with pip: pip install path_to_wheelfile.whl
## Then the 'osgeo import' shall work

raster_path = "d:/git/building-footprints/PL_PS_20200723T0742_ALL_Tile_0_0_qKSm9prB.tif"
datasrc = gdal.Open(raster_path)
numpy_array = datasrc.ReadAsArray().astype(np.float)

## Try to some statistics first, e.g. with numpy