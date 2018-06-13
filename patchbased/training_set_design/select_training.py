# !/usr/bin/python2.7
# -*-coding:Latin-1 -*

from osgeo import gdal, gdalnumeric, ogr, osr
import numpy as np
from scipy import ndimage
from rasterProcessing import raster2array, array2raster
import os, time

# Crop raster around a given pixel i j
def cropAroundPixel(base_raster, out_crop, i, j, patch_size):

	offset = int(patch_size/2)
	list_base_raster_arr = raster2array(base_raster)
	list_raster_arr = []
	for band in range(0, len(list_base_raster_arr)):
		current_band = list_base_raster_arr[band]
		raster_arr = current_band[i-offset:i+offset+1, j-offset:j+offset+1]
		list_raster_arr.append(raster_arr)
	array2raster(base_raster, out_crop, list_raster_arr, False, True, patch_size, i, j)



# Draw random polygon of a class within layer for training
def DrawSampleTraining(base_raster, path2binaryRaster, base_name, n_training, patch_size, temp_dir, training_dir):

	# Erosion for edge problems
	ds = gdal.Open(base_raster)
	ds_arr = ds.GetRasterBand(1).ReadAsArray()
	noData_arr = ds_arr
	struct = np.ones((patch_size, patch_size), dtype=bool)
	debut = time.time()
	ds_noData_arr = ndimage.binary_erosion(noData_arr, struct)
	fin = time.time()
	#print("time to compute erosion for training selection : "+str(fin-debut))
    
	# Binary raster (class "A" / not class "A")
	ds_bin = gdal.Open(path2binaryRaster)
	geoTransform = ds_bin.GetGeoTransform()
	band = ds_bin.GetRasterBand(1)
	ds_bin_arr = band.ReadAsArray()
	ds_bin_arr[ ds_noData_arr == False ] = 0
	array2raster(path2binaryRaster, temp_dir+base_name+"/"+base_name+"_erosion.tif", [ds_bin_arr], False, False)
	
	# Number of pixels belonging to class "A"
	n_pix = np.count_nonzero(ds_bin_arr == 255)
	if n_pix > n_training*10:

		# Stride to pick a training sample
		s = int(n_pix / n_training)
		
		# Get indices of class "A" pixels
		pix_255 = np.where(ds_bin_arr == 255)

		# Keep one pixel every s pixels
		pix_training = (pix_255[0][0:pix_255[0].shape[0]:s], pix_255[1][0:pix_255[0].shape[0]:s])

		#### Run through the ROI to pick training patch, beginning upper left corner
		try:
			os.makedirs(training_dir+base_name)
		except OSError:
			pass
		for k in range (0, n_training):
			out_crop = training_dir+base_name+"/training_"+base_name+"_"+str(k+1)+".tif"
			cropAroundPixel(base_raster, out_crop, pix_training[0][k], pix_training[1][k], patch_size)

