from osgeo import ogr

def getClasses(shapefile, field):

	ds = ogr.Open(shapefile)
	lyr = ds.GetLayer()

	ListFieldValues = []

	# Fetch all classes in field
	for feature in lyr:
		if feature.GetField(field) not in ListFieldValues:
			print(feature.GetField(field))
			ListFieldValues.append(feature.GetField(field))

	return ListFieldValues
