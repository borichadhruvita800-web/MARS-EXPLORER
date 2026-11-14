USGS SIM 1083 Geologic Map of Mars Renovation
June 19, 2016

The original Geologic Map of Mars by David H. Scott and Michael H. Carr (1978) was created from images acquired during the Mariner 9 television experiment.  The equatorial region, which lies between latitudes 65 south and 65 north, was mapped at 1:25,000,000 scale.  The poles, areas above and below 60 degrees latitude, were mapped at approximately 1:5,000,000 scale.  To bring this benchmark map into current mapping context it was digitized and renovated by Tenielle Gaither using the semi-controlled THEMIS Daytime IR 100 m/pixel basemap.  This was completed by georeferencing each map to the new base and adjusting the existing geologic unit interpretations based on the higher resolution imagery.  The three datasets were then merged to create one global dataset to be used in a variety of GIS applications.  

------------------------------------------

Data are saved in a feature dataset in SIM_1083/Data/SIM1083_Mariner9_25M_Mars2000.gdb/SIM1083_Mars2000
All datasets are defined using the Mars 2000 Sphere geographic coordinate system.

GeoUnits (polygon feature class)
GeoContacts (polyline feature class)
LinearFeatures (polyline feature class)

Included in the SIM_1083/Data/... folder is the spreadhseet UnitAbbr.xlsx which provides full names for units, and has been joined to the GeoUnits layers in SIM1083_Mars_Global_Marine9.mxd.

Layer files (*.lyr) storing the standard symbology for each feature type are located in .../Data/LayerFiles/.

Separate shapefiles (*.shp) for geologic data are located in the ...Data/Shapefiles/ directory.             

Coordinate systems (*.prj) used for each quadrangle are stored in ...Data/CoordinateSystems/.

Geographic Coordinate System: Mars_2000_Sphere
Angular Unit: Degree (0.0174532925199433)
Prime Meridian: Reference_Meridian (0.0)
Datum: <custom>
  Spheroid: <custom>
    Semimajor Axis: 3396190.0
    Semiminor Axis: 3396190.0
    Inverse Flattening: 0.0

Projection: Stereographic_North_Pole
False_Easting: 0.0
False_Northing: 0.0
Central_Meridian: 0.0
Standard_Parallel_1: 90.0
Linear Unit: Meter (1.0)

Projection: Stereographic_South_Pole
False_Easting: 0.0
False_Northing: 0.0
Central_Meridian: 0.0
Standard_Parallel_1: -90.0
Linear Unit: Meter (1.0)

------------------------------------------

The scanned, clipped and georeferenced original maps are included, located in the .../GeoRef_Maps/ folder.
1) SIM1083_Map_65N_65S.tif
2) SIM1083_Map_North_Pole.tif
3) SIM1083_Map_South_Pole.tif

------------------------------------------

Basemaps have not been included in this download package, but the most current base using THEMIS Daytime IR data is available through ArcGIS Online or can be downloaded at the following link:
http://astrogeology.usgs.gov/search/map/Mars/Odyssey/THEMIS-IR-Mosaic-ASU/Mars_MO_THEMIS-IR-Day_mosaic_global_100m_v12

------------------------------------------

Original reference:
Scott, David H., and Carr, Michael H. (1978). Geologic Map of Mars: U.S. Geological Survey Investigations Series I-1083, scale 1:25000000, https://pubs.er.usgs.gov/publication/i1083.

For more information contact:

Tenielle Gaither
tgaither@usgs.gov
USGS Astrogeology Science Center
Flagstaff, AZ
(928)556-7387