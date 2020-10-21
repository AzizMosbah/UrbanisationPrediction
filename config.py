DATA_PATH = "../Data"


# Data given to us as part of the hackathon by Bain & TNC
ANALYTICAL = "{0}/tabular_data/analytical_table.csv".format(DATA_PATH)
TARGET = "{0}/tabular_data/target.csv".format(DATA_PATH)
COLUMNS_DESCRIPTION = "{0}/metadata/columns_description.csv".format(DATA_PATH)
LANDCOVER_MAP = "{0}/metadata/landcover_map.csv".format(DATA_PATH)
SUBMISSION_SAMPLE = "{0}/metadata/submission_sample.csv".format(DATA_PATH)
RASTER = "{0}/wisconsin_2001.tif".format(DATA_PATH)
SHAPEFILE = "{0}/tabular_data/tiles.shp".format(DATA_PATH)

# External data
EXTERNAL_CENSUS = "{0}/external_sources/census_urban_rural.csv".format(DATA_PATH)
COUNTY_SHAPEFILE = "{0}/external_sources/county_sub_divisions.shp".format(DATA_PATH)

# Temporary NPY & Json files created to store the output of hardware heavy pipelines
TILE_TO_PIXELS = '{0}/synthetic_data/tile_to_pixels.npy'.format(DATA_PATH)
JSON_NEIGHBORS = '{0}/synthetic_data/neighbors.json'.format(DATA_PATH)

# Dataframe created to map [tile_h, tile_v] to a hashable index
MAP_INDEX_TILES = '{0}/synthetic_data/index_to_tiles.csv'.format(DATA_PATH)


