import pandas as pd
import geopandas as gpd

from config import SUBMISSION_SAMPLE, ANALYTICAL, TARGET, COLUMNS_DESCRIPTION, LANDCOVER_MAP, SHAPEFILE

analytical_table = pd.read_csv(ANALYTICAL)
target = pd.read_csv(TARGET)
columns_description = pd.read_csv(COLUMNS_DESCRIPTION)
land_cover_map = pd.read_csv(LANDCOVER_MAP)
submission_sample = pd.read_csv(SUBMISSION_SAMPLE)
shapefile = gpd.read_file(SHAPEFILE)
