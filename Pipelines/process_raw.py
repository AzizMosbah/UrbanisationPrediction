import pandas as pd
import geopandas as gpd
from numpy import load
from config import (SUBMISSION_SAMPLE, ANALYTICAL, TARGET, COLUMNS_DESCRIPTION, LANDCOVER_MAP, SHAPEFILE,
                    EXTERNAL_CENSUS, COUNTY_SHAPEFILE, TILE_TO_PIXELS, MAP_INDEX_TILES, JSON_NEIGHBORS)
import json, codecs


def process_analytical_table():
    return pd.read_csv(ANALYTICAL)


def process_target():
    return pd.read_csv(TARGET)


def process_columns_description():
    return pd.read_csv(COLUMNS_DESCRIPTION)


def process_land_cover_map():
    return pd.read_csv(LANDCOVER_MAP)


def process_submission_sample():
    return pd.read_csv(SUBMISSION_SAMPLE)


def process_shapefile():
    return gpd.read_file(SHAPEFILE)


def process_census() -> pd.DataFrame:
    """
    Processes census data from its rawest form

    :return:
    Dataframe with formatted header,
    formatted GEOID to match ShapeFile
    Filtered for columns of interest

    """

    df = pd.read_csv(EXTERNAL_CENSUS)
    new_header = df.iloc[0]
    df.columns = new_header
    df = df[1:]
    df['id'] = df['id'].str.replace(r"^([^US]*)US*", '').astype(int)
    df = df.loc[:, ['id', 'Total', 'Total!!Urban']]
    return df


def process_county_shapefile() -> gpd.geodataframe.GeoDataFrame:
    """
    Processes raw county subdivision shapefile

    :return: geopandas dataframe with census data's
    county subdivision geometry
    """
    shapefile_cd = gpd.read_file(COUNTY_SHAPEFILE)
    shapefile_cd = shapefile_cd.loc[:, ['GEOID', 'geometry']]
    shapefile_cd.GEOID = shapefile_cd.GEOID.astype(int)
    return shapefile_cd


def process_tile_to_pixels() -> pd.DataFrame:
    """
    Processes the tile_to_pixels array loaded in memory in transformations.py
    :return: pd.DataFrame with columns tile_h, tile_v, and respective pixels
    which constitutes a (30,30) array
    """
    tile_to_pixels = load(TILE_TO_PIXELS, allow_pickle=True)
    return pd.DataFrame(tile_to_pixels, columns=['tile_h', 'tile_v', 'pixels'])


def process_index_to_tile() -> pd.DataFrame:
    return pd.read_csv(MAP_INDEX_TILES).rename(columns={"Unnamed: 0": "ind"})


def process_neighbors_dict() -> dict:
    obj_text = codecs.open(JSON_NEIGHBORS, 'r', encoding='utf-8').read()
    mydat = json.loads(obj_text)
    new_dict = {}

    for k, v in mydat.items():
        if int(k) in v:
            v.remove(int(k))
        new_dict[int(k)] = v
    return new_dict
