from Pipelines.process_raw import (
    process_analytical_table,
    process_shapefile,
    process_target,
    process_census,
    process_county_shapefile,
    process_neighbors_dict,
    process_index_to_tile,
    process_tile_to_pixels,
)
from config import RASTER, TILE_TO_PIXELS, MAP_INDEX_TILES, JSON_NEIGHBORS
import numpy as np
import rasterstats as rs
import rasterio as rio
import json
import pandas as pd
import geopandas as gpd
import re
from numpy import save


def raster_to_dict() -> dict:
    """

    :return: Returns a dictionary having the tiles -from the shapefile- as keys and a (33,33)
    array of the pixels -from the raster file- sitting within those tiles as values
    """

    tiles_to_pixels = {}
    shapefile = process_shapefile()

    for index, shape in shapefile.iterrows():
        tile_h = shape['tile_h']
        tile_v = shape['tile_v']
        primary_key = str((tile_h, tile_v))
        polygon = shape['geometry']
        with rio.open(RASTER) as stack_src:
            array = stack_src.read(1)
            affine = stack_src.transform
            raster_chunk = rs.zonal_stats(polygon, array, affine=affine, raster_out=True)
            pixel_array = np.ma.getdata(raster_chunk[0]['mini_raster_array'])
        tiles_to_pixels[primary_key] = pixel_array.tolist()

    return tiles_to_pixels


def get_tile_from_string(key: str):
    """
    This is used to get back the tiles from their hashable -string- format
    from the dict output of the raster_to_dict function

    :param key: a string which contains integers
    :return: a list of the integers in the string
    """
    return [int(s) for s in re.findall(r'\d+', key)]


def write_dict_to_npy():
    """
    Dumps the output of the raster_to_dict function into memory.
    It's too hardware heavy to do it every time we need the data

    """
    dat = raster_to_dict()
    df = pd.DataFrame(columns=['tile_h', 'tile_v', 'pixels'])
    for key, value in dat.items():
        df_tmp = pd.DataFrame(
            {'tile_h': [get_tile_from_string(key)[0]], 'tile_v': [get_tile_from_string(key)[1]],
             'pixels': [np.array(value)]})
        df = df.append(df_tmp, ignore_index=True)

    save(TILE_TO_PIXELS, df.to_numpy())


def census_urban_pop_rate() -> pd.DataFrame:
    """

    :return: Census data with changed column names
    """
    df = process_census()
    df['Total'], df['Total!!Urban'] = df['Total'].astype(float), df['Total!!Urban'].astype(float)
    df['urban_pop_rate'] = df['Total!!Urban'] / df['Total']
    df.columns = ['GEOID', 'housing_units', 'urban_housing_units', 'urban_housing_units_rate']
    return df


def merge_census_shapefile() -> gpd.geodataframe.GeoDataFrame:
    """

    :return: Merged census data with its respective shapefile
    """
    shapefile_cd = process_county_shapefile()
    df = census_urban_pop_rate()
    merged = shapefile_cd.merge(df, how='inner', on='GEOID')

    return merged


def merge_analytical_shapefile() -> gpd.geodataframe.GeoDataFrame:
    """

    :return: Merged analytical table with its respective shapefile
    """
    analytical_table = process_analytical_table()
    shapefile = process_shapefile()

    geo_analytical_table = shapefile.merge(analytical_table, how='inner', on=['tile_h', 'tile_v'])

    return geo_analytical_table


def merge_analytical_census() -> gpd.geodataframe.GeoDataFrame:
    """

    :return: Spatially joined analytical table with external census table
    after projecting respective geometries for the values in the latter table
    to match the tiles of the former table
    """
    geo_census = merge_census_shapefile()
    geo_analytical = merge_analytical_shapefile()
    geo_census = geo_census.to_crs(geo_analytical.crs)
    result_df = gpd.sjoin(geo_analytical, geo_census, how="left", op='intersects')
    result_df = result_df.reset_index().drop_duplicates(subset=['tile_h', 'tile_v'], keep="first")

    return result_df


def get_enhanced_analytical_table() -> gpd.geodataframe.GeoDataFrame:
    """
    Removes useless column from output of merge_analytical_census

    :return:
    Returns an enhanced ready-to-use dataframe for Analysis
    """
    df = merge_analytical_census()
    df = df.drop(['index_right', 'GEOID', 'block_size', 'index'], axis=1)
    return df


def merged_with_target() -> gpd.geodataframe.GeoDataFrame:
    """
    Merges our final enhanced analytical table with the targets
    :return: DataFrame with targets
    """

    df = get_enhanced_analytical_table()
    target = process_target()
    return df.merge(target, how='left', on=['tile_h', 'tile_v'])


def tiles_to_neighbors():
    """
    Using the shapefile to link every tile with their neighbor tiles
    Creates a dictionary that has index of tiles as keys and list of
    indexes as values
    Takes a long time to run, so output is written to memory for further use
    """
    df = merged_with_target()
    df.loc[:, ['tile_h', 'tile_v']].to_csv(MAP_INDEX_TILES)
    index_to_neighbors = {}
    for index, tile in df.iterrows():
        neighbors = df.loc[df.has_target == 1].loc[
            ~(df.loc[df.has_target == 1].geometry.disjoint(tile.geometry))].index
        index_to_neighbors[index] = neighbors.to_list()

    with open(JSON_NEIGHBORS, 'w', encoding='utf-8') as f:
        json.dump(index_to_neighbors, f, ensure_ascii=False, indent=4)


def merge_ind_w_table(df: pd.DataFrame) -> pd.DataFrame:
    ind = process_index_to_tile()
    df = df.merge(ind, how='inner', on=['tile_h', 'tile_v'])
    return df


def append_nones(length, list_):
    """
    Appends Nones to list to get length of list equal to `length`.
    If list is too long raise AttributeError
    """
    diff_len = length - len(list_)
    if diff_len <= 0:
        return list_
    return list_ + [None] * diff_len


def add_neighbors_variables(a_table: pd.DataFrame, flatten: str):
    """
    Adding our synthetic variables: neighbor1, neighbor2, and neighbor3

    :param a_table: Dataframe to which we want to add the neighbor variables
    (Either the pixel arrays or the analytical table)
    :param flatten: Variable which we want to observe for the neighbors
    (Either 'pixels' or 'target'
    :return: Dataframe enhanced with neighbor's outcome as features
    """
    tn = process_neighbors_dict()
    neighbors_df = pd.DataFrame(columns=['neighbor1', 'neighbor2', 'neighbor3'])
    df = merge_ind_w_table(a_table)
    df = df.set_index('ind')
    for index, row in df.iterrows():
        values = append_nones(3, list(df.loc[tn[index], :][flatten].values))
        a_series = pd.Series(values[:3],
                             index=neighbors_df.columns)
        neighbors_df = neighbors_df.append(a_series, ignore_index=True)

    result = pd.concat([df, neighbors_df], axis=1, sort=False)
    return result


def pixel_for_NN():
    """
    Gathers all the information needed about tiles and pixels
    to feed to a neural network (including targets)
    :return: DataFrame that contains the np.arrays for the NN
    """
    df_pixels = process_tile_to_pixels()
    df = add_neighbors_variables(df_pixels, 'pixels')
    df.neighbor1 = df.neighbor1.fillna(method='ffill')
    df.neighbor2 = df.neighbor2.fillna(df.neighbor1)
    df.neighbor3 = df.neighbor3.fillna(df.neighbor2)
    target = process_target()
    return df.merge(target, how='inner', on=['tile_h', 'tile_v'])


def table_for_model():
    """
    Makes the analytical table "ML-ingestable" by removing all NAs
    :return: DataFrame for modeling
    """
    df_target = merged_with_target()
    df = add_neighbors_variables(df_target, 'target')
    df.neighbor1 = df.neighbor1.fillna(0)
    df.neighbor2 = df.neighbor2.fillna(df.neighbor1)
    df.neighbor3 = df.neighbor3.fillna(df.neighbor2)
    df.loc[:, ['housing_units', 'urban_housing_units', 'urban_housing_units_rate']] = df[
        ['housing_units', 'urban_housing_units', 'urban_housing_units_rate']].fillna(method='ffill')
    df.loc[:, ['housing_units', 'urban_housing_units', 'urban_housing_units_rate']] = df[
        ['housing_units', 'urban_housing_units', 'urban_housing_units_rate']].fillna(0)
    return df


def split(df: pd.DataFrame):
    """

    :return: Two dataframes, with and without target
    respectively train and test sets
    """
    return df.loc[~df.target.isna(), :], df.loc[df.target.isna(), :]
