from Pipelines.process_raw import analytical_table, land_cover_map, shapefile, target
from config import RASTER, JSON_TMP
import numpy as np
import rasterstats as rs
import rasterio as rio
import json


def raster_to_dict() -> dict:
    """

    :return: Returns a dictionary having the tiles -from the shapefile- as keys and a (33,33)
    array of the pixels -from the raster file- sitting within those tiles as values
    """

    tiles_to_pixels = {}

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


def write_dict_to_json():
    """

    :return: writes the return of the above function as a json file into memory
    """
    with open(JSON_TMP, 'w', encoding='utf-8') as f:
        json.dump(raster_to_dict(), f, ensure_ascii=False, indent=4)



