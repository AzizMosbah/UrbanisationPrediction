---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.6.0
  kernelspec:
    display_name: bainhack
    language: python
    name: bainhack
---

# First Steps Into the Project
    - Reading the tabular data
    - Reading the raster file and the information it contains
    - Displaying the tiles on the raster file 


### Setting working directory

```python
import os
os.chdir('../')
```

### Modules and Libraries Needed

```python
from Pipelines.process_raw import analytical_table, target, columns_description, land_cover_map, shapefile
import pandas as pd
import numpy as np
import rasterio as rio
import geopandas as gpd
import earthpy as et
from rasterio.plot import show
import gdal
import matplotlib.pyplot as plt
import fiona
import georasters as gr
import earthpy.plot as ep
from rasterio.plot import plotting_extent
import rasterstats as rs
from rasterio.plot import show
import plotly.graph_objects as go
import plotly.offline as pyo
import plotly.graph_objs as go
# Set notebook mode to work in offline
pyo.init_notebook_mode()
```

## Tabular Data

```python
analytical_table.head()
```

```python
print("The total number of tiles on the state of Wisconsin is {0}".format(len(analytical_table)))
```

```python
target.head()
```

```python
print("We have a target for {0} of these tiles".format(len(target)))
```

```python
columns_description
```

    Some information (especially on elevation and slope of the land) appears on the tabular data but is not on the raster file 
    Let's visualize that data using the shapefile 


### Shapefile

```python
shapefile.head()
```

```python
fig, ax = plt.subplots(1, figsize=(10, 6))
ax.axis('off')
shapefile.plot(ax = ax)
```

    We merge the data of interest to the shapefile from the analytical table using the primary key [tile_h, tile_v]

```python
geo_analytical_table = shapefile.merge(analytical_table, how = 'inner', on = ['tile_h', 'tile_v'])
```

```python
fig, axs = plt.subplots(3,3, figsize=(36,36))
axs[0,0].axis('off')
axs[0,0].set_title('Tile Elevation')
geo_analytical_table.plot(column = 'elevation', ax = axs[0,0], cmap = 'hot', legend=True)
axs[0,1].axis('off')
axs[0,1].set_title('Elevation Standard Deviation')
geo_analytical_table.plot(column = 'elevation_sd', ax = axs[0,1], cmap = 'hot', legend=True)
axs[0,2].axis('off')
axs[0,2].set_title('Tile Degree of Slope')
geo_analytical_table.plot(column = 'degree_of_slope', ax = axs[0,2], cmap = 'hot', legend=True)
axs[1,0].axis('off')
axs[1,0].set_title('Standard Deviation of Degree of Slope')
geo_analytical_table.plot(column = 'degree_of_slope_sd', ax = axs[1,0], cmap = 'hot', legend=True)
axs[1,1].axis('off')
axs[1,1].set_title('Degree of Slope Minimum')
geo_analytical_table.plot(column = 'degree_of_slope_min', ax = axs[1,1], cmap = 'hot', legend=True)
axs[1,2].axis('off')
axs[1,2].set_title('Degree of Slope Maximum')
geo_analytical_table.plot(column = 'degree_of_slope_max', ax = axs[1,2], cmap = 'hot', legend=True)
axs[2,0].axis('off')
axs[2,0].set_title('Elevation Maximum')
geo_analytical_table.plot(column = 'elevation_max', ax = axs[2,0], cmap = 'hot', legend=True)
axs[2,1].axis('off')
axs[2,1].set_title('Elevation Minimum')
geo_analytical_table.plot(column = 'elevation_min', ax = axs[2,1], cmap = 'hot', legend=True)
axs[2,2].axis('off')
axs[2,2].set_title('Target Availability')
geo_analytical_table.plot(column = 'has_target', ax = axs[2,2], cmap = 'hot', legend=True)
```

```python
landcover_map
```

## Raster File


### Properties of the Raster File

```python
raster = gdal.Open(RASTER)

# Dimensions
print("This image has {0} on the x-axis and {1} on the y-axis for total image resolutio of {2}p".format(raster.RasterXSize,
                                                                                                         raster.RasterYSize,
                                                                                                         raster.RasterYSize*raster.RasterYSize))
# Number of bands
print("It is a {0}-band raster".format(raster.RasterCount))
      
band1 = raster.GetRasterBand(1)
band2 = raster.GetRasterBand(2)
```

### 2-Band Raster 
    In displaying the two bands, we can see that Band1 gives us the landcover code while Band2 is just an indicator if the pixel is on the state or outside of it

```python
with rio.open(RASTER) as DEM_src:
    fig, (axr, axg) = plt.subplots(1,2, figsize=(14,7))
    show(DEM_src.read(1, masked=True), ax = axr, title = 'Band1')
    show(DEM_src.read(2, masked=True), ax = axg, title = 'Band2')
```

We can also see that the origin of the file is at the top left corner


### Let's take a closer look at Band1
    Band1 contains the data we need since the tiles always encompass at least some "instate pixels", let us join it with "landcover_map" to look at the spatial distribution of land types across the state

```python
with rio.open(RASTER) as DEM_src:
    DEM_data = DEM_src.read(1, masked=False)
    counts = DEM_data.ravel()
    unique, counts = np.unique(counts, return_counts=True)
    unique = [landcover_map.loc[landcover_map['pixel_value'] == u ,['land_cover_name']].values[0][0] if u != 0 else "Na" for u in unique]
    dist = dict(zip(unique, counts))
    del dist['Na']
    dist = {k: v for k, v in sorted(dist.items(), key=lambda item: item[1], reverse=True)}
    x = list(dist.keys())
    y = list(dist.values())
    
    fig = go.Figure(data=[go.Bar(
                x=x, y=y,
                text=y,
                textposition='auto',
            )])
    fig.update_layout(title_text='Land cover type distribution in Wisconsin in 2001')


    fig.show()
```

### How do the tiles cover the state ?
    This is a supervised learning problem and our observations are the tiles covering the state. Let us plot these tiles on top of the raster file for visualisation purposes. For each blue square that doesn't have a target we will have to predict how many pixels will belong to one of these 4 categories in 2016:
        - developed_open_space
        - developed_low_intensity
        - developed_medium_intensity
        - developed_high_intensity

```python
county_BD= gpd.GeoDataFrame.from_file(SHAPEFILE)
county_BD.geom_type.head()

fig, ax = plt.subplots(figsize=(36, 36))

ep.plot_bands(DEM_data,
              extent=plotting_extent(DEM_src),
              cmap='Greys',
              title="GP DEM",
              scale=True,
              ax=ax)
county_BD.plot(ax=ax,
               edgecolor='blue',
              facecolor='none')
ax.set_axis_off()
plt.show()
```
