import os 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import xarray as xr
os.chdir('C:\\Users\\tbr910\\Documents\\Forecast_action_analysis')
from function_def import *
import cartopy.crs as ccrs


# Set working directory
SM_path= 'C:\\Users\\tbr910\\Documents\\ML\\SE_data\\climate_vars\\Gleam'
county_path='C:\\Users\\tbr910\Documents\\Forecast_action_analysis\\vector_data\\geo_boundaries\\Kenya'

# load county gdf 
os.chdir(county_path)
county_gdf=gpd.read_file('county.shp')
county_gdf=county_gdf.set_index('OBJECTID')
county_gdf.rename(columns={'COUNTY':'county'}, inplace=True)  


# define counties 
counties= list(county_gdf['county'].unique().flatten())

# load SM data
os.chdir(SM_path)
SM=xr.open_dataset('SMroot_2003-2022_GLEAM_v3.7b_MO.nc')
# rename lat and lon to latutide and longitude
SM=SM.rename({'lat':'latitude','lon':'longitude'})

# extract mean SM per county
county_raster=rasterize_shp(county_gdf,SM, 'as_input',2)
# rename lat and lon to latutide and longitude
county_raster=county_raster.rename({'lat':'latitude','lon':'longitude'})

#drop nan values in county_raster xarray
county_raster=county_raster.dropna(dim='latitude', how='all')
county_raster=county_raster.dropna(dim='longitude',how='all')

counties=['Garissa']
for county in counties:
    ID=county_gdf.where(county_gdf['county']==county).dropna(how='all').index.values[0]
    SM_county=SM.where(county_raster==ID)
    # SM_county.SMroot.isel(time=200).plot()
    # plt.show()
    # plt.close()
    if county=='Garissa':
        map_proj = ccrs.LambertConformal(central_longitude=39.4, central_latitude=-0.45)
        p = SM_county.SMroot.plot(
            transform=ccrs.PlateCarree(),  # the data's projection
            col="time",
            col_wrap=1,  # multiplot settings
            aspect=SM_county.dims["longitude"] / SM_county.dims["latitude"],  # for a sensible figsize
            subplot_kws={"projection": map_proj},
        )  # the plot's projection
        for ax in p.axes.flat:
            ax.coastlines() 
            #ax.set_extent([-160, -30, 5, 75])

