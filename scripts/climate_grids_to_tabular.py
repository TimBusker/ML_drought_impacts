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
import cartopy.feature as cf


# Set working directory
SM_path= 'C:\\Users\\tbr910\\Documents\\ML\\SE_data\\climate_vars\\Gleam'
county_path='C:\\Users\\tbr910\Documents\\Forecast_action_analysis\\vector_data\\geo_boundaries'


################################################### create gdf with counties ########################################################
#Kenya 
os.chdir(county_path+'\\Kenya')
county_gdf_ken=gpd.read_file('county.shp')
county_gdf_ken=county_gdf_ken.set_index('OBJECTID')
county_gdf_ken=county_gdf_ken[['COUNTY','geometry']]
county_gdf_ken.rename(columns={'COUNTY':'county'}, inplace=True)  

# Ethiopia  
os.chdir(county_path+'\\Ethiopia')
county_gdf_eth=gpd.read_file('eth_admbnda_adm2_csa_bofedb_2021.shp') # admin 1 in ethiopia is too big 
county_gdf_eth=county_gdf_eth[['ADM2_EN','geometry']]
county_gdf_eth.rename(columns={'ADM2_EN':'county'}, inplace=True)

# Somalia
os.chdir(county_path+'\\Somalia')
county_gdf_som=gpd.read_file('Som_Admbnda_Adm1_UNDP.shp')
county_gdf_som=county_gdf_som[['admin1Name','geometry']]
county_gdf_som.rename(columns={'admin1Name':'county'}, inplace=True)

# merge all gdf to one 
county_gdf=pd.concat([county_gdf_ken,county_gdf_eth,county_gdf_som], axis=0)
# give each county a unique ID as index
county_gdf['ID']=np.arange(0,len(county_gdf))
county_gdf.set_index(county_gdf.ID, inplace=True)
county_gdf.drop(columns='ID', inplace=True)

# define counties 
counties= list(county_gdf['county'].unique().flatten())

################################################### Make SM ROOTZONE/SURFACE dataset per county ########################################################
os.chdir(SM_path)



file_names= ['SMroot_2003-2022_GLEAM_v3.7b_MO.nc', 'SMsurf_2003-2022_GLEAM_v3.7b_MO.nc']

for file in file_names: 
    SM_df=pd.DataFrame()

    SM=xr.open_dataset(file)

    # rename lat and lon to latutide and longitude
    SM=SM.rename({'lat':'latitude','lon':'longitude'})

    # extract mean SM per county
    county_raster=rasterize_shp(county_gdf,SM, 'as_input',2)
    # rename lat and lon to latutide and longitude
    county_raster=county_raster.rename({'lat':'latitude','lon':'longitude'})

    #drop nan values in county_raster xarray
    county_raster=county_raster.dropna(dim='latitude', how='all')
    county_raster=county_raster.dropna(dim='longitude',how='all')



    for county in counties:
        print (county)
        ID=county_gdf.where(county_gdf['county']==county).dropna(how='all').index.values[0]
        SM_county=SM.where(county_raster==ID)

        
        # plot SM for each county
        # plt.figure()
        # map_proj = ccrs.LambertConformal(central_longitude=39.4, central_latitude=-0.45)
        # test=SM_county.SMroot.isel(time=[200,202])
        # p = test.plot(
        #     transform=ccrs.PlateCarree(),  # the data's projection
        #     col="time",
        #     col_wrap=1,  # multiplot settings
        #     aspect=SM_county.dims["longitude"] / SM_county.dims["latitude"],  # for a sensible figsize
        #     subplot_kws={"projection": map_proj},
            
        # )  # the plot's projection
        # for ax in p.axes.flat:
        #     ax.coastlines() 
        #     ax.add_feature(cf.BORDERS)
        # print (county)
        # plt.show()
        # plt.close()

        # save to SM_df
        SM_county_mean=SM_county.mean(dim=('latitude', 'longitude')) 
        # to pandas series
        SM_county_mean=SM_county_mean.to_dataframe()

        SM_county_mean=SM_county_mean.resample('MS').mean()

        if file=='SMroot_2003-2022_GLEAM_v3.7b_MO.nc':
            SM_county_mean.rename(columns={'SMroot':county}, inplace=True)
        elif file=='SMsurf_2003-2022_GLEAM_v3.7b_MO.nc':
            SM_county_mean.rename(columns={'SMsurf':county}, inplace=True)


        # date stamps are set to roughly the end (i.e. the end of the aggregation period) of the month or year, respectively
        SM_df=pd.concat([SM_df,SM_county_mean], axis=1)
    SM_df.to_excel(file+'.xlsx')
    
################################################### ########################################################
