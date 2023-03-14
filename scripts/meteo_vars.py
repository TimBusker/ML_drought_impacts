# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 14:55:34 2023

@author: tbr910
"""

import os 
import xarray as xr
import xclim as xc
from datetime import datetime
from xclim.indices import standardized_precipitation_index
import matplotlib.pyplot as plt
os.chdir('C:\\Users\\tbr910\\Documents\\Forecast_action_analysis')
from function_def import *
import pandas as pd
import numpy as np


path='C:/Users/tbr910/Documents/Forecast_action_analysis'
CHIRPSV2_HAD=os.path.join(path, 'CHIRPS025/HAD') 


r_indicators= ['rainfall_totals','wet_days', 'dry_spells']#, 

#%%% pre-process rainfall 
os.chdir(CHIRPSV2_HAD)
P_HAD=xr.open_dataset('chirps-v2_ALL_YEARS_sub_HAD_ORIGINAL.nc')#chirps-v2_ALL_YEARS_sub_HAD_NEW
P_HAD=P_HAD.rename(band_data='tp') #precip 
P_HAD_DN=P_HAD.where(P_HAD['tp']!=-9.999e+03) ## save -9999 values as NAN
P_HAD_DN=P_HAD_DN.where(P_HAD_DN['tp']<1e+10) ## delete all high values (becomes nan). ASK VICKY WHERE THESE HIGH VALUES COME FROM! 
P_HAD_DN=P_HAD_DN.where(P_HAD_DN['tp']>-2000) #Delete very small rainfall values

#%%% Calculate monthly rainfall per county 

#%%% land mask 
# land mask
P_HAD_DN=P_HAD_DN.assign(total_rainfall=(P_HAD_DN.tp.sum(dim=['time']))) ## only to mask areas that never get rainfall 
land_mask=P_HAD_DN['total_rainfall'].where(P_HAD_DN['total_rainfall'] ==0, 1)
P_HAD_DN=P_HAD_DN.where(land_mask==1, np.nan).drop('total_rainfall')

#%%% wet days 

wd=wet_days(P_HAD_DN, 1,'MS')
wd=wd.where(land_mask==1, np.nan) ## land mask again
wd=wd.rename(tp='wd')

ds=max_dry_spells(P_HAD_DN, 1,'MS')
ds=ds.where(land_mask==1, np.nan) ## land mask again
ds=ds.rename(tp='ds')



#%%% resample tp to monthly
P_monthly=P_HAD_DN.resample(time='MS').sum() ## monthly 

#%% Load counties 
county_sf=gpd.read_file('C:/Users/tbr910/Documents/Forecast_action_analysis/vector_data/geo_boundaries/Kenya/County.shp')
county_sf.set_index('OBJECTID', inplace=True)
county_raster=rasterize_shp(county_sf,P_HAD_DN, 'upscaled', 2) 
county_sf.rename(columns={'COUNTY':'county'}, inplace=True)


counties=['Garissa','Isiolo','Mandera','Marsabit','Samburu','Tana River','Turkana','Wajir','Baringo','Kajiado','Kilifi','Kitui','Laikipia','Makueni','Meru','Taita Taveta','Tharaka','West Pokot','Lamu','Nyeri','Narok']

#%% load impact data 
os.chdir('C:\\Users\\tbr910\\Documents\\Forecast_action_analysis\\impacts')
file='impacts_county_level.xlsx'
impact_df= pd.read_excel(file, index_col=0) # note that this is slightly different from the data submitted in previous paper. Some gaps in impact data are filled. Note that recent months are not used, because rain data goes untill 01/01/2021
drop_cols=['Rain Season','NDVI_P', 'NDVI_P_range','NDVI_P_crop', 'OBJECTID', 'SP', 'CAM_P', 'BP', 'month']+[i for i in impact_df.columns if '_IP' in i]
impact_df = impact_df.drop(columns=drop_cols)
#impact_df.reset_index(inplace=True)
#impact_df.rename({'NDVI_P':'NDVI_IP', 'NDVI_P_range':'NDVI_IP_range', 'NDVI_P_crop':'NDVI_IP_crop'}, axis=1, inplace=True)


#%% append rainfall indicators to impact dataframe 

input_data=pd.DataFrame()
for county in counties:
    print ('county=', county)
    impact_df_county=impact_df[impact_df.county==county]
    ######################################################## CHIRPS rainfall per county ###########################################        
    ##### make a mask 
    longitude = P_monthly.longitude.values
    latitude = P_monthly.latitude.values
    mask = regionmask.mask_geopandas(county_sf,longitude,latitude)
    mask=mask.rename({'lon': 'longitude','lat': 'latitude'})
            
    ID=county_sf.index[county_sf['county']==county].tolist()
    ID = int(ID[0])
    
    #%%% rainfall 
    rainfall_county= P_monthly.where(mask==ID, np.nan) 
    tp_county_mean=rainfall_county.mean(dim=('latitude', 'longitude'))
    tp_county_mean=tp_county_mean.to_dataframe()
    


    
    #%%% wet days 
    wd_county= wd.where(mask==ID, np.nan) 
    wd_county_mean=wd_county.mean(dim=('latitude', 'longitude'))    
    wd_county_mean=wd_county_mean.to_dataframe()
    
    #%%% Dry spells 
    ds_county= ds.where(mask==ID, np.nan) 
    ds_county_mean=ds_county.mean(dim=('latitude', 'longitude'))    
    ds_county_mean=ds_county_mean.to_dataframe()
    
    
    #%%% SPI 
    index='SPI'
    accumulation= [1,3,6,12,24]
    spi_dataframe=pd.DataFrame(index=tp_county_mean.index) #see slack marthe for pixel level spi 
    

    for a in accumulation:
        accumulateddata=moving_sum(tp_county_mean.values, a)      
        best_distributions=calculate_index(accumulateddata, np.arange(1981,2022), np.arange(1981,2022), np.arange(0,12), 'SPI',a)[1:2][0]
        spi_dataframe['SPI'+str(a)]=calculate_index(accumulateddata, np.arange(1981,2022), np.arange(1981,2022), np.arange(0,12), 'SPI',a)[0:1][0] #accumulateddata[12:], index


    
    
    #%%% Merge impacts and rainfall vars
    merged_df=pd.merge(impact_df_county, tp_county_mean, left_index=True, right_index=True, how='inner')
    merged_df=pd.merge(merged_df, spi_dataframe, left_index=True, right_index=True, how='inner')
    merged_df=pd.merge(merged_df, wd_county_mean, left_index=True, right_index=True, how='inner')
    merged_df=pd.merge(merged_df, ds_county_mean, left_index=True, right_index=True, how='inner')
    
    input_data=pd.concat([input_data,merged_df])    

input_data.insert(0, 'county', input_data.pop('county'))
input_data.to_excel('C:\\Users\\tbr910\\Documents\\ML\\input_data.xlsx',freeze_panes=(1,2))


# Save dataset
#ds.to_netcdf('spi.nc')