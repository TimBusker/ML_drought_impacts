
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

################## load CHIRPS and extract lon lats 
os.chdir('C:/Users/tbr910/Documents/Forecast_action_analysis/CHIRPS025/HAD')
rainfall= xr.open_dataset('chirps-v2_ALL_YEARS_sub_HAD_ORIGINAL.nc').load()#chirps-v2_ALL_YEARS_sub_HAD_NEW
lon=rainfall.longitude.values
lat=rainfall.latitude.values

################## downscale grid
upscale_factor = 2
new_width = rainfall.rio.width * upscale_factor
new_height = rainfall.rio.height * upscale_factor
rainfall_mask=rainfall.rio.write_crs(4326, inplace=True)
xds_upsampled = rainfall.rio.reproject(
    rainfall_mask.rio.crs,
    shape=(new_height, new_width),
    resampling=Resampling.bilinear,
)

lon_mask=xds_upsampled.x.values
lat_mask=xds_upsampled.y.values

################################################### Clustering based on livelihoods ###################################################
##################################### 
os.chdir('C:\\Users\\tbr910\\Documents\\Forecast_action_analysis\\vector_data\\livelihood_zones_erin')
#ap_mask
ap_mask = gpd.read_file('livelihood_ap_2.shp')#.set_index("FNID") 
ap_mask = regionmask.mask_geopandas(ap_mask,lon_mask,lat_mask)
   
ap_mask=ap_mask.to_dataset()
ap_mask=ap_mask.where(ap_mask.mask>=0, -9999)
ap_mask=ap_mask.where(ap_mask.mask==-9999, 1)        
ap_mask=ap_mask.where(ap_mask.mask==1,0)   
ap_mask=ap_mask.rename({'lon': 'longitude','lat': 'latitude'})  
#p_mask
p_mask = gpd.read_file('livelihood_p.shp')#.set_index("FNID") 
p_mask = regionmask.mask_geopandas(p_mask,lon_mask,lat_mask)
#p_mask=p_mask.rename({'lon': 'longitude','lat': 'latitude'})     
p_mask=p_mask.to_dataset()
p_mask=p_mask.where(p_mask.mask>=0, -9999)
p_mask=p_mask.where(p_mask.mask==-9999, 1)        
p_mask=p_mask.where(p_mask.mask==1,0) 
p_mask=p_mask.rename({'lon': 'longitude','lat': 'latitude'})  

#other_mask
other_mask = gpd.read_file('livelihood_other.shp')#.set_index("FNID") 
other_mask = regionmask.mask_geopandas(other_mask,lon_mask,lat_mask)
#other_mask=other_mask.rename({'lon': 'longitude','lat': 'latitude'})     
other_mask=other_mask.to_dataset()
other_mask=other_mask.where(other_mask.mask>=0, -9999)
other_mask=other_mask.where(other_mask.mask==-9999, 1)        
other_mask=other_mask.where(other_mask.mask==1,0)         
other_mask=other_mask.rename({'lon': 'longitude','lat': 'latitude'})  

#ap-p mask (NOT USED)
app_mask= other_mask.where(other_mask.mask==0, np.nan) 
app_mask= app_mask.where(app_mask.mask!=0, 1)         
app_mask= app_mask.where(app_mask.mask==1, 0)         
#app_mask= app_mask.where(land_mask==1, 0)     

################################################### Counties GDF ###################################################
county_sf=gpd.read_file('C:/Users/tbr910/Documents/Forecast_action_analysis/vector_data/geo_boundaries/Kenya/County.shp')
county_sf=county_sf.set_index('OBJECTID')
county_sf.rename(columns={'COUNTY':'county'}, inplace=True)  


county_raster=rasterize_shp(county_sf,p_mask, 'as_input',2) # just a mask for the counties
county_raster=county_raster.rename({'lon': 'longitude','lat': 'latitude'})
lh_df= pd.DataFrame()


counties=['Garissa','Isiolo','Mandera','Marsabit','Samburu','Tana River','Turkana','Wajir','Baringo','Kajiado','Kilifi','Kitui','Laikipia','Makueni','Meru','Taita Taveta','Tharaka','West Pokot','Lamu','Nyeri','Narok']


for county in counties:
    print (county)
    ID=county_sf.where(county_sf['county']==county).dropna(how='all').index.values[0]
    lh_p_county=p_mask.where(county_raster==ID)
    lh_ap_county=ap_mask.where(county_raster==ID)
    lh_other_county=other_mask.where(county_raster==ID)
    
    p=float(lh_p_county.mean(dim=('latitude', 'longitude')).mask.values)
    ap=float(lh_ap_county.mean(dim=('latitude', 'longitude')).mask.values)
    other=float(lh_other_county.mean(dim=('latitude', 'longitude')).mask.values)
    lh_df.loc[county,'p']=p
    lh_df.loc[county,'ap']=ap
    lh_df.loc[county,'other']=other

# make new column which represents the column with the highest value
lh_df['max']=lh_df.idxmax(axis=1)

# save as csv
lh_df.to_csv('C:\\Users\\tbr910\\Documents\\ML\\clusters.csv')
