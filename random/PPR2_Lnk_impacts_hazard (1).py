# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 16:28:29 2022

@author: roo290
"""

# In[1]

#Loading the packages needed for the script

import pandas as pd
import pickle
import os
import math
import matplotlib.pyplot as plt
import sys
import numpy as np
import netCDF4 as nc 
from netCDF4 import Dataset
from osgeo import gdal
import rasterio
import seaborn as sns
import geopandas as gpd
import earthpy as et
import earthpy.plot as ep
from shapely.geometry import mapping
import rioxarray as rxr
import xarray as xr
import functools
from scipy.stats import percentileofscore
from statistics import NormalDist
import scipy.stats as stats
from numpy import savetxt
import itertools
import regionmask
import csv 
import matplotlib.ticker as ticker
from shapely.ops import unary_union
from shapely.ops import cascaded_union
import glob 

#machine learning packages
from sklearn.linear_model import LinearRegression # to build a LR model for comparison
import plotly.graph_objects as go # for data visualization
import plotly.express as px # for data visualization 
import statsmodels.api as sm # to build a LOWESS model
from scipy.interpolate import interp1d # for interpolation of new data points
from statsmodels.nonparametric.smoothers_lowess import lowess

# from mpl_toolkits.Basemap import Basemap 
gdal.UseExceptions()
 

# In[2]

#Setting the work folder

os.chdir(r'U:\Rhoda\Surfdrive\Data\MSWEP_data\MSWEP_V280\Past\Daily')
working_directory=os.getcwd()
print(working_directory)
    
# In[8]

# PRECIPITATION
    #loading precipitation dataset in sets of 10 to reduce to time and space allocated for the data

data = xr.open_mfdataset('*.nc',chunks={"time":10})

data =data.sel(lon =slice (30,53.5), lat = slice (17,-5))#slice the data to the study area

#data.to_netcdf(r'C:\Users\roo290\surfdrive (backup)\Data\precip_combined.nc')               #convert to netcdf to enable easier regridding
# data = xr.open_dataset ('precip_combined.nc')
print(data)

#%%

#loading the shapefile for the Administrative units


fp_Adminunits = r'C:\Users\roo290\OneDrive - Vrije Universiteit Amsterdam\Data\Admin_units\HAD_Admin_units.shp'

# In[4]

# Read file using gpd.read_file()

data_Adminunits = gpd.read_file(fp_Adminunits,crs="epsg:4326")                            #Admin units for Kenya, Somalia and Ethiopia

# In[42]

#Aggregate to the catchment resolution resolution
# CALCULATE MASK for Precipitation data  

sheds_mask_poly = regionmask.Regions(name = 'sheds_mask', numbers = list(range(0,157)), names = list(data_Adminunits.ADMIN_UNIT),
                                     abbrevs = list(data_Adminunits.ADMIN_UNIT), outlines = list(data_Adminunits.geometry.values[i] for i in range(0,157)))
print(sheds_mask_poly)


mask = sheds_mask_poly.mask(data.isel(time = 0 ), lon_name = 'lon', lat_name = 'lat')

#mask.to_netcdf(r'C:\Users\roo290\surfdrive (backup)\Data\MSWEP_data\Past\mask_Sm.nc') 

lat = mask.lat.values                                                                       # the lat values contained in the mask of the regions
lon = mask.lon.values

# In[11]

#AGGREGATION TO TIMESERIES FOR PRECIPITATION

# the function aggregates the gridded data to a timeseries per catchment 
    #shed_Id is the data_sheds dataframe (the admin keys/ names) 
    #Index_Id is the indexes of data_sheds dataframe which is the region Id and ranges from 0-320
    # mask_data is the mask created from the catchment regions
    #array_data is the variable to be converted to timeseries



'''aggregating the gridded data to timeseries and converting it to a dataframe for each catchment
then appending it to a single dataframe with the HYBAS ID as header'''

region_ID = np.arange(0,157)

data_prec_Admin = pd.DataFrame(index =pd.date_range(start="1980-01-01",end="2020-12-31",freq='M'))
for index in region_ID:
    print(index)
    ID_REGION=index
   
    sel_mask = mask.where(mask == ID_REGION).values
    
    id_lon = lon[np.where(~np.all(np.isnan(sel_mask), axis=0))]
    id_lat = lat[np.where(~np.all(np.isnan(sel_mask), axis=1))]
    
    out_sel1 = data.sel(lat = slice(id_lat[0], id_lat[-1]), lon = slice(id_lon[0], id_lon[-1])).compute().where(mask == ID_REGION)
    
    plt.figure(figsize=(12,8))
    ax = plt.axes()
    out_sel1.precipitation.isel(time = 1).plot(ax = ax)
    data_Adminunits.plot(ax = ax, alpha = 0.8, facecolor = 'none')
    #plt.savefig(f'Modeloutputfolder/spei/{data_SHEDs.HYBAS_ID[ID_REGION]}.png')
        
    x = out_sel1.resample(time = '1M').sum()
    
    monthly_mean=x.precipitation.mean(dim=('lon','lat'))
    df=monthly_mean.to_dataframe()

    # df.to_excel(f'Modeloutputfolder/spei/{data_SHEDs.HYBAS_ID[ID_REGION]}.xlsx',sheet_name=f'{data_SHEDs.HYBAS_ID[ID_REGION]}',index='False', 
    #             engine='xlsxwriter')
    
    data_prec_Admin[data_Adminunits.ADMIN_UNIT[ID_REGION]]=df['precipitation'].values


#%%

#saving the dataframe to an excel file

data_prec_Admin.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/data_prec_Admin.xlsx', index='False', engine='xlsxwriter') 

# In[16]

# Choose standardized indices used in analysis
      #Standardized Precipitation Index                    : 'SPI'
      #Standardized Precipitation Evapotranspiration Index : 'SPEI' 
      #Standardized Soil Moisture Index                    : 'SSMI'
      #Standardized Streamflow Index                       : 'SSFI'

indices = ['SPEI', 'SPI','SSFI','SSMI']
indicesfull = ['Standardized Precipitation Index','Standardized Precipitation Evapotranspiration Index','Standardized Soil Moisture Index','Standardized Streamflow Index']  


# Indicate the time for which data is available
start = 1980
end   = 2020
years = end - start 

# Choose reference time period (years from first year of data)
      # this can be either the historic records or the full data series 
      # (depending on how droughts are defined and the goal of the study)
        
refstart = 1981   # ! First year of data series cannot be used ! 
refend   = 2020  

# In[17]

# function to accumulate hydrological data
   # with a the input data and b the accumulation time
   # -> the accumulated value coincidences with the position of the last value 
   #     used in the accumulation process.

def moving_sum(a, b) :
    
    cummuldata = np.cumsum(a, dtype=float)                 
    cummuldata[b:] = cummuldata[b:] - cummuldata[:-b]         
    cummuldata[:b - 1] = np.nan                                           
    
    return cummuldata

def moving_mean(a, b):

    cummuldata = np.cumsum(a, dtype=float)                 
    cummuldata[b:] = cummuldata[b:] - cummuldata[:-b]         
    cummuldata[:b - 1] = np.nan                                           
    
    return cummuldata/b
# In[18]

# function to find the best fitting statistical distribution
    # with a the reference time series to test the distributions 
    # and b the standardized index that is analysed
    # (possible distributions differ per hydrological data source)

def get_best_distribution(a, b):
    
    if b == 'SPEI':                     # Suggestions by Stagge et al. (2015) 
        dist_names = ['norm','genextreme', 'genlogistic', 'pearson3']                  
    elif b == 'SSMI':                   # Suggestions in Ryu et al. (2005)
        dist_names = ['norm','beta',  'pearson3','fisk']                               
    elif b == 'SPI' :                   # Suggestions by Stagge et al. (2015) 
        dist_names = ['norm','gamma', 'exponweib', 'lognorm']
    elif b == 'SSFI':                   # Suggestions by Vincent_Serrano et al. (2012)
        dist_names = ['exponweib','lognorm', 'pearson3', 'genextreme'] 
    else:
        print('problem finding distribution')

    # find fit for each optional distribution
    dist_results = []
    params = {}
    for dist_name in dist_names:                                                # Find distribution parameters        
        dist = getattr(stats, dist_name)
        param = dist.fit(a)
        params[dist_name] = param
        
        # Assess goodness-of-fit using Kolmogorov–Smirnov test
        D, p = stats.kstest(a, dist_name, args=param)                      # Applying the Kolmogorov-Smirnov test
        dist_results.append((dist_name, p))
  
    # find best fitting statistical distribution
    best_dist, best_p = (max(dist_results, key=lambda item: item[1]))           # Select the best fitted distribution

    return best_dist, best_p, params[best_dist]



# In[19]

# function to calculate Z values for a time series of one selected month
    # with a the data series over which the index is calculated
    # and b the standardized index that is analysed 
                            
def calculate_Zvalue(a, b):
        
    # extract reference time series
    referenceseries = a[refstart-start:refend-start]     
    
    # find fitting distribution for reference sereis
    best_dist, best_p, params = get_best_distribution(referenceseries, b)  
    
    # fit full time series over best distribution
    z = np.zeros(len(a))
    dist = getattr(stats, str(best_dist))                                   
    rv = dist(*params)         
        
    # Create suitable cummulative distribution function
    # Solve issue with zero values in Gamma distribution (cfr.Stagge et al. 2015)
    if dist == 'gamma':                                                     
        nyears_zero = len(a) - np.count_nonzero(a)
        p_zero = nyears_zero / len(a)
        p_zero_mean = (nyears_zero + 1) / (2 * (len(a) + 1))           

        ppd = (a * 0 ) + p_zero_mean
        ppd[np.nonzero(a)] = p_zero+((1-p_zero)*rv.cdf(a[np.nonzero(a)]))
       
    else:
        ppd = rv.cdf(a)
    
    # Standardize the fitted cummulative distribtuion distribution 
    z = stats.norm.ppf(ppd)                                   
    
    # limit extreme, unlikely values 
    z[z>3] = 3
    z[z<-3] = -3 
            
    return z
#%%
# dist_shed = pd.DataFrame(index=['best_dist', 'best_p', 'params'], columns = data_monthly_prec.columns)
# dist_shed = dist_shed.T
# for i,shed in enumerate(dist_shed.index):
#     best_dist, best_p, params = get_best_distribution(data_monthly_sm[shed].fillna(0), 'SSMI')
#     dist_shed.loc[shed,0]=best_dist
#     dist_shed.loc[shed,1]=best_p
#     #dist_shed.loc[shed,2]=params
# In[20]

# function to calculate standardized indices per month
    # with a the full accumulated data series over which the index is calculated
    # and b the standardized index that is analysed
        
def calculate_Index(a, b):

    indexvalues = a * np.nan 
                                
    for m in range(12):  
     
        # Extract monthly values
        monthlyvalues = a * np.nan
        for yr in range(int(len(a)/12)): 
            
            monthlyvalues[yr] = a[(12*yr)+m]                                 
                                
        # Retrieve index per month
        Zval = calculate_Zvalue(monthlyvalues,b)
                            
        # Reconstruct time series
        for yr in range(int(len(a)/12)):
            indexvalues[(12*yr)+m] = Zval[yr]
            
    return indexvalues  
        
# In[21]

# function calculates the indicator values for each catchment in a loop 
    #data_variable is the data series over which the index is calculated 
    #accumulation is the accumulation values over which the data is accumulated 
    #index is the type of indicator being calculated
  
def calculate_indicators_spi (data_variable,accumulation, index):
    indicator=pd.DataFrame(index =pd.date_range(start="1981-01-01",end="2020-12-31",freq='M')) #Creating empty dataframes for the indices calculation
    for df in data_variable:
        print(df)
        accumulateddata=moving_sum(data_variable[df].values, accumulation)  
        indicator[df]=calculate_Index(accumulateddata[12:], index)
    return indicator

# In[22]

#Standardized Precipitation Index (SPI)
    #calling the function to calculation SPI with accumulation periods between 1-24
    #the distribution applied here is gamma distribution

spi_1=calculate_indicators_spi(data_prec_Admin, 1, 'SPI')
spi_3=calculate_indicators_spi(data_prec_Admin, 3, 'SPI')
spi_6=calculate_indicators_spi(data_prec_Admin, 6, 'SPI')
spi_12=calculate_indicators_spi(data_prec_Admin, 12,'SPI')
spi_24=calculate_indicators_spi(data_prec_Admin, 24,'SPI')
spi_2=calculate_indicators_spi(data_prec_Admin, 2, 'SPI')
spi_4=calculate_indicators_spi(data_prec_Admin, 4, 'SPI')
spi_5=calculate_indicators_spi(data_prec_Admin, 5, 'SPI')
spi_7=calculate_indicators_spi(data_prec_Admin, 7,'SPI')
spi_8=calculate_indicators_spi(data_prec_Admin, 8, 'SPI')
spi_9=calculate_indicators_spi(data_prec_Admin, 9, 'SPI')
spi_10=calculate_indicators_spi(data_prec_Admin, 10, 'SPI')
spi_11=calculate_indicators_spi(data_prec_Admin, 11,'SPI')
spi_13=calculate_indicators_spi(data_prec_Admin, 13, 'SPI')
spi_14=calculate_indicators_spi(data_prec_Admin, 14, 'SPI')
spi_15=calculate_indicators_spi(data_prec_Admin, 15, 'SPI')
spi_17=calculate_indicators_spi(data_prec_Admin, 17,'SPI')
spi_18=calculate_indicators_spi(data_prec_Admin, 18, 'SPI')
spi_19=calculate_indicators_spi(data_prec_Admin, 19, 'SPI')
spi_20=calculate_indicators_spi(data_prec_Admin, 20, 'SPI')
spi_16=calculate_indicators_spi(data_prec_Admin, 16,'SPI')
spi_21=calculate_indicators_spi(data_prec_Admin, 21, 'SPI')
spi_22=calculate_indicators_spi(data_prec_Admin, 22, 'SPI')
spi_23=calculate_indicators_spi(data_prec_Admin, 23, 'SPI')

# In[23]

#saving files the indicator files to excel files

spi_1.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/spi_1.xlsx', index='False', engine='xlsxwriter') 
spi_3.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/spi_3.xlsx', index='False', engine='xlsxwriter') 
spi_6.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/spi_6.xlsx', index='False', engine='xlsxwriter') 
spi_12.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/spi_12.xlsx', index='False', engine='xlsxwriter') 
spi_24.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/spi_24.xlsx', index='False', engine='xlsxwriter') 
spi_2.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/spi_2.xlsx', index='False', engine='xlsxwriter') 
spi_4.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/spi_4.xlsx', index='False', engine='xlsxwriter') 
spi_5.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/spi_5.xlsx', index='False', engine='xlsxwriter') 
spi_7.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/spi_7.xlsx', index='False', engine='xlsxwriter') 
spi_8.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/spi_8.xlsx', index='False', engine='xlsxwriter') 
spi_9.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/spi_9.xlsx', index='False', engine='xlsxwriter') 
spi_10.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/spi_10.xlsx', index='False', engine='xlsxwriter') 
spi_11.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/spi_11.xlsx', index='False', engine='xlsxwriter') 
spi_13.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/spi_13.xlsx', index='False', engine='xlsxwriter') 
spi_14.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/spi_14.xlsx', index='False', engine='xlsxwriter') 
spi_15.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/spi_15.xlsx', index='False', engine='xlsxwriter') 
spi_16.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/spi_16.xlsx', index='False', engine='xlsxwriter') 
spi_17.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/spi_17.xlsx', index='False', engine='xlsxwriter') 
spi_18.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/spi_18.xlsx', index='False', engine='xlsxwriter') 
spi_19.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/spi_19.xlsx', index='False', engine='xlsxwriter') 
spi_20.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/spi_20.xlsx', index='False', engine='xlsxwriter') 
spi_21.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/spi_21.xlsx', index='False', engine='xlsxwriter') 
spi_22.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/spi_22.xlsx', index='False', engine='xlsxwriter') 
spi_23.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/spi_23.xlsx', index='False', engine='xlsxwriter') 

# In[38]

#P-ET -> Standardized Precipitation Evaporation Index (SPEI)
#loading the regridded p-potential evapotranspiration file (regridded to 0.1 resolution from 0.25)

data_PET = xr.open_dataset(r'U:\Rhoda\pevapotranspiration.nc')


print(data_PET)

#%%
'''aggregating the gridded data to timeseries and converting it to a dataframe for each catchment
then appending it to a single dataframe with the HYBAS ID as header'''

mask_pet = sheds_mask_poly.mask(data_PET.isel(time = 0 ), lon_name = 'lon', lat_name = 'lat')

lat_pet = mask_pet.lat.values
lon_pet = mask_pet.lon.values

region_ID = np.arange(0,3)

data_PET_Admin = pd.DataFrame(index =pd.date_range(start="1980-01-01",end="2020-12-31",freq='M'))

for index in region_ID:
    print(index)
    ID_REGION=index
   
    sel_mask_pet = mask_pet.where(mask_pet == ID_REGION).values
    
    id_lon_pet = lon_pet[np.where(~np.all(np.isnan(sel_mask_pet), axis=0))]
    id_lat_pet = lat_pet[np.where(~np.all(np.isnan(sel_mask_pet), axis=1))]
    
    out_sel1_pet = data_PET.sel(lat = slice(id_lat_pet[0], id_lat_pet[-1]), lon = slice(id_lon_pet[0], id_lon_pet[-1])).compute().where(mask_pet == ID_REGION)
    
    plt.figure(figsize=(12,8))
    ax = plt.axes()
    out_sel1_pet.pet.isel(time = 1).plot(ax = ax)
    data_Adminunits.plot(ax = ax, alpha = 0.8, facecolor = 'none')
    #plt.savefig(f'Modeloutputfolder/spei/{data_SHEDs.HYBAS_ID[ID_REGION]}.png')
        
    x_pet = out_sel1_pet.resample(time = '1M').sum()
    
    monthly_mean_pet=x_pet.pet.mean(dim=('lon','lat'))
    df_pet=monthly_mean_pet.to_dataframe()

    # df.to_excel(f'Modeloutputfolder/spei/{data_SHEDs.HYBAS_ID[ID_REGION]}.xlsx',sheet_name=f'{data_SHEDs.HYBAS_ID[ID_REGION]}',index='False', 
    #             engine='xlsxwriter')
    
    data_PET_Admin[data_Adminunits.ADMIN_UNIT[ID_REGION]]=df_pet['pet'].values
            

#%%

#saving the dataframe of the P ET timeseries into excel

data_PET_Admin.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/data_PET_Admin.xlsx', index='False', engine='xlsxwriter') 

#%%
#SPEI
    #calling the function to calculate SSFI based on the accumulation periods per catchment
    #the distribution applied is 
    
spei_1=calculate_indicators_spi(data_PET_Admin, 1, 'SPEI')
spei_3=calculate_indicators_spi(data_PET_Admin, 3, 'SPEI')
spei_6=calculate_indicators_spi(data_PET_Admin, 6, 'SPEI')
spei_12=calculate_indicators_spi(data_PET_Admin, 12,'SPEI')
spei_24=calculate_indicators_spi(data_PET_Admin, 24,'SPEI')
spei_2=calculate_indicators_spi(data_PET_Admin, 2, 'SPEI')
spei_4=calculate_indicators_spi(data_PET_Admin, 4, 'SPEI')
spei_5=calculate_indicators_spi(data_PET_Admin, 5, 'SPEI')
spei_7=calculate_indicators_spi(data_PET_Admin, 7,'SPEI')
spei_8=calculate_indicators_spi(data_PET_Admin, 8, 'SPEI')
spei_9=calculate_indicators_spi(data_PET_Admin, 9, 'SPEI')
spei_10=calculate_indicators_spi(data_PET_Admin, 10, 'SPEI')
spei_11=calculate_indicators_spi(data_PET_Admin, 11,'SPEI')
spei_13=calculate_indicators_spi(data_PET_Admin, 13, 'SPEI')
spei_14=calculate_indicators_spi(data_PET_Admin, 14, 'SPEI')
spei_15=calculate_indicators_spi(data_PET_Admin, 15, 'SPEI')
spei_17=calculate_indicators_spi(data_PET_Admin, 17,'SPEI')
spei_18=calculate_indicators_spi(data_PET_Admin, 18, 'SPEI')
spei_19=calculate_indicators_spi(data_PET_Admin, 19, 'SPEI')
spei_20=calculate_indicators_spi(data_PET_Admin, 20, 'SPEI')
spei_16=calculate_indicators_spi(data_PET_Admin, 16,'SPEI')
spei_21=calculate_indicators_spi(data_PET_Admin, 21, 'SPEI')
spei_22=calculate_indicators_spi(data_PET_Admin, 22, 'SPEI')
spei_23=calculate_indicators_spi(data_PET_Admin, 23, 'SPEI')

# In[59]

#saving files of the SPEI index into excel files

spei_1.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/spei_1.xlsx', index='False', engine='xlsxwriter') 
spei_3.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/spei_3.xlsx', index='False', engine='xlsxwriter') 
spei_6.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/spei_6.xlsx', index='False', engine='xlsxwriter') 
spei_12.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/spei_12.xlsx', index='False', engine='xlsxwriter') 
spei_24.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/spei_24.xlsx', index='False', engine='xlsxwriter') 
spei_2.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/spei_2.xlsx', index='False', engine='xlsxwriter') 
spei_4.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/spei_4.xlsx', index='False', engine='xlsxwriter') 
spei_5.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/spei_5.xlsx', index='False', engine='xlsxwriter') 
spei_7.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/spei_7.xlsx', index='False', engine='xlsxwriter') 
spei_8.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/spei_8.xlsx', index='False', engine='xlsxwriter') 
spei_9.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/spei_9.xlsx', index='False', engine='xlsxwriter') 
spei_10.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/spei_10.xlsx', index='False', engine='xlsxwriter') 
spei_11.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/spei_11.xlsx', index='False', engine='xlsxwriter') 
spei_13.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/spei_13.xlsx', index='False', engine='xlsxwriter') 
spei_14.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/spei_14.xlsx', index='False', engine='xlsxwriter') 
spei_15.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/spei_15.xlsx', index='False', engine='xlsxwriter') 
spei_16.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/spei_16.xlsx', index='False', engine='xlsxwriter') 
spei_17.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/spei_17.xlsx', index='False', engine='xlsxwriter') 
spei_18.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/spei_18.xlsx', index='False', engine='xlsxwriter') 
spei_19.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/spei_19.xlsx', index='False', engine='xlsxwriter') 
spei_20.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/spei_20.xlsx', index='False', engine='xlsxwriter') 
spei_21.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/spei_21.xlsx', index='False', engine='xlsxwriter') 
spei_22.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/spei_22.xlsx', index='False', engine='xlsxwriter') 
spei_23.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/spei_23.xlsx', index='False', engine='xlsxwriter') 

#%%

#SOIL MOISTURE->Standardized Soil Moisture Index

#Loading the regridded Soil moisture dataset

data_Sm =xr.open_dataset(r'C:\Users\roo290\surfdrive (backup)\Data\MSWEP_data\Past\regridded_file_Sm.nc')
                                      
print(data_Sm)

# In[42]

#Aggregate to the catchment resolution resolution
# CALCULATE MASK for SM data  

mask_Sm = sheds_mask_poly.mask(data_Sm.isel(time = 0 ), lon_name = 'lon', lat_name = 'lat')

#mask.to_netcdf(r'C:\Users\roo290\surfdrive (backup)\Data\MSWEP_data\Past\mask_Sm.nc') 

lat_Sm = mask_Sm.lat.values                                                                       # the lat values contained in the mask of the regions
lon_Sm = mask_Sm.lon.values

# the function aggregates the gridded data to a timeseries per catchment 

'''aggregating the gridded data to timeseries and converting it to a dataframe for each catchment
then appending it to a single dataframe with the HYBAS ID as header'''

region_ID = np.arange(0,3)

data_Sm_Admin = pd.DataFrame(index =pd.date_range(start="1980-01-01",end="2020-12-31",freq='M'))
for index in region_ID:
    print(index)
    ID_REGION=index
   
    sel_mask_Sm = mask_Sm.where(mask_Sm == ID_REGION).values
    
    id_lon_Sm = lon_Sm[np.where(~np.all(np.isnan(sel_mask_Sm), axis=0))]
    id_lat_Sm = lat_Sm[np.where(~np.all(np.isnan(sel_mask_Sm), axis=1))]
    
    out_sel1_Sm = data_Sm.sel(lat = slice(id_lat_Sm[0], id_lat_Sm[-1]), lon = slice(id_lon_Sm[0], id_lon_Sm[-1])).compute().where(mask_Sm == ID_REGION)
    
    plt.figure(figsize=(12,8))
    ax = plt.axes()
    out_sel1_Sm.SMroot.isel(time = 1).plot(ax = ax)
    data_Adminunits.plot(ax = ax, alpha = 0.8, facecolor = 'none')
    #plt.savefig(f'Modeloutputfolder/spei/{data_SHEDs.HYBAS_ID[ID_REGION]}.png')
        
    x_Sm = out_sel1_Sm.resample(time = '1M').mean()
    
    monthly_mean_Sm=x_Sm.SMroot.mean(dim=('lon','lat'))
    df_Sm=monthly_mean_Sm.to_dataframe()

    # df.to_excel(f'Modeloutputfolder/spei/{data_SHEDs.HYBAS_ID[ID_REGION]}.xlsx',sheet_name=f'{data_SHEDs.HYBAS_ID[ID_REGION]}',index='False', 
    #             engine='xlsxwriter')
    
    data_Sm_Admin[data_Adminunits.ADMIN_UNIT[ID_REGION]]=df_Sm['SMroot'].values
            


# In[32]

#saving the dataframe of the P ET timeseries into excel

data_Sm_Admin.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/data_Sm_Admin.xlsx', index='False', engine='xlsxwriter') 

#%%
def calculate_indicators_ssmi_ssfi (data_variable,accumulation, index):
    indicator=pd.DataFrame(index =pd.date_range(start="1981-01-01",end="2020-12-31",freq='M')) #Creating empty dataframes for the indices calculation
    for df in data_variable:
        print(df)
        data_variable[df]= data_variable[df].fillna(0)
        accumulateddata=moving_mean(data_variable[df].values, accumulation)
        indicator[df]=calculate_Index(accumulateddata[12:], index)
    return indicator


# In[47]

#Standardized Soil Moisture Index (SSMI)
    #calling the function to calculate SSMI based on the accumulation periods per catchment
    #the distribution applied is 
    
ssmi_1=calculate_indicators_ssmi_ssfi(data_Sm_Admin, 1, 'SSMI')
ssmi_3=calculate_indicators_ssmi_ssfi(data_Sm_Admin, 3, 'SSMI')
ssmi_6=calculate_indicators_ssmi_ssfi(data_Sm_Admin, 6, 'SSMI')
ssmi_12=calculate_indicators_ssmi_ssfi(data_Sm_Admin, 12,'SSMI')
ssmi_24=calculate_indicators_ssmi_ssfi(data_Sm_Admin, 24,'SSMI')
ssmi_2=calculate_indicators_ssmi_ssfi(data_Sm_Admin, 2, 'SSMI')
ssmi_4=calculate_indicators_ssmi_ssfi(data_Sm_Admin, 4, 'SSMI')
ssmi_5=calculate_indicators_ssmi_ssfi(data_Sm_Admin, 5, 'SSMI')
ssmi_7=calculate_indicators_ssmi_ssfi(data_Sm_Admin, 7,'SSMI')
ssmi_8=calculate_indicators_ssmi_ssfi(data_Sm_Admin, 8, 'SSMI')
ssmi_9=calculate_indicators_ssmi_ssfi(data_Sm_Admin, 9, 'SSMI')
ssmi_10=calculate_indicators_ssmi_ssfi(data_Sm_Admin, 10, 'SSMI')
ssmi_11=calculate_indicators_ssmi_ssfi(data_Sm_Admin, 11,'SSMI')
ssmi_13=calculate_indicators_ssmi_ssfi(data_Sm_Admin, 13, 'SSMI')
ssmi_14=calculate_indicators_ssmi_ssfi(data_Sm_Admin, 14, 'SSMI')
ssmi_15=calculate_indicators_ssmi_ssfi(data_Sm_Admin, 15, 'SSMI')
ssmi_17=calculate_indicators_ssmi_ssfi(data_Sm_Admin, 17,'SSMI')
ssmi_18=calculate_indicators_ssmi_ssfi(data_Sm_Admin, 18, 'SSMI')
ssmi_19=calculate_indicators_ssmi_ssfi(data_Sm_Admin, 19, 'SSMI')
ssmi_20=calculate_indicators_ssmi_ssfi(data_Sm_Admin, 20, 'SSMI')
ssmi_16=calculate_indicators_ssmi_ssfi(data_Sm_Admin, 16,'SSMI')
ssmi_21=calculate_indicators_ssmi_ssfi(data_Sm_Admin, 21, 'SSMI')
ssmi_22=calculate_indicators_ssmi_ssfi(data_Sm_Admin, 22, 'SSMI')
ssmi_23=calculate_indicators_ssmi_ssfi(data_Sm_Admin, 23, 'SSMI')

# In[48]

#saving files of the SSMI index into excel files

ssmi_1.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/ssmi_1.xlsx', index='False', engine='xlsxwriter') 
ssmi_3.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/ssmi_3.xlsx', index='False', engine='xlsxwriter') 
ssmi_6.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/ssmi_6.xlsx', index='False', engine='xlsxwriter') 
ssmi_12.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/ssmi_12.xlsx', index='False', engine='xlsxwriter') 
ssmi_24.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/ssmi_24.xlsx', index='False', engine='xlsxwriter') 
ssmi_2.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/ssmi_2.xlsx', index='False', engine='xlsxwriter') 
ssmi_4.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/ssmi_4.xlsx', index='False', engine='xlsxwriter') 
ssmi_5.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/ssmi_5.xlsx', index='False', engine='xlsxwriter') 
ssmi_7.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/ssmi_7.xlsx', index='False', engine='xlsxwriter') 
ssmi_8.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/ssmi_8.xlsx', index='False', engine='xlsxwriter') 
ssmi_9.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/ssmi_9.xlsx', index='False', engine='xlsxwriter') 
ssmi_10.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/ssmi_10.xlsx', index='False', engine='xlsxwriter') 
ssmi_11.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/ssmi_11.xlsx', index='False', engine='xlsxwriter') 
ssmi_13.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/ssmi_13.xlsx', index='False', engine='xlsxwriter') 
ssmi_14.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/ssmi_14.xlsx', index='False', engine='xlsxwriter') 
ssmi_15.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/ssmi_15.xlsx', index='False', engine='xlsxwriter') 
ssmi_16.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/ssmi_16.xlsx', index='False', engine='xlsxwriter') 
ssmi_17.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/ssmi_17.xlsx', index='False', engine='xlsxwriter') 
ssmi_18.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/ssmi_18.xlsx', index='False', engine='xlsxwriter') 
ssmi_19.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/ssmi_19.xlsx', index='False', engine='xlsxwriter') 
ssmi_20.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/ssmi_20.xlsx', index='False', engine='xlsxwriter') 
ssmi_21.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/ssmi_21.xlsx', index='False', engine='xlsxwriter') 
ssmi_22.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/ssmi_22.xlsx', index='False', engine='xlsxwriter') 
ssmi_23.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/ssmi_23.xlsx', index='False', engine='xlsxwriter') 

# In[50]

#DISCHARGE
#Loading discharge grib files 

data_dis= xr.open_mfdataset(r'U:\Rhoda\Surfdrive\Data\Glofas\*.grib',chunks={"time":10}, engine='cfgrib')

print(data_dis)
#%%


'''aggregating the gridded data to timeseries and converting it to a dataframe for each catchment
then appending it to a single dataframe with the HYBAS ID as header'''

data_dis_Admin = pd.DataFrame(index =pd.date_range(start="1980-01-01",end="2020-12-31",freq='M'))


region_ID = np.arange(0,3)

mask_dis = sheds_mask_poly.mask(data_dis.isel(time = 0 ), lon_name = 'longitude', lat_name = 'latitude')

lat_dis = mask_dis.latitude.values
lon_dis = mask_dis.longitude.values


for idx in region_ID:
    print(idx)
    ID_REGION=idx
   
    sel_mask_dis = mask_dis.where(mask_dis == ID_REGION).values
    
    id_lon_dis = lon_dis[np.where(~np.all(np.isnan(sel_mask_dis), axis=0))]
    id_lat_dis = lat_dis[np.where(~np.all(np.isnan(sel_mask_dis), axis=1))]
    
    out_sel1_dis = data_dis.sel(latitude = slice(id_lat_dis[0], id_lat_dis[-1]), longitude = slice(id_lon_dis[0], id_lon_dis[-1])).compute().where(mask_dis == ID_REGION)
    
    x_dis = out_sel1_dis.resample(time = '1M').mean()

    plt.figure(figsize=(12,8))
    ax = plt.axes()
    x_dis.dis24.isel(time = 1).plot(ax = ax)
    data_Adminunits.plot(ax = ax, alpha = 0.8, facecolor = 'none')
    #plt.savefig(f'U:/Rhoda/Surfdrive/Data/MSWEP_data/Past/ssfi/{data_SHEDs.HYBAS_ID[ID_REGION]}.png')
        
    monthly_mean_dis= (x_dis.dis24.max(dim=('longitude','latitude'))).to_dataframe()
    
    data_dis_Admin[data_Adminunits.HYBAS_ID[idx]] = monthly_mean_dis['dis24']        
            

            
# In[56]

#saving the dataframe of the P ET timeseries into excel

data_dis_Admin.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/data_dis_Admin.xlsx', index='False', engine='xlsxwriter') 

# In[58]

#Standardized Streamflow Index
    #calling the function to calculate SSFI based on the accumulation periods per catchment
    #the distribution applied is 
    
ssfi_1=calculate_indicators_ssmi_ssfi(data_dis_Admin, 1, 'SSFI')
ssfi_3=calculate_indicators_ssmi_ssfi(data_dis_Admin, 3, 'SSFI')
ssfi_6=calculate_indicators_ssmi_ssfi(data_dis_Admin, 6, 'SSFI')
ssfi_12=calculate_indicators_ssmi_ssfi(data_dis_Admin, 12,'SSFI')
ssfi_24=calculate_indicators_ssmi_ssfi(data_dis_Admin, 24,'SSFI')
ssfi_2=calculate_indicators_ssmi_ssfi(data_dis_Admin, 2, 'SSFI')
ssfi_4=calculate_indicators_ssmi_ssfi(data_dis_Admin, 4, 'SSFI')
ssfi_5=calculate_indicators_ssmi_ssfi(data_dis_Admin, 5, 'SSFI')
ssfi_7=calculate_indicators_ssmi_ssfi(data_dis_Admin, 7,'SSFI')
ssfi_8=calculate_indicators_ssmi_ssfi(data_dis_Admin, 8, 'SSFI')
ssfi_9=calculate_indicators_ssmi_ssfi(data_dis_Admin, 9, 'SSFI')
ssfi_10=calculate_indicators_ssmi_ssfi(data_dis_Admin, 10, 'SSFI')
ssfi_11=calculate_indicators_ssmi_ssfi(data_dis_Admin, 11,'SSFI')
ssfi_13=calculate_indicators_ssmi_ssfi(data_dis_Admin, 13, 'SSFI')
ssfi_14=calculate_indicators_ssmi_ssfi(data_dis_Admin, 14, 'SSFI')
ssfi_15=calculate_indicators_ssmi_ssfi(data_dis_Admin, 15, 'SSFI')
ssfi_17=calculate_indicators_ssmi_ssfi(data_dis_Admin, 17,'SSFI')
ssfi_18=calculate_indicators_ssmi_ssfi(data_dis_Admin, 18, 'SSFI')
ssfi_19=calculate_indicators_ssmi_ssfi(data_dis_Admin, 19, 'SSFI')
ssfi_20=calculate_indicators_ssmi_ssfi(data_dis_Admin, 20, 'SSFI')
ssfi_16=calculate_indicators_ssmi_ssfi(data_dis_Admin, 16,'SSFI')
ssfi_21=calculate_indicators_ssmi_ssfi(data_dis_Admin, 21, 'SSFI')
ssfi_22=calculate_indicators_ssmi_ssfi(data_dis_Admin, 22, 'SSFI')
ssfi_23=calculate_indicators_ssmi_ssfi(data_dis_Admin, 23, 'SSFI')

# In[59]

#saving files of the SSMI index into excel files

ssfi_1.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/ssfi_1.xlsx', index='False', engine='xlsxwriter') 
ssfi_3.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/ssfi_3.xlsx', index='False', engine='xlsxwriter') 
ssfi_6.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/ssfi_6.xlsx', index='False', engine='xlsxwriter') 
ssfi_12.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/ssfi_12.xlsx', index='False', engine='xlsxwriter') 
ssfi_24.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/ssfi_24.xlsx', index='False', engine='xlsxwriter') 
ssfi_2.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/ssfi_2.xlsx', index='False', engine='xlsxwriter') 
ssfi_4.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/ssfi_4.xlsx', index='False', engine='xlsxwriter') 
ssfi_5.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/ssfi_5.xlsx', index='False', engine='xlsxwriter') 
ssfi_7.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/ssfi_7.xlsx', index='False', engine='xlsxwriter') 
ssfi_8.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/ssfi_8.xlsx', index='False', engine='xlsxwriter') 
ssfi_9.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/ssfi_9.xlsx', index='False', engine='xlsxwriter') 
ssfi_10.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/ssfi_10.xlsx', index='False', engine='xlsxwriter') 
ssfi_11.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/ssfi_11.xlsx', index='False', engine='xlsxwriter') 
ssfi_13.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/ssfi_13.xlsx', index='False', engine='xlsxwriter') 
ssfi_14.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/ssfi_14.xlsx', index='False', engine='xlsxwriter') 
ssfi_15.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/ssfi_15.xlsx', index='False', engine='xlsxwriter') 
ssfi_16.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/ssfi_16.xlsx', index='False', engine='xlsxwriter') 
ssfi_17.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/ssfi_17.xlsx', index='False', engine='xlsxwriter') 
ssfi_18.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/ssfi_18.xlsx', index='False', engine='xlsxwriter') 
ssfi_19.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/ssfi_19.xlsx', index='False', engine='xlsxwriter') 
ssfi_20.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/ssfi_20.xlsx', index='False', engine='xlsxwriter') 
ssfi_21.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/ssfi_21.xlsx', index='False', engine='xlsxwriter') 
ssfi_22.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/ssfi_22.xlsx', index='False', engine='xlsxwriter') 
ssfi_23.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/ssfi_23.xlsx', index='False', engine='xlsxwriter') 


#%%

#the function loads excel files into dataframe
    #path is the location of the file to be loaded

def loading_excel (path):
    file= pd.read_excel (path)
    file.set_index('Unnamed: 0',inplace=True) # make the first column into the index
    file.index.rename('Index',inplace=True) #rename the new index
    return file

#%%
#loading the dataframes for the monthly hydrometeorological variables
data_prec_Admin = loading_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/data_prec_Admin.xlsx')
data_Sm_Admin = loading_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/data_Sm_Admin.xlsx')
data_PET_Admin = loading_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/data_PET_Admin.xlsx')
data_dis_Admin = loading_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/data_dis_Admin.xlsx')

#%%
# fig = plt.figure(figsize=(30,10) ) # define size of plot
# l1=plt.plot(super_dictionary['ssmi','12']['1060008100'].index,super_dictionary['ssmi','1']['1060008100'],"-", color='blue', label = 'new')
# l2=plt.plot(super_dictionary_old['ssmi','12'][1060008100].index,super_dictionary_old['ssmi','1'][1060008100],"-", color='black', label = '0ld')

# #plt.title('SPI_1',size=20) # give plots a title
# # plt.xlabel('yield (t/ha)')
# plt.ylabel('SSMI-1 values')
# plt.legend() # create legend
# plt.show()

#%%

#Loading all the indices into one dictionary

super_dictionary = {}

def create_super_dictionary(indicator, accumulation):
    opened_file = pd.read_excel(f"C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/{indicator}_{accumulation}.xlsx")
    frame = opened_file.set_index('Unnamed: 0')
    frame.index.rename('Index',inplace=True) 
    super_dictionary.update({(indicator, accumulation):frame})
    
indicator = ['spi','ssmi', 'spei','ssfi']
accumulation = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17',
                '18','19', '20', '21', '22', '23', '24']

#Loading the datasets for the indices
for ind, acc in itertools.product(indicator, accumulation):
    create_super_dictionary(indicator=ind, accumulation=acc)
    
#%%
'''CROP YIELD'''
#obtained from GDHY dataset: 1981-2016
#Loading crop yield data and aggregating it to timeseries for each of the catchments
# list_of_files = glob.glob(r'C:\Users\roo290\OneDrive - Vrije Universiteit Amsterdam\Data\crop yield\*.nc4')

yield_regridded= xr.open_dataset (r'C:\Users\roo290\OneDrive - Vrije Universiteit Amsterdam\Data\crop yield\yield_regridded.nc')

#%%
# # Plot yearly yield
# import cartopy.crs as ccrs

# yield_global = yield_regridded['var'][:]
# ax = plt.axes(projection=ccrs.PlateCarree())
# ax.coastlines()
# yield_global.plot()
# plt.show()
#%%
'''aggregating the gridded data to timeseries and converting it to a dataframe for each catchment
then appending it to a single dataframe with the HYBAS ID as header'''

mask = sheds_mask_poly.mask(yield_regridded, lon_name = 'lon', lat_name = 'lat')
region_ID = np.arange(0,157)

#mask.to_netcdf(r'C:\Users\roo290\surfdrive (backup)\Data\MSWEP_data\Past\mask_pet.nc') 

lat = mask.lat.values                                                                       # the lat values contained in the mask of the regions
lon = mask.lon.values

# In[43]

#plotting the mask for one of the catchments

mask_o=mask.where(mask==145) 
plt.figure(figsize=(16,8))
ax = plt.axes()
mask_o.plot(ax = ax)
data_Adminunits.plot(ax = ax, alpha = 0.8, facecolor = 'none', lw = 1)

# In[44]

#extract timeseries for the a single catchment as a test 

ID_REGION = 64
print(data_Adminunits.ADMIN_UNIT[ID_REGION])

#%%

sel_mask = mask.where(mask == ID_REGION).values

#%%
id_lon = lon[np.where(~np.all(np.isnan(sel_mask), axis=0))]
id_lat = lat[np.where(~np.all(np.isnan(sel_mask), axis=1))]

out_sel1 = yield_regridded.sel(lat = slice(id_lat[0], id_lat[-1]), lon = slice(id_lon[0], id_lon[-1])).compute().where(mask == ID_REGION)

#%%
#plotting the extracted catchment P ET values for the first year

plt.figure(figsize=(12,8))
ax = plt.axes()
out_sel1['var'].isel(time = 1).plot(ax = ax)
data_Adminunits.plot(ax = ax, alpha = 0.8, facecolor = 'none')
#plt.savefig(r'U:\Rhoda\Plots_csv' + str(data_SHEDs.HYBAS_ID[ID_REGION])+'.pdf')
 
#%%
#x = np.where((out_sel1.var == -9999),np.nan, out_sel1.var)
annual_yied=out_sel1['var'].mean(dim=('lon','lat'))
#%%

data_yield_Admin = pd.DataFrame(index =pd.date_range(start="1981",end="2017",freq='Y'))
for index in region_ID:
    print(index)
    ID_REGION=index
   
    sel_mask = mask.where(mask == ID_REGION).values
    
    id_lon = lon[np.where(~np.all(np.isnan(sel_mask), axis=0))]
    id_lat = lat[np.where(~np.all(np.isnan(sel_mask), axis=1))]
    
    out_sel1 = yield_regridded.sel(lat = slice(id_lat[0], id_lat[-1]), lon = slice(id_lon[0], id_lon[-1])).compute().where(mask == ID_REGION)
    
    plt.figure(figsize=(12,8))
    ax = plt.axes()
    out_sel1['var'].isel(time = 1).plot(ax = ax)
    data_Adminunits.plot(ax = ax, alpha = 0.8, facecolor = 'none')
    #plt.savefig(f'Modeloutputfolder/spei/{data_SHEDs.HYBAS_ID[ID_REGION]}.png')
        
    #x = np.where((out_sel1.var == -9999),np.nan, out_sel1.var)
    
    annual_yied=out_sel1['var'].mean(dim=('lon','lat'))
    df=annual_yied.to_dataframe()

    # df.to_excel(f'Modeloutputfolder/spei/{data_SHEDs.HYBAS_ID[ID_REGION]}.xlsx',sheet_name=f'{data_SHEDs.HYBAS_ID[ID_REGION]}',index='False', 
    #             engine='xlsxwriter')
    
    data_yield_Admin[data_Adminunits.ADMIN_UNIT[ID_REGION]]=df['var'].values

#%%

#saving the yield timeseries

data_yield_Admin.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/data_yield_Admin.xlsx', index='False', engine='xlsxwriter') 
data_yield_Admin = loading_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/data_yield_Admin.xlsx')

#%%

#creating a data frame with all the nan values to see which admin units have no data
#this is to compare with the landuse cover map to confirm crop areas
rows = data_yield_Admin.columns
df_yield_nan = pd.DataFrame(index = rows, columns=['count'])
for i,col in enumerate (df_yield_nan.index):
    print(col)
    count_nan = data_yield_Admin[col].isna().sum()
    df_yield_nan['count'].iloc[i]=count_nan
    
#%%

plt.figure(figsize=(12,8))
ax = plt.axes()
df_yield_nan.plot(ax = ax)
data_Adminunits.plot(ax = ax, alpha = 0.8, facecolor = 'none')

#writing the geodataframe
df_yield_nan = gpd.GeoDataFrame(df_yield_nan, crs="EPSG:4326", geometry=data_Adminunits['geometry'])

#saving geodataframe to shapefile
df_yield_nan.to_file(driver = 'ESRI Shapefile', filename= 'C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/SHEDS/df_yield_nan.shp')

#%%

'''CROP YIELD '''


#Detrending the yield datasets and comparing with a linear model, this is done per catchment

#we will use the loess regression model for detrending 

# determine whether a linear regression cannot produce a good fit for the datasets by making linear scatter plots for each catchment

# Create a line plot for each of the admin units crop yield

data_yield_Admin.reset_index(inplace=True)
data_yield_Admin.rename(columns = {'Index': 'Year'}, inplace= True)
data_yield_Admin.set_index('Year', inplace=True)
#%%
data_yield_Admin['Year']=data_yield_Admin['Year'].dt.year
    
#%%

for col in data_yield_Admin:
    
    fig = px.scatter(data_yield_Admin, x= 'Year', y= col, opacity=0.8, color_discrete_sequence=['black'], trendline = 'ols')
    
    # Change chart background color
    fig.update_layout(dict(plot_bgcolor = 'white'))
    
    # Update axes lines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey', showline=True, linewidth=1, linecolor='black')
    
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey', showline=True, linewidth=1, linecolor='black')
    
    # Set figure title
    fig.update_layout(title=dict(text="Crop yield over the years", font=dict(color='black')))
    
    # Update marker size
    fig.update_traces(marker=dict(size=3))
    fig.write_html(f"C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/crop yield/yield_plots/{col}.html")
    
    fig.show()
#%%



#selecting the data to be used for each per admin unit; this includes the indices data

#Aggregating spi values to yearl values using -0.5 & -1.5 thresholds

df_selected_Admin = {}


accumulation_sublist = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']



for indices in indicator:
    
    counties_list = list(super_dictionary[(indices, accumulation[0])].keys())
    
for key in counties_list:
    df_final = pd.DataFrame(index =np.arange(1981,2021,1))
    for indices, index in super_dictionary:
        #print (indices, index,key)

    # np_final = np.zeros_like(df_final)
    # for idx, index in enumerate(accumulation_sublist):
    #     print(indices, key, index)
        if index == '1':
            mnth_data1= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==1)]).values
            mnth_data2= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==2)]).values
            mnth_data3= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==3)]).values
            mnth_data4= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==4)]).values
            mnth_data5= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==5)]).values
            mnth_data6= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==6)]).values
            mnth_data7= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==7)]).values
            mnth_data8= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==8)]).values
            mnth_data9= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==9)]).values
            mnth_data10= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==10)]).values
            mnth_data11= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==11)]).values
            mnth_data12= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==12)]).values
                            
            df_final['%s%s_%s' % (indices,index,1)]=mnth_data1; df_final['%s%s_%s' % (indices,index,2)]=mnth_data2;df_final['%s%s_%s' % (indices,index,3)]=mnth_data3;df_final['%s%s_%s' % (indices,index,4)]=mnth_data4;
            df_final['%s%s_%s' % (indices,index,5)]=mnth_data5;df_final['%s%s_%s' % (indices,index,6)]=mnth_data6;df_final['%s%s_%s' % (indices,index,7)]=mnth_data7;df_final['%s%s_%s' % (indices,index,8)]=mnth_data8;
            df_final['%s%s_%s' % (indices,index,9)]=mnth_data9;df_final['%s%s_%s' % (indices,index,10)]=mnth_data10;df_final['%s%s_%s' % (indices,index,11)]=mnth_data11;df_final['%s%s_%s' % (indices,index,12)]=mnth_data12

        elif index=='2':

            mnth_data2= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==2)]).values
            mnth_data3= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==3)]).values
            mnth_data4= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==4)]).values
            mnth_data5= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==5)]).values
            mnth_data6= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==6)]).values
            mnth_data7= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==7)]).values
            mnth_data8= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==8)]).values
            mnth_data9= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==9)]).values
            mnth_data10= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==10)]).values
            mnth_data11= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==11)]).values
            mnth_data12= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==12)]).values
                            

            df_final['%s%s_%s' % (indices,index,2)]=mnth_data2;df_final['%s%s_%s' % (indices,index,3)]=mnth_data3;df_final['%s%s_%s' % (indices,index,4)]=mnth_data4;
            df_final['%s%s_%s' % (indices,index,5)]=mnth_data5;df_final['%s%s_%s' % (indices,index,6)]=mnth_data6;df_final['%s%s_%s' % (indices,index,7)]=mnth_data7;df_final['%s%s_%s' % (indices,index,8)]=mnth_data8;
            df_final['%s%s_%s' % (indices,index,9)]=mnth_data9;df_final['%s%s_%s' % (indices,index,10)]=mnth_data10;df_final['%s%s_%s' % (indices,index,11)]=mnth_data11;df_final['%s%s_%s' % (indices,index,12)]=mnth_data12
        
        elif index=='3':

            mnth_data3= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==3)]).values
            mnth_data4= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==4)]).values
            mnth_data5= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==5)]).values
            mnth_data6= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==6)]).values
            mnth_data7= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==7)]).values
            mnth_data8= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==8)]).values
            mnth_data9= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==9)]).values
            mnth_data10= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==10)]).values
            mnth_data11= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==11)]).values
            mnth_data12= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==12)]).values
                            
          
            df_final['%s%s_%s' % (indices,index,3)]=mnth_data3;df_final['%s%s_%s' % (indices,index,4)]=mnth_data4;
            df_final['%s%s_%s' % (indices,index,5)]=mnth_data5;df_final['%s%s_%s' % (indices,index,6)]=mnth_data6;df_final['%s%s_%s' % (indices,index,7)]=mnth_data7;df_final['%s%s_%s' % (indices,index,8)]=mnth_data8;
            df_final['%s%s_%s' % (indices,index,9)]=mnth_data9;df_final['%s%s_%s' % (indices,index,10)]=mnth_data10;df_final['%s%s_%s' % (indices,index,11)]=mnth_data11;df_final['%s%s_%s' % (indices,index,12)]=mnth_data12
        
        elif index=='4':

            mnth_data4= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==4)]).values
            mnth_data5= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==5)]).values
            mnth_data6= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==6)]).values
            mnth_data7= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==7)]).values
            mnth_data8= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==8)]).values
            mnth_data9= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==9)]).values
            mnth_data10= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==10)]).values
            mnth_data11= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==11)]).values
            mnth_data12= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==12)]).values
                            
            
            df_final['%s%s_%s' % (indices,index,4)]=mnth_data4;
            df_final['%s%s_%s' % (indices,index,5)]=mnth_data5;df_final['%s%s_%s' % (indices,index,6)]=mnth_data6;df_final['%s%s_%s' % (indices,index,7)]=mnth_data7;df_final['%s%s_%s' % (indices,index,8)]=mnth_data8;
            df_final['%s%s_%s' % (indices,index,9)]=mnth_data9;df_final['%s%s_%s' % (indices,index,10)]=mnth_data10;df_final['%s%s_%s' % (indices,index,11)]=mnth_data11;df_final['%s%s_%s' % (indices,index,12)]=mnth_data12
        
        elif index=='5':

            mnth_data5= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==5)]).values
            mnth_data6= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==6)]).values
            mnth_data7= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==7)]).values
            mnth_data8= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==8)]).values
            mnth_data9= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==9)]).values
            mnth_data10= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==10)]).values
            mnth_data11= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==11)]).values
            mnth_data12= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==12)]).values
                            
            
            df_final['%s%s_%s' % (indices,index,5)]=mnth_data5;df_final['%s%s_%s' % (indices,index,6)]=mnth_data6;df_final['%s%s_%s' % (indices,index,7)]=mnth_data7;df_final['%s%s_%s' % (indices,index,8)]=mnth_data8;
            df_final['%s%s_%s' % (indices,index,9)]=mnth_data9;df_final['%s%s_%s' % (indices,index,10)]=mnth_data10;df_final['%s%s_%s' % (indices,index,11)]=mnth_data11;df_final['%s%s_%s' % (indices,index,12)]=mnth_data12
        
        elif index=='6':

            mnth_data6= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==6)]).values
            mnth_data7= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==7)]).values
            mnth_data8= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==8)]).values
            mnth_data9= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==9)]).values
            mnth_data10= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==10)]).values
            mnth_data11= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==11)]).values
            mnth_data12= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==12)]).values
                            

            df_final['%s%s_%s' % (indices,index,6)]=mnth_data6;df_final['%s%s_%s' % (indices,index,7)]=mnth_data7;df_final['%s%s_%s' % (indices,index,8)]=mnth_data8;
            df_final['%s%s_%s' % (indices,index,9)]=mnth_data9;df_final['%s%s_%s' % (indices,index,10)]=mnth_data10;df_final['%s%s_%s' % (indices,index,11)]=mnth_data11;df_final['%s%s_%s' % (indices,index,12)]=mnth_data12
        
        elif index=='7':

            mnth_data7= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==7)]).values
            mnth_data8= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==8)]).values
            mnth_data9= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==9)]).values
            mnth_data10= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==10)]).values
            mnth_data11= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==11)]).values
            mnth_data12= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==12)]).values
                            

            df_final['%s%s_%s' % (indices,index,7)]=mnth_data7;df_final['%s%s_%s' % (indices,index,8)]=mnth_data8;
            df_final['%s%s_%s' % (indices,index,9)]=mnth_data9;df_final['%s%s_%s' % (indices,index,10)]=mnth_data10;df_final['%s%s_%s' % (indices,index,11)]=mnth_data11;df_final['%s%s_%s' % (indices,index,12)]=mnth_data12
        
        elif index=='8':

            mnth_data8= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==8)]).values
            mnth_data9= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==9)]).values
            mnth_data10= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==10)]).values
            mnth_data11= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==11)]).values
            mnth_data12= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==12)]).values
                            
            df_final['%s%s_%s' % (indices,index,8)]=mnth_data8;
            df_final['%s%s_%s' % (indices,index,9)]=mnth_data9;df_final['%s%s_%s' % (indices,index,10)]=mnth_data10;df_final['%s%s_%s' % (indices,index,11)]=mnth_data11;df_final['%s%s_%s' % (indices,index,12)]=mnth_data12
        
        elif index=='9':
           
            mnth_data9= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==9)]).values
            mnth_data10= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==10)]).values
            mnth_data11= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==11)]).values
            mnth_data12= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==12)]).values
                            
            
            df_final['%s%s_%s' % (indices,index,9)]=mnth_data9;df_final['%s%s_%s' % (indices,index,10)]=mnth_data10;df_final['%s%s_%s' % (indices,index,11)]=mnth_data11;df_final['%s%s_%s' % (indices,index,12)]=mnth_data12
        
        elif index=='10':
            
            mnth_data10= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==10)]).values
            mnth_data11= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==11)]).values
            mnth_data12= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==12)]).values
                            
            df_final['%s%s_%s' % (indices,index,10)]=mnth_data10;df_final['%s%s_%s' % (indices,index,11)]=mnth_data11;df_final['%s%s_%s' % (indices,index,12)]=mnth_data12
        
        elif index=='11':
        
            mnth_data11= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==11)]).values
            mnth_data12= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==12)]).values
                            
            df_final['%s%s_%s' % (indices,index,11)]=mnth_data11;df_final['%s%s_%s' % (indices,index,12)]=mnth_data12
        
        elif index=='12':
            
            mnth_data12= (super_dictionary[(indices, index)][key].loc[(super_dictionary[(indices, index)][key].index.month==12)]).values
                            
            df_final['%s%s_%s' % (indices,index,12)]=mnth_data12
            df_final['yield_values'] = data_yield_Admin[key]
            
    df_selected_Admin[key]= df_final   
            # #df_final.drop([1980,2017,2018,2019,2020], axis=0, inplace=True)
            # df_final['yield_values']= country_yield_dict[key]['yield_values'].values
            # df_final.dropna(inplace =True)
            # # print (df_final.head())
            # country_dict[key] = df_final
                       
  #%%
#dropping the nans in the yield data
for key in df_selected_Admin:
    df_selected_Admin[key].dropna(inplace=True)
df_selected_Admin =  {k:v for (k,v) in df_selected_Admin.items() if not v.empty}

#%%
#saving selected data for yield analysis

#Saving files
a_file = open("C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/df_selected_yield.pkl", "wb")
pickle.dump(df_selected_Admin, a_file)
a_file.close()

a_file = open("C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/df_selected_yield.pkl", "rb")
df_selected_yield = pickle.load(a_file)

#%%
'''detrending the crop yield data per admin unit'''

#fitting lowess model to the yield values per admin unit
def detrending_lowess_model (country_dict,new_dict):
    
    for key in country_dict:
        df = country_dict[key][['yield_values']].copy()
        df=df.reset_index().rename(columns={'index':'Year'})
        X= df['Year']
        y=df['yield_values']
        results = lowess(y,X, frac = 0.25)#fitting lowess model
        #f_linear = interp1d(results[:,0], results[:,1], bounds_error=False, kind='linear', fill_value='extrapolate')
        Y_pred = pd.DataFrame (results)
        df['residuals'] = (df["yield_values"].values-Y_pred[1])/(Y_pred[1])
        mild= ((df['yield_values'].values).mean())*(-0.05)
        extreme =((df['yield_values'].values).mean())*(-0.1)
        severe =((df['yield_values'].values).mean())*(-0.15)
        df['mild'] = np.where((df['residuals'] < mild),1,0)
        df['extreme'] = np.where((df['residuals'] < extreme),1,0)
        df['severe'] = np.where((df['residuals'] < severe),1,0)
        new_dict[key]=df
        country_dict[key]['mild']=df['mild'].values
        country_dict[key]['extreme']=df['extreme'].values
        country_dict[key]['severe']=df['severe'].values
        
#%%
df_yield_lowess={}
detrending_lowess_model(df_selected_yield,df_yield_lowess)

#%%
#saving the selected data
a_file = open("C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/df_yield_lowess.pkl", "wb")
pickle.dump(df_yield_lowess, a_file)
a_file.close()

a_file = open("C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/PPR2_results/df_yield_lowess.pkl", "rb")
df_yield_lowess = pickle.load(a_file)

#%%

