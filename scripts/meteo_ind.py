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
from function_def import plot_xarray_dataset
path='C:/Users/tbr910/Documents/Forecast_action_analysis'
CHIRPSV2_HAD=os.path.join(path, 'CHIRPS025/HAD') 

#%%% pre-process rainfall 
os.chdir(CHIRPSV2_HAD)
P_HAD=xr.open_dataset('chirps-v2_ALL_YEARS_sub_HAD_ORIGINAL.nc')#chirps-v2_ALL_YEARS_sub_HAD_NEW
P_HAD=P_HAD.rename(band_data='tp') #precip 
P_HAD_DN=P_HAD.where(P_HAD['tp']!=-9.999e+03) ## save -9999 values as NAN
P_HAD_DN=P_HAD_DN.where(P_HAD_DN['tp']<1e+10) ## delete all high values (becomes nan). ASK VICKY WHERE THESE HIGH VALUES COME FROM! 
P_HAD_DN=P_HAD_DN.where(P_HAD_DN['tp']>-2000) #Delete very small rainfall values


#%%% resample to monthly
P_monthly=P_HAD_DN.resample(time='MS').sum() ## monthly 


#%% SPI 
pr=P_HAD_DN.tp



#%% with Xclim 
pr.attrs["units"] = "mm/day" # spi needs a unit as attribute
pr_cal=pr.sel(time=slice(datetime(1981, 1, 1), datetime(2021, 12, 31))) #reference/calibration period. Should be at least 30 years: https://edo.jrc.ec.europa.eu/documents/factsheets/factsheet_spi_ado.pdf

# Define accumulation periods
periods = [1, 3, 6, 12, 24, 48]

# Calculate SPI values
for period in periods:
    spi = standardized_precipitation_index(pr, pr_cal, freq='MS',window=1, method='ML',dist='gamma')

#2021 nov
plot_xarray_array(spi.isel(time=490),'latitude', 'longitude','test_plot','test_plot', 1,'monthly')


zero_precip=P_monthly.isel(time=490)==0
plot_xarray_dataset(P_monthly.isel(time=490),'tp','latitude', 'longitude','test_plot', 1,'monthly')

.isel(time=200)



# Save dataset
#ds.to_netcdf('spi.nc')