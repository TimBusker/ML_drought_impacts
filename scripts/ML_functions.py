# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 11:33:21 2022

@author: tbr910
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
from netCDF4 import Dataset as netcdf_dataset
import numpy as np
import cartopy
from cartopy import config
import cartopy.crs as ccrs
import cartopy.feature as cf
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
import cartopy.feature as cfeature
import cartopy.mpl.geoaxes
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.cm as cm
import geopandas as gpd
import regionmask
import xarray as xr
import urllib.request
import requests
import re
import glob
from bs4 import BeautifulSoup
import datetime as dtmod
import re
from rasterio.enums import Resampling
import scipy.stats as stats
path = 'C:/Users/tbr910/Documents/Forecast_action_analysis'

os.chdir(path)


#%% ML Functions 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, explained_variance_score
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# transform a time series dataset into a supervised learning dataset
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True): # creates moving window of arrays with length 7
	#n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols = list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
	# put it all together
	agg = pd.concat(cols, axis=1)
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg.values

# transform a time series dataset into a supervised learning dataset
def series_to_supervised_complex(data, n_in=1, n_out=1, dropnan=True): # creates moving window of arrays with length 7
     """
     Frame a time series as a supervised learning dataset.
     Arguments:
     data: Sequence of observations as a list or NumPy array.
     n_in: Number of lag observations as input (X).
     n_out: Number of observations as output (y).
     dropnan: Boolean whether or not to drop rows with NaN values.
     Returns:
     Pandas DataFrame of series framed for supervised learning.
     """
     n_vars = 1 if type(data) is list else data.shape[1]
     df = pd.DataFrame(data)
     cols, names = list(), list()
     # input sequence (t-n, ... t-1)
     for i in range(n_in, 0, -1):
         cols.append(df.shift(i))
         names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
         # forecast sequence (t, t+1, ... t+n)
     for i in range(0, n_out):
         cols.append(df.shift(-i))
         if i == 0:
             names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
         else:
             names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
     # put it all together
     agg = pd.concat(cols, axis=1)
     agg.columns = names
     # drop rows with NaN values
     if dropnan:
         agg.dropna(inplace=True)
     return agg




# split a univariate dataset into train/test sets
def train_test_split_2(data, n_test):
	return data[:-n_test, :], data[-n_test:, :]


# fit an random forest model and make a one step prediction
def random_forest_forecast(train, testX):
	# transform list into array
	train = np.asarray(train)
	# split into input and output columns
	trainX, trainy = train[:, :-1], train[:, -1]
	# fit model
	model = RandomForestRegressor(n_estimators=1000)
	model.fit(trainX, trainy)
	# make a one-step prediction
	yhat = model.predict([testX])
	return yhat[0]


def walk_forward_validation(data, n_test):
	predictions = list()
	# split dataset
	train, test = train_test_split(data, n_test)
	# seed history with training dataset
	history = [x for x in train]
	# step over each time-step in the test set
	for i in range(len(test)):
		# split test row into input and output columns
		testX, testy = test[i, :-1], test[i, -1]
		# fit model on history and make a prediction
		yhat =     (history, testX)
		# store forecast in list of predictions
		predictions.append(yhat)
		# add actual observation to history for the next loop
		history.append(test[i])
		# summarize progress
		print('>expected=%s, predicted=%s'%(testy, yhat))
	# estimate prediction error
	error = mean_absolute_error(test[:, -1], predictions)
	return error, test[:, 1], predictions
        
#%%% Mask areas 
def P_mask(p_input,month, resample, p_thres):
    
    ## vars needed to sel months 
    months= (range(1,13))
    months_z= []
    for i in months: 
        j=f"{i:02d}"
        months_z.append(j)  
        
    def month_selector(month_select):
        return (month_select == month_of_interest)  ## can also be a range
                
    mask= p_input.resample(time=resample).sum()
    month_of_interest= int(months_z[month]) ## oct ## convert to int to select month, doesnt work with string
    mask = mask.sel(time=month_selector(mask['time.month']))
    mask=mask.mean(dim='time')
    mask= mask.where(mask.tp>p_thres, 0)
    mask=mask.where(mask.tp ==0, 1)
    
    return mask
   

def rasterize_shp(input_shp, input_raster,resolution, upscaling_fac): # files should include file names and dir, res can be either 'as_input' or 'upscaled'. input raster can be of any resolution. 
    
    rainfall= input_raster#chirps-v2_ALL_YEARS_sub_HAD_NEW
    if resolution=='as_input': 
        #%% load CHIRPS and extract lon lats 
        
        lon_raster=rainfall.longitude.values
        lat_raster=rainfall.latitude.values
        
    if resolution=='upscaled': 
        #%% create mask 
        upscale_factor = upscaling_fac
        new_width = rainfall.rio.width * upscale_factor
        new_height = rainfall.rio.height * upscale_factor
        rainfall_mask=rainfall.rio.write_crs(4326, inplace=True)
        xds_upsampled = rainfall.rio.reproject(
            rainfall_mask.rio.crs,
            shape=(new_height, new_width),
            resampling=Resampling.bilinear,
        )

        lon_raster=xds_upsampled.x.values
        lat_raster=xds_upsampled.y.values
        
    
           
    #%% make county raster with ID's 
    sf = input_shp
    
    sf_raster= regionmask.mask_geopandas(sf,lon_raster,lat_raster)        
    
    return (sf_raster)
    


#%% Wet days calculation 
def wet_days(rainfall_input, threshold, period):
    #%%%%% flag dry and wet days 
    dry_days=rainfall_input.where((rainfall_input['tp'] >threshold) | rainfall_input.isnull(), 1) # isnull causes nan values to persist 
    dry_days=dry_days.where((dry_days['tp'] ==1) | dry_days.isnull(), 0)
    
    wet_days=rainfall_input.where((rainfall_input['tp'] >threshold) | rainfall_input.isnull(), 0)
    wet_days=wet_days.where((wet_days['tp'] ==0) | wet_days.isnull(), 1)
    #%%%%% monthly sums 
    wet_days_number=wet_days.resample(time=period).sum()
    #wet_days_number = wet_days_number.where((((wet_days_number['time'].dt.month >=3) & (wet_days_number['time'].dt.month <=5)) | ((wet_days_number['time'].dt.month >=10) & (wet_days_number['time'].dt.month <=12))), np.nan)#this makes dry season months nan's
    return(wet_days_number)
    

#%% dry days-spells calculation 
def max_dry_spells(rainfall_input, threshold, period):
    #%%% dry days 
    dry_days=rainfall_input.where((rainfall_input['tp'] >threshold) | rainfall_input.isnull(), 1) # isnull causes nan values to persist 
    dry_days=dry_days.where((dry_days['tp'] ==1) | dry_days.isnull(), 0)
    
    #%%% dry spells 
    ###### put beginning of season to 0! Leave all the other values representing the dry spell length --> creates inaccuracy of dry spell length with 0 day for dry spells that start at first day of the season. This is important as otherwise the fake dry days of the non-rainy season months (np.nan values) will create a very long dry spell in the rainy season. 
    dry_spell=dry_days.where((((dry_days['time'].dt.month !=3) | (dry_days['time'].dt.day !=1)) & ((dry_days['time'].dt.month !=10) | (dry_days['time'].dt.day !=1))), 0)
    ###### make a cumsum using only the dry days ##########
    # This function restarts cumsum after every 0, so at every wet day or beginning of the season. 
    cumulative = dry_spell['tp'].cumsum(dim='time')-dry_spell['tp'].cumsum(dim='time').where(dry_spell['tp'].values == 0).ffill(dim='time').fillna(0)
    dry_spell_length= cumulative.where(cumulative>=5, 0) ## only keep length of dry spells when >=5 days. Rest of the values to zero

    
    ####################################################################################### number of dry spells per season  ################################################################################
    # dry_spell_start=dry_spell_length.where(dry_spell_length==5, 0)
    # dry_spell_start=dry_spell_start.where(dry_spell_start!=5, 1) ## binary dry spell start  
    
    # dry_spell_number=dry_spell_start.resample(time='MS').sum() ## sum all dry spell starts 
    # dry_spell_number = dry_spell_number.where((((dry_spell_number['time'].dt.month >=3) & (dry_spell_number['time'].dt.month <=5)) | ((dry_spell_number['time'].dt.month >=10) & (dry_spell_number['time'].dt.month <=12))), np.nan)#this makes dry season months nan's
    
    # dry_spell_number=dry_spell_number.where(land_mask==1, np.nan) ## land mask
    # dry_spell_number=dry_spell_number.to_dataset()

    ####################################################################################### maximum (seasonal!!) dry spell length ################################################################################
    dry_spell_max= dry_spell_length.resample(time=period).max() ## max dry spell length per month, per season
    dry_spell_max=dry_spell_max.to_dataset() ## for processing later on 
    
    return (dry_spell_max)





#%% SPI 

# Indicate the time for which data is available
start = 1981
end   = 2021
years = end - start 

# Choose reference time period (years from first year of data)
      # this can be either the historic records or the full data series 
      # (depending on how droughts are defined and the goal of the study)
        
refstart = 1982   # ! First year of data series cannot be used ! 
refend   = 2021



# function to accumulate hydrological data
   # with a the input data and b the accumulation time
   # -> the accumulated value coincidences with the position of the last value 
   #     used in the accumulation process.

def moving_sum(a, b) :
    
    cummuldata = np.cumsum(a, dtype=float)                 
    cummuldata[b:] = cummuldata[b:] - cummuldata[:-b]         
    cummuldata[:b - 1] = np.nan                                           
    return cummuldata


def calculate_Zval(refseries, fullseries, Index):


    # find fitting distribution for reference sereis                          
    if Index == 'SPEI':                     # Suggestions by Stagge et al. (2015)
        dist_names = ['genextreme','genlogistic','pearson3','fisk']                  
    elif Index == 'SSMI':                   # Suggestions in Ryu et al. (2005)
        dist_names = ['genextreme','norm','beta','pearson3']    
    elif Index == 'SEI':                   # Suggestions in Ryu et al. (2005)
        dist_names = ['genextreme','norm','beta','pearson3']  
    elif Index == 'SPI' :                   # Suggestions by Stagge et al. (2015)
        dist_names = ['logistic','gamma', 'weibull_min', 'gumbel_r']
    elif Index == 'SSFI' :                   # Suggestions by Vincent_Serrano et al. (2012) Modarres 2007 https://agupubs-onlinelibrary-wiley-com.vu-nl.idm.oclc.org/doi/full/10.1002/2016WR019276
        dist_names = ['genlogistic','lognorm','gamma', 'weibull_min', 'kappa3'] #'fisk','gumbel','genpareto','weibull_min', 

    # find fit for each optional distribution on reference data
    dist_results = []
    params = {}
    for dist_name in dist_names:                                                # Find distribution parameters       
        dist = getattr(stats, dist_name)
        param = dist.fit(np.sort(refseries)) # sort was not in Rhoda's script 
        params[dist_name] = param

        # Assess fitting of different distributions on reference data

        D, p = stats.kstest(np.sort(refseries), dist_name, args=param)                      # Applying the Kolmogorov-Smirnov test

        dist_results.append((dist_name, p))          

    # find best fitting statistical distribution on reference data
    best_dist, best_p = (max(dist_results, key=lambda item: item[1]))           # Select the best fitted distribution
    dist = getattr(stats, str(best_dist))                                  
    rv = dist(*params[best_dist])
    #best_dist='gamma' # force on gamma dist
    # fit historic time series over best distribution    
    if best_dist == 'gamma':                                                    
        nyears_zero = len(fullseries) - np.count_nonzero(fullseries)
        p_zero = nyears_zero / len(fullseries)
        p_zero_mean = (fullseries + 1) / (2 * (len(fullseries) + 1))          
        ppd = (fullseries.copy() * 0 ) + p_zero_mean
        if len(fullseries[np.nonzero(fullseries)]) > 0:
            ppd[np.nonzero(fullseries)] = p_zero+((1-p_zero)*rv.cdf(fullseries[np.nonzero(fullseries)]))                
    else:
        ppd = rv.cdf(fullseries)

    Zval = stats.norm.ppf(ppd)                                 
    Zval[Zval>3] = 3
    Zval[Zval<-3] = -3
    #Zval[np.isnan(Zval)] = 0  

    return (Zval, best_dist)


def calculate_index(values, Years_his, Years, Months, Index, accumulation_period):
    
    indices = np.zeros(len(values))        
    best_dist_monthly=[]
    for m in range(len(Months)):    
              
        monthlyvalues = np.zeros(len(Years))                      
        for yr in range(len(Years)):                             
            monthlyvalues[yr] = values[(12*yr)+m]                                 
                                                
        # extract reference time series and calculate Z values         
        if accumulation_period<24:
            Z,best_dist = calculate_Zval(monthlyvalues[1:len(Years_his)],monthlyvalues, Index) # 1 added to not use the NAN values to calculate dist
        else:
            Z,best_dist = calculate_Zval(monthlyvalues[2:len(Years_his)],monthlyvalues, Index) # 1 added to not use the NAN values to calculate dist
        # Reconstruct time series        
        for yr in range(len(Years)):
            indices[(12*yr)+m] = Z[yr]
        best_dist_monthly.append(str(best_dist))
        
    return (indices, best_dist_monthly)


#%% with Xclim (tested. Gives similar results! )
#pr=P_HAD_DN.tp #input rainfall
# pr_test=pr.isel(longitude=1).isel(latitude=1)
# pr_test.attrs["units"] = "mm/day" # spi needs a unit as attribute
# pr_cal=pr_test.sel(time=slice(datetime(1981, 1, 1), datetime(2021, 12, 31))) #reference/calibration period. Should be at least 30 years: https://edo.jrc.ec.europa.eu/documents/factsheets/factsheet_spi_ado.pdf

# # Define accumulation periods
# periods = [1, 3, 6, 12, 24, 48]

# # Calculate SPI values
# for period in periods:
#     spi = standardized_precipitation_index(pr_test, pr_cal, freq='MS',window=24, method='APP',dist='gamma')

# spi=spi.to_dataframe(name='tp')




#%% plot netCDF dataset
def plot_xarray_dataset(dataset, var, lat, lon, title, unit_factor, time_aggregation_method): 
    
    ###################################### read data #############################################
    #os.chdir(folder)
    dataset=dataset#make dataset a callable variable
    dataset=dataset.where((dataset.latitude > -4.5) & (dataset.latitude<14.9) & (dataset.longitude > 33.0) &(dataset.longitude < 51) , drop=True)
    if time_aggregation_method=='mean': 
        if 'number' in dataset.dims:
            dataset=dataset.median(dim='number')
            value = dataset.variables[var]#.mean(dim='step')
        else:
            value = dataset.variables[var]#.mean(dim='time')            
    elif time_aggregation_method=='sum':
        if 'number' in dataset.dims:
            dataset=dataset.median(dim='number')
            value = dataset.variables[var].sum(dim='step')
        else:
            value = dataset.variables[var].sum(dim='time')
    elif time_aggregation_method=='median':
        if 'number' in dataset.dims:
            dataset=dataset.median(dim='number')
            value = dataset.variables[var].median(dim='step')
        else:
            value = dataset.variables[var].median(dim='time')
    elif time_aggregation_method=='max':
        if 'number' in dataset.dims:
            dataset=dataset.median(dim='number')
            value = dataset.variables[var].max(dim='step')
        else:
            value = dataset.variables[var].max(dim='time')
    else: 
        value=dataset.variables[var]
        
    ######################################## plot ################################################   
    fig=plt.figure(figsize=(12,12))
    lats = dataset.variables[lat][:]
    lons = dataset.variables[lon][:]
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    plt.contourf(lons,lats, value*unit_factor,30,
                 transform=ccrs.PlateCarree(), vmin=0, vmax=200,cmap=cm.Blues)
    #ax.plot(37.568358,0.341406, w'bo', markersize=7, transform=ccrs.PlateCarree())
    m = plt.cm.ScalarMappable(cmap=cm.Blues)
    m.set_array(value*unit_factor)
    m.set_clim(0., 200.)
    cbar=plt.colorbar(m, boundaries=np.linspace(0, 200, 11))
    cbar.ax.tick_params(labelsize=20) 
    ax.coastlines()
    plt.title(title, size=20)
    
    ax.add_feature(cartopy.feature.BORDERS)
    plt.show()


#%% Facet plot from netCDF   
def plot_xr_facet_seas5(dataset, var, vmin, vmax, TOI, cmap, cbar_label, title):
    dataset=dataset
    country_borders = cfeature.NaturalEarthFeature(
      category='cultural',
      name='‘admin_0_boundary_lines_land',
      scale='50m',
      facecolor='none')
    proj0=ccrs.PlateCarree(central_longitude=0)
    fig=plt.figure(figsize=(20,20))# (W,H)
    
    gs=fig.add_gridspec(3,3,wspace=0,hspace=0.1)
    ax1=fig.add_subplot(gs[0,0],projection=proj0) # 2:,2:
    ax2=fig.add_subplot(gs[0,1],projection=proj0)
    ax3=fig.add_subplot(gs[0,2],projection=proj0)
    ax4=fig.add_subplot(gs[1,0],projection=proj0)
    ax5=fig.add_subplot(gs[1,1],projection=proj0)
    ax6=fig.add_subplot(gs[1,2],projection=proj0)
    ax7=fig.add_subplot(gs[2,0],projection=proj0)

    lead0=dataset.isel(lead=0)[var].plot.pcolormesh(ax=ax1,transform=ccrs.PlateCarree(central_longitude=0),add_colorbar=False,vmin=vmin, vmax=vmax,cmap=cmap)
    lead1=dataset.isel(lead=1)[var].plot.pcolormesh(ax=ax2,transform=ccrs.PlateCarree(central_longitude=0),add_colorbar=False,vmin=vmin, vmax=vmax,cmap=cmap)
    lead2=dataset.isel(lead=2)[var].plot.pcolormesh(ax=ax3,transform=ccrs.PlateCarree(central_longitude=0),add_colorbar=False,vmin=vmin, vmax=vmax,cmap=cmap)
    lead3=dataset.isel(lead=3)[var].plot.pcolormesh(ax=ax4,transform=ccrs.PlateCarree(central_longitude=0),add_colorbar=False,vmin=vmin, vmax=vmax,cmap=cmap)
    lead4=dataset.isel(lead=4)[var].plot.pcolormesh(ax=ax5,transform=ccrs.PlateCarree(central_longitude=0),add_colorbar=False,vmin=vmin, vmax=vmax,cmap=cmap)
    lead5=dataset.isel(lead=5)[var].plot.pcolormesh(ax=ax6,transform=ccrs.PlateCarree(central_longitude=0),add_colorbar=False,vmin=vmin, vmax=vmax,cmap=cmap)
    lead6=dataset.isel(lead=6)[var].plot.pcolormesh(ax=ax7,transform=ccrs.PlateCarree(central_longitude=0),add_colorbar=False,vmin=vmin, vmax=vmax,cmap=cmap)
       
 
    
    #%%%ax1 
    lead=str(int(round((float(dataset.isel(lead=0).lead.values)/(2592000000000000)),0)))
    ax1.set_title('lead=%s months'%(lead), size=20)
    gl = ax1.gridlines(crs=proj0, draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = True
    gl.bottom_labels=False
    gl.xlabel_style = {'color': 'gray'}
    gl.ylabel_style = {'color': 'gray'}
    ax1.coastlines()
    ax1.add_feature(cfeature.BORDERS)
    

    #%%%ax2 
    lead=str(int(round((float(dataset.isel(lead=1).lead.values)/(2592000000000000)),0)))
    ax2.set_title('lead=%s months'%(lead), size=20)
    gl = ax2.gridlines(crs=proj0, draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = False
    gl.bottom_labels=False
    gl.xlabel_style = {'color': 'gray'}
    gl.ylabel_style = {'color': 'gray'}
    ax2.coastlines()
    ax2.add_feature(cfeature.BORDERS)
    
    #%%%ax3
    lead=str(int(round((float(dataset.isel(lead=2).lead.values)/(2592000000000000)),0)))
    ax3.set_title('lead=%s months'%(lead), size=20)
    gl = ax3.gridlines(crs=proj0, draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = True
    gl.left_labels = False
    gl.bottom_labels=False
    gl.xlabel_style = {'color': 'gray'}
    gl.ylabel_style = {'color': 'gray'}
    ax3.coastlines()
    ax3.add_feature(cfeature.BORDERS)
 
    #%%%ax4
    lead=str(int(round((float(dataset.isel(lead=3).lead.values)/(2592000000000000)),0)))
    ax4.set_title('lead=%s months'%(lead), size=20)
    gl = ax4.gridlines(crs=proj0, draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = True
    gl.bottom_labels=False
    gl.xlabel_style = {'color': 'gray'}
    gl.ylabel_style = {'color': 'gray'}
    ax4.coastlines()
    ax4.add_feature(cfeature.BORDERS)
    #%%%ax5
    lead=str(int(round((float(dataset.isel(lead=4).lead.values)/(2592000000000000)),0)))
    ax5.set_title('lead=%s months'%(lead), size=20)
    
    gl = ax5.gridlines(crs=proj0, draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = False
    gl.bottom_labels=True
    gl.xlabel_style = {'color': 'gray'}
    gl.ylabel_style = {'color': 'gray'}
    ax5.coastlines()
    ax5.add_feature(cfeature.BORDERS)
    #%%%ax6
    lead=str(int(round((float(dataset.isel(lead=5).lead.values)/(2592000000000000)),0)))
    ax6.set_title('lead=%s months'%(lead),size=20)
    
    gl = ax6.gridlines(crs=proj0, draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = True
    gl.left_labels = False
    gl.xlabel_style = {'color': 'gray'}
    gl.ylabel_style = {'color': 'gray'}
    ax6.coastlines()
    ax6.add_feature(cfeature.BORDERS) 
    #%%%ax7
    lead=str(int(round((float(dataset.isel(lead=6).lead.values)/(2592000000000000)),0)))
    ax7.set_title('lead=%s months'%(lead), size=20)
    gl = ax7.gridlines(crs=proj0, draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = True
    gl.left_labels = True
    gl.xlabel_style = {'color': 'gray'}
    gl.ylabel_style = {'color': 'gray'}
    ax7.coastlines()
    ax7.add_feature(cfeature.BORDERS)
    
    #%%% formatting 
    cax1= fig.add_axes([0.5,0.2,0.3,0.02]) #[left, bottom, width, height]
    cbar=plt.colorbar(lead6,pad=0.05,cax=cax1,orientation='horizontal',cmap=cmap)
    cbar.set_label(label=cbar_label, size='large', weight='bold')
    cbar.ax.tick_params(labelsize=15) 

    plt.suptitle(title, fontsize=15, fontweight='bold')#x=0.54, y=0.1
    fig.tight_layout()
    plt.show()
    fig.savefig(os.path.join(path,'plots\\python\\facet_%s_%s.pdf'%(var,TOI)), bbox_inches='tight')


######################## function to plot netcdf data ##############################
def plot_xarray_array(dataset, lat, lon, title, folder, unit_factor, time_aggregation_method): 
    
    ###################################### read data #############################################
    dataset=dataset#make dataset a callable variable
    dataset=dataset.where((dataset.latitude > -4.5) & (dataset.latitude<14.9) & (dataset.longitude > 33.0) &(dataset.longitude < 51) , drop=True)
    if time_aggregation_method=='mean': 
        if 'number' in dataset.dims:
            dataset=dataset.median(dim='number')
            value = dataset.mean(dim='step')
        else:
            value = dataset.mean(dim='time')            
    elif time_aggregation_method=='sum':
        if 'number' in dataset.dims:
            dataset=dataset.median(dim='number')
            value = dataset.sum(dim='step')
        else:
            value = dataset.sum(dim='time')
    elif time_aggregation_method=='median':
        if 'number' in dataset.dims:
            dataset=dataset.median(dim='number')
            value = dataset.median(dim='step')
        else:
            value = dataset.median(dim='time')
    elif time_aggregation_method=='max':
        if 'number' in dataset.dims:
            dataset=dataset.median(dim='number')
            value = dataset.max(dim='step')
        else:
            value = dataset.max(dim='time')
    else: 
        value=dataset
        
    ######################################## plot ################################################   
    lats = dataset.latitude
    lons = dataset.longitude
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    plt.contourf(lons,lats, value*unit_factor,
                 transform=ccrs.PlateCarree())
    ax.plot(37.568358,0.341406, 'bo', markersize=7, transform=ccrs.PlateCarree())
    ax.coastlines()
    plt.title(title, size=10)
    cbar = plt.colorbar()
    ax.add_feature(cartopy.feature.BORDERS)
    plt.show()