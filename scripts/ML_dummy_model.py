# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 15:33:24 2023

@author: tbr910

This script makes use of the following scripts and input data 

1. Pure drought impact data: impact_df_creation.py --> impacts_county_level.xlsx
2. Meteo data: meteo_vars.py --> input_data2.xlsx
3. ML model: ML_dummy_model.py --> ML_dummy_model.py

Functions are being called from ML_functions.py and function_def.py. This needs to be merged to one script.

"""

#%% 
import os
import random as python_random
import numpy as np
import matplotlib.pyplot as plt
import datetime 
from dateutil.relativedelta import relativedelta
import pandas as pd
import seaborn as sns
os.chdir('C:\\Users\\tbr910\\Documents\\Forecast_action_analysis')
from ML_functions import *

#%% Machine learning model training
#import tensorflow as tf`
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit,RandomizedSearchCV

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, explained_variance_score
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from statsmodels.tsa.stattools import adfuller
#%%Visualizations
from sklearn.tree import export_graphviz
from sklearn import tree


#%% import sklearn metrics
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import plot_roc_curve

from sklearn.metrics import roc_auc_score
from sklearn.metrics import plot_precision_recall_curve


## feature selection 
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression
import pydot
from graphviz import Source
from sklearn.feature_selection import RFE

## FEATURE EXPLANATION
import shap

shap.initjs() # load JS visualization code to notebook
#from keras.layers import Activation
#from keras.layers import BatchNormalization
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense
# from keras.callbacks import ModelCheckpoint
#from tensorflow.keras.layers import Dropout
#from scikeras.wrappers import KerasRegressor
#import lightgbm as lgb
#from tensorflow.keras import regularizers




################## define master variables 
Aggregation_level=['all','county','cluster'] #cluster,'county'
Aggregation_level= ['county']

feature_engineering=True #without feature engineering the errors are much higher (almost 2x)   
forecast=True
fill_nans_target=False
leads=[0,1,2,3,4,8,12] # check:  12 lead creates quite some nan's in the training data 
random_split=False
hp_tuning=False
with_NDMA=True
with_TC=True
split_by_time=True


for aggregation in Aggregation_level:
    counties=['Garissa','Isiolo','Mandera','Marsabit','Samburu','Tana River','Turkana','Wajir','Baringo','Kajiado','Kilifi','Kitui','Laikipia','Makueni','Meru','Taita Taveta','Tharaka','West Pokot','Lamu','Nyeri','Narok']

    #input data 
    DATA_FOLDER = 'C:\\Users\\tbr910\\Documents\\Forecast_action_analysis\\'
    CLIM_FOLDER = 'C:\\Users\\tbr910\\Documents\\ML\\SE_data\\climate_vars'
    SE_FOLDER= 'C:\\Users\\tbr910\\Documents\\ML\\SE_data\\'
    # results 
    BASE_FOLDER = 'C:\\Users\\tbr910\\Documents\\ML\\'
    RESULT_FOLDER = 'C:\\Users\\tbr910\\Documents\\ML\\ML_results\\'
    
    # Make subfolders according to the aggregation level 
    
    PLOT_FOLDER= 'C:\\Users\\tbr910\\Documents\\ML\\ML_results\\plots\\'+aggregation
    TREE_FOLDER= 'C:\\Users\\tbr910\\Documents\\ML\\trees\\'+aggregation

    # check if the PLOT_FOLDER AND TREE FOLDER EXCIST, if not create them
    if not os.path.exists(PLOT_FOLDER):
        os.makedirs(PLOT_FOLDER)
    else: 
        # remove all content from plot folder
        files = os.listdir(PLOT_FOLDER)
        for f in files:
            os.remove(os.path.join(PLOT_FOLDER, f))
        
    if not os.path.exists(TREE_FOLDER):
        os.makedirs(TREE_FOLDER)
    else:
        # remove all content from tree folder
        files = os.listdir(TREE_FOLDER)
        for f in files:
            os.remove(os.path.join(TREE_FOLDER, f))




    os.chdir(BASE_FOLDER)

        
    df=pd.read_excel('input_data2.xlsx', index_col=0) # This df is until 2021-12, update with 2022 data! 
    df.drop('GP',axis=1, inplace=True)

    input_df=pd.DataFrame()

    stats_df=pd.DataFrame(columns=('county', 'accuracy', 'accuracy_baseline', 'accuracy_baseline2','accuracy_lr', 'var_score','var_score_baseline','var_score_baseline2','var_score_lr', 'mae', 'mae_baseline', 'mae_baseline2','mae_lr', 'mse', 'mse_baseline', 'mse_baseline2', 'mse_lr', 'lead'))
    features_df=pd.DataFrame(columns=('county', 'feature', 'feature_imp', 'lead'))             
    features_df_full=pd.DataFrame()
    best_params_df=pd.DataFrame()




    ############################################# get extra data #############################################

    ###################### HUMANITARIAN DATA ######################
    os.chdir(DATA_FOLDER+'\\impacts\\FEWS')
    
    ###################### ACLED ###################### https://acleddata.com/data/acled-api/
    #for codebook, see https://acleddata.com/acleddatanew/wp-content/uploads/2021/11/ACLED_Codebook_v1_January-2021.pdf
    os.chdir(SE_FOLDER)
    acled=pd.read_csv([i for i in os.listdir() if 'acled' in i][0], index_col=0)
    acled_kenya=acled[acled['country']=='Kenya']
    # extract event date, admin1, and fatalities (later maybe also source scale?) #assoc_actor_1 contains pastoralists!! May be of use 
    acled_kenya=acled_kenya[['event_date','admin1','fatalities']]
    # datetime management 
    acled_kenya['event_date']=pd.to_datetime(acled_kenya['event_date'])
    acled_kenya.set_index('event_date',inplace=True)
    acled_kenya.rename(columns={'admin1':'county','fatalities':'acled_fatalities'}, inplace=True)
    
    ##################### FSNAU for Somalia ##################### https://data.humdata.org/dataset/fsna-2019-2020-somalia


    ###################### Soil Moisture (root_zone and surface) ######################

    os.chdir(CLIM_FOLDER+'\\Gleam')   

    sm_root=pd.read_excel([i for i in os.listdir() if ('SMroot' in i) & ('.xlsx' in i)][0], index_col=0)
    sm_surf=pd.read_excel([i for i in os.listdir() if ('SMsurf' in i) & ('.xlsx' in i)][0], index_col=0)


    ######################  food prices ###################### https://data.humdata.org/dataset/wfp-food-prices-for-kenya
    os.chdir(SE_FOLDER)
    food_prices= pd.read_csv('food_prices.csv', index_col=0, header=1, delimiter=',')
    # get header names from dataframe and simplify the names
    food_prices.columns = food_prices.columns.str.replace('#','')
    food_prices.columns = food_prices.columns.str.replace('+','_')
    food_prices.rename(columns={'adm2_name':'county'}, inplace=True)

    #keep only county, item_name, iten_price_flag, value_usd 
    food_prices=food_prices[['county','item_name','item_price_flag','value_usd']]
    # keep only rows with 'actual' and 'aggregate' price flags, and when the item_name contains 'Maize' 
    food_prices=food_prices[((food_prices['item_price_flag']=='actual') | (food_prices['item_price_flag']=='aggregate')) & (food_prices['item_name'].str.contains('Maize'))]
    #food_prices=food_prices[(food_prices['item_price_flag']=='actual') & (food_prices['item_name'].str.contains('Maize'))]


    # rename value_usd to maize_price and drop item_name and item_price_flag
    food_prices.rename(columns={'value_usd':'maize_price'}, inplace=True)
    food_prices.drop(['item_name','item_price_flag'], axis=1, inplace=True)

    #convert date to datetime index and set the day to first day in the month 
    food_prices.index=pd.to_datetime(food_prices.index)
    food_prices.index=food_prices.index.to_period('M').to_timestamp('D', 'start')


    ############################################# SST INDEX #############################################

    ########################## load SST index data ##########################
    if with_TC==True: 
        # WVG processed sheet
        WVG=pd.read_excel(CLIM_FOLDER+'\\SST_index.xlsx', sheet_name='WVG_processed', index_col=0, header=0)

        # shift WVG one row forward 
        WVG=WVG.shift(1)
        # For WVG, ffill the missing values 
        WVG.fillna(method='ffill', inplace=True)
        # MEI sheet 
        MEI=pd.read_excel(CLIM_FOLDER+'\\SST_index.xlsx', sheet_name='MEI', index_col=0, header=0)
        # remove rows with values <-10
        MEI=MEI[MEI['MEI']>-10]
        # NINA34 sheet
        NINA34=pd.read_excel(CLIM_FOLDER+'\\SST_index.xlsx', sheet_name='NINA34', index_col=0, header=0)
        # remove rows with values <-10
        NINA34=NINA34[NINA34['NINA34']>-10]
        # IOD sheet
        IOD=pd.read_excel(CLIM_FOLDER+'\\SST_index.xlsx', sheet_name='IOD', index_col=0, header=0)
        # convert index to datetime
        IOD.index=pd.to_datetime(IOD.index)

    ######################  population ###################### https://data.humdata.org/dataset/kenya-population-statistics-2019

    # population= pd.read_excel('ken_admpop_2019.xlsx',sheet_name='ken_admpop_ADM1_2019',index_col=0, header=0)
    # total_pop= population[['ADM1_NAME','T_TL']].set_index('ADM1_NAME')


    # extract the feature names from the dataframe and convert to dataframe with nice names in new column 
    feature_names=pd.DataFrame(df.columns, columns=['feature'])




    ############################# ############ Hyper parameters tuning #########################################
    # Script at the end of the file 

    #####################################################################################################################
    ################################################### INPUT DATAFRAME ###################################################
    #####################################################################################################################

    #  This dataframe will consist of all features/labels for all counties and all leads

    for i,county in enumerate(counties): 

        ################################### create empty lead dataframe ###################################
        features_l=pd.DataFrame() 

        print (county)
        print(i)

        features= df[df['county']==county]#['FEWS_CS']#[~df['FEWS_CS'].isnull()]

        # fill na for indicators (debatable!)
        NDMA_indicators=['MPR', 'CP', 'MP', 'HDW', 'LDW', 'MUAC'] #'GP'
        if with_NDMA==False: 
            features.drop(NDMA_indicators, axis=1, inplace=True)


        ########################## SST data ##########################
        if with_TC==True:
            features=features.merge(WVG, how='left', left_index=True, right_index=True)# merge WVG to features df
            features=features.merge(MEI, how='left', left_index=True, right_index=True)# merge MEI to features df
            features=features.merge(NINA34, how='left', left_index=True, right_index=True)# merge NINA34 to features df
            features=features.merge(IOD, how='left', left_index=True, right_index=True)# merge IOD to features df


        ############################################# Food prices #############################################
        

        # merge food prices for a county with the features dataframe
        food_prices_county=food_prices[food_prices['county']==county]
        # sort index 
        food_prices_county.sort_index(inplace=True)
        # mean values for same datetime index 
        food_prices_county=food_prices_county.groupby(food_prices_county.index).mean()
        # print list of missing months in the index
        #print(food_prices_county.index[~food_prices_county.index.isin(features.index)])
        
        # add food prices to features as a new column, for the same datetime index. Keep all rows in features, even if there is no food price data for that month
        features=features.merge(food_prices_county, how='left', left_index=True, right_index=True)


        ############################################# soil moisture #############################################
        sm_root_county=sm_root[county]
        sm_surf_county=sm_surf[county]
        # rename pandas series to 'sm_root'
        sm_root_county.rename('sm_root', inplace=True)
        sm_surf_county.rename('sm_surf', inplace=True)

        # add soil moisture to features as a new column
        features=features.merge(sm_root_county, how='left', left_index=True, right_index=True)
        features=features.merge(sm_surf_county, how='left', left_index=True, right_index=True)


        ############################################# Conflicts  #############################################
        acled_county=acled_kenya[acled_kenya['county']==county]
        acled_county['acled_count']=1
        acled_county=acled_county.resample('MS').sum()
        
        # merge with features
        if len(acled_county.values>0):   
            features=features.merge(acled_county, how='left', left_index=True, right_index=True)
        
        

        ############################################# NAN values processing #############################################

    
        #make new SE_indicators variable based on whether there is SE data for the county or not
        columns_to_impute= [column for column in features.columns if column not in ['FEWS_CS', 'county', 'aridity']]

        for column in columns_to_impute:
            features[column].fillna((features[column].mean()), inplace=True) # CHECK IF THIS IS CORRECT


        if fill_nans_target==True: #Default: False
            #features['FEWS_CS']=features['FEWS_CS'].interpolate(method='time',limit_area='inside')
            features['FEWS_CS']=features['FEWS_CS'].fillna(method="ffill")
        

        # create dataframe column called 'FEWS_CS_baseline2' with baseline2 float   numbers (based on min and max of FEWS_CS)
        #features['FEWS_CS_baseline2']= np.baseline2.uniform(features['FEWS_CS'].min(), features['FEWS_CS'].max(), size=len(features))
        





        #################################################### feature engineering ####################################################
    

        if feature_engineering==True:
            ############################################# ROLLING MEANS  #############################################
            # NDVI, WD, DS, MAIZE PRICE, soil moisture
            features['NDVI_roll_mean']=features['NDVI'].rolling(window=4).mean().shift(1)
            features['NDVI_range_roll_mean']=features['NDVI_range'].rolling(window=4).mean().shift(1)
            features['NDVI_crop_roll_mean']=features['NDVI_crop'].rolling(window=4).mean().shift(1)
            features['wd_roll_mean']=features['wd'].rolling(window=4).mean().shift(1)
            features['ds_roll_mean']=features['ds'].rolling(window=4).mean().shift(1)
            features['sm_root_roll_mean']=features['sm_root'].rolling(window=4).mean().shift(1)
            features['sm_surf_roll_mean']=features['sm_surf'].rolling(window=4).mean().shift(1)
            

            features['NDVI_roll_mean_12']=features['NDVI'].rolling(window=12).mean().shift(1)
            features['NDVI_range_roll_mean_12']=features['NDVI_range'].rolling(window=12).mean().shift(1)
            features['NDVI_crop_roll_mean_12']=features['NDVI_crop'].rolling(window=12).mean().shift(1)
            features['wd_roll_mean_12']=features['wd'].rolling(window=12).mean().shift(1)
            features['ds_roll_mean_12']=features['ds'].rolling(window=12).mean().shift(1)
            features['sm_root_roll_mean_12']=features['sm_root'].rolling(window=12).mean().shift(1)
            features['sm_surf_roll_mean_12']=features['sm_surf'].rolling(window=12).mean().shift(1)
            
            if len(acled_county.values>0):
                features['acled_count_roll_mean']=features['acled_count'].rolling(window=4).mean().shift(1)
                features['acled_count_roll_mean_12']=features['acled_count'].rolling(window=12).mean().shift(1)
                features['acled_fatalities_roll_mean']=features['acled_fatalities'].rolling(window=4).mean().shift(1)
                features['acled_fatalities_roll_mean_12']=features['acled_fatalities'].rolling(window=12).mean().shift(1)

            # maize prices 
            if food_prices_county.empty:
                pass
            else:
                features['maize_price_roll_mean']=features['maize_price'].rolling(window=4).mean().shift(1)
                features['maize_price_roll_mean_12']=features['maize_price'].rolling(window=12).mean().shift(1)
                

            # SST indicators (WVG, IOD, MEI, NINA34)
            if with_TC==True: 
                
                features['WVG_roll_mean']=features['WVG'].rolling(window=4).mean().shift(1)
                features['IOD_roll_mean']=features['IOD'].rolling(window=4).mean().shift(1)
                features['MEI_roll_mean']=features['MEI'].rolling(window=4).mean().shift(1)
                features['NINA34_roll_mean']=features['NINA34'].rolling(window=4).mean().shift(1)

                features['WVG_roll_mean_12']=features['WVG'].rolling(window=12).mean().shift(1)
                features['IOD_roll_mean_12']=features['IOD'].rolling(window=12).mean().shift(1)
                features['MEI_roll_mean_12']=features['MEI'].rolling(window=12).mean().shift(1)
                features['NINA34_roll_mean_12']=features['NINA34'].rolling(window=12).mean().shift(1)

            ############################################# FILL NANs #############################################

            # fill nan values that came out of rolling
            features['NDVI_roll_mean'].fillna((features['NDVI_roll_mean'].mean()), inplace=True) # check: fillna with mean is incorrect! 
            features['NDVI_range_roll_mean'].fillna((features['NDVI_range_roll_mean'].mean()), inplace=True)
            features['NDVI_crop_roll_mean'].fillna((features['NDVI_crop_roll_mean'].mean()), inplace=True)
            features['wd_roll_mean'].fillna((features['wd_roll_mean'].mean()), inplace=True)
            features['ds_roll_mean'].fillna((features['ds_roll_mean'].mean()), inplace=True)
            features['sm_root_roll_mean'].fillna((features['sm_root_roll_mean'].mean()), inplace=True)
            features['sm_surf_roll_mean'].fillna((features['sm_surf_roll_mean'].mean()), inplace=True)
            

            features['NDVI_roll_mean_12'].fillna((features['NDVI_roll_mean_12'].mean()), inplace=True)
            features['NDVI_range_roll_mean_12'].fillna((features['NDVI_range_roll_mean_12'].mean()), inplace=True)
            features['NDVI_crop_roll_mean_12'].fillna((features['NDVI_crop_roll_mean_12'].mean()), inplace=True)
            features['wd_roll_mean_12'].fillna((features['wd_roll_mean_12'].mean()), inplace=True)
            features['ds_roll_mean_12'].fillna((features['ds_roll_mean_12'].mean()), inplace=True)
            features['sm_root_roll_mean_12'].fillna((features['sm_root_roll_mean_12'].mean()), inplace=True)
            features['sm_surf_roll_mean_12'].fillna((features['sm_surf_roll_mean_12'].mean()), inplace=True)
            

            if with_TC==True: 
                # fill nan values that came out of rolling
                features['WVG_roll_mean'].fillna((features['WVG_roll_mean'].mean()), inplace=True)
                features['IOD_roll_mean'].fillna((features['IOD_roll_mean'].mean()), inplace=True)
                features['MEI_roll_mean'].fillna((features['MEI_roll_mean'].mean()), inplace=True)
                features['NINA34_roll_mean'].fillna((features['NINA34_roll_mean'].mean()), inplace=True)

                features['WVG_roll_mean_12'].fillna((features['WVG_roll_mean_12'].mean()), inplace=True)
                features['IOD_roll_mean_12'].fillna((features['IOD_roll_mean_12'].mean()), inplace=True)
                features['MEI_roll_mean_12'].fillna((features['MEI_roll_mean_12'].mean()), inplace=True)
                features['NINA34_roll_mean_12'].fillna((features['NINA34_roll_mean_12'].mean()), inplace=True)

            if len(food_prices_county)>0:
                # fill nan values that came out of rolling for maize price
                features['maize_price_roll_mean'].fillna((features['maize_price_roll_mean'].mean()), inplace=True)# check: incorrect nan filling. Fill with obs 
                features['maize_price_roll_mean_12'].fillna((features['maize_price_roll_mean_12'].mean()), inplace=True)

            if len(acled_county)>0:
                features['acled_count_roll_mean'].fillna((features['acled_count_roll_mean'].mean()), inplace=True) 
                features['acled_count_roll_mean_12'].fillna((features['acled_count_roll_mean_12'].mean()), inplace=True)
                features['acled_fatalities_roll_mean'].fillna((features['acled_fatalities_roll_mean'].mean()), inplace=True)
                features['acled_fatalities_roll_mean_12'].fillna((features['acled_fatalities_roll_mean_12'].mean()), inplace=True)

        
        # drop nan values when whole column is nan (CROP column)
        features.dropna(axis=1, how='all', inplace=True)



        #################################################### Extract MAM and OND seasons ####################################################
        features['month']=features.index.month
        features['day']=features.index.day
        features['year']=features.index.year

        # add OND and MAM rainy season
        # OND --> months 10,11,12, flag in a new column with boolean values
        features['OND']=((features['month']==10) | (features['month']==11) | (features['month']==12))
        # MAM --> months 3,4,5, flag in a new column with boolean values
        features['MAM']=((features['month']==3) | (features['month']==4) | (features['month']==5))

        # make a seperate dataframe with day, month, year columns 
        features_date=features[['day','month','year']]

        # drop day, month, year columns from features
        features.drop(['day','month','year'], axis=1, inplace=True)


        if feature_engineering==True:
            #################################################### feature engineering for FEWS ####################################################
            #create a forward-filled column for FEWS_CS
            features['base_ini']=features['FEWS_CS'].fillna(method='ffill')
            
            # FEWS_CS lags, and fill nan values with mean of the column
            features['FEWS_CS_lag1']=features['base_ini'].shift(1)
            features['FEWS_CS_lag1'].fillna((features['base_ini'].mean()), inplace=True)

            features['FEWS_CS_lag2']=features['base_ini'].shift(2)
            features['FEWS_CS_lag2'].fillna((features['base_ini'].mean()), inplace=True)

            features['FEWS_CS_lag3']=features['base_ini'].shift(3)
            features['FEWS_CS_lag3'].fillna((features['base_ini'].mean()), inplace=True)
            
            features['FEWS_CS_lag4']=features['base_ini'].shift(4) # CHECK: FILL NANS WITH MEAN IS INCORRECT? --> FILL MEANS WITH JUST THE FEWS OBS COLUMN! 
            features['FEWS_CS_lag4'].fillna((features['base_ini'].mean()), inplace=True)    

            # create new variables from rolling mean of existing features, where the preceding 4 months are used to calculate the mean. Do not include the current month 
            features['FEWS_CS_roll_mean']=features['base_ini'].rolling(window=4).mean().shift(1)
            features['FEWS_CS_roll_mean'].fillna((features['base_ini'].mean()), inplace=True)
            

        # One-hot encode the data using pandas get_dummies
        features = pd.get_dummies(features) 
        
        


        #################################################### Add lead times to dataframe ####################################################
        if forecast==True:
            # Shift features by the lead (1,4,8,12) months. This trains and tests the model with features from the past, creating a lag to predict the future.
            for lead in leads:
                if lead==0:
                    features_l=features.copy()


                else: 
                    features_l=features.copy()
                    # shift the features by the lead. Does NOT include FEWS_CS and base_ini columns
                    shift_columns=features_l.columns[~features_l.columns.isin(['FEWS_CS'])]
                    # shift only the columns that are not FEWS_CS 
                    features_l[shift_columns]=features_l[shift_columns].shift(lead)
                    
                    # remove the last X rows based on the lead
                    features_l=features_l.drop(features_l.index[:lead])
                
                # column management 
                features_l=features_l.loc[:, ~features_l.columns.str.contains('^county')]
                features_l=features_l.loc[:, ~features_l.columns.str.contains('^aridity')]
                features_l['lead']=lead
                features_l['county']=county
                ############################# Create base forecast, to be used later #############################
                features_l.rename(columns={'base_ini': 'base_forecast'}, inplace=True)
                # add all leads individually to the input dataframe
                input_df=pd.concat([input_df,features_l], axis=0)

    ############################################# Load clusters ##############################################   
    clusters=pd.read_csv(BASE_FOLDER+'\\clusters.csv', index_col=0)
    # add column to input df with cluster
    input_df['cluster']=input_df['county'].map(clusters['max'])

    #################################################### END OF DATAFRAME CONSTRUCTION  ####################################################
    #########################################################################################################################
                
            

    #################################################### Fill NAN values which are NAN for all timesteps in county #####################################
    # for labels --> extra crop/range nans are created due to the stacking of rows in the input_df

    crop_cols=[col for col in input_df.columns if 'crop' in col]
    if len(crop_cols)>0:
        input_df[crop_cols]=input_df[crop_cols].fillna(input_df.NDVI_crop.mean()) # fill nans of crop columns with mean of NDVI_crop column
    # range ndvi columns
    range_cols=[col for col in input_df.columns if 'range' in col]
    if len(range_cols)>0:
        input_df[range_cols]=input_df[range_cols].fillna(input_df.NDVI_range.mean()) # fill nans of range columns with mean of NDVI_range column

    # maize prices
    maize_cols=[col for col in input_df.columns if 'maize' in col]
    if len(maize_cols)>0:
        input_df[maize_cols]=input_df[maize_cols].fillna(input_df.maize_price.mean()) # fill nans of maize columns with mean of maize_price column

    #acled columns 
    acled_cols=[col for col in input_df2.columns if 'acled' in col]
    if len(acled_cols)>0:
        input_df[acled_cols]=input_df[acled_cols].fillna(0) # fill nans of acled columns with 0
    
    # check which input_df columns have at least 1 NAN values 
    #nan_cols=input_df.columns[input_df.isna().any()].tolist()


    #####################################################################################################################
    ################################################### ML algorithms ###################################################
    #####################################################################################################################
    if aggregation=='cluster':
        cluster_list=list(clusters['max'].unique())
    ############# set aggregation level for ML training/testing############ 
    elif aggregation=='all':
        counties=['all']
        cluster_list= ['no_cluster'] 

    elif aggregation=='county':
        counties=list(input_df['county'].unique())
        cluster_list=['no_cluster']


    for cluster in cluster_list:
        
        if aggregation=='cluster':
            input_df2=input_df[input_df['cluster']==cluster]
            input_df2.dropna(axis=1, how='all', inplace=True)
            counties=['cluster_%s'%cluster]

        for i, county in enumerate(counties):
            print('County: ', county, 'in cluster:', cluster)
            # read in data

            if aggregation=='county': 
                input_df2=input_df[input_df['county']==county]
                input_df2.dropna(axis=1, how='all', inplace=True)# drop nan values when whole column is nan (CROP column)
            elif aggregation=='all':
                input_df2=input_df.copy()
            
            


            ##################################################### Keep only timesteps with FEWS data #####################################

            # For label (FEWS_CS)
            input_df2=input_df2[input_df2['FEWS_CS'].notna()] # no nans 


            
            

            ############################################### START LEAD TIME LOOP ###############################################
            
            if forecast==True: 
                for lead in leads: 
                    ############################################### Deleting NANs ###############################################


                    ############################################### Select lead ###############################################
                    input_df3=input_df2[input_df2['lead']==lead]
                    


                    
                    ############################################### Sort by date ###############################################
                    if split_by_time==True:
                        # sort on datetime index 
                        input_df3=input_df3.sort_index()
                    
                    ############################################### Save county dataframe ##############################################
                    county_df=input_df3['county']

                    ############################################### Base prediction ###############################################
                    base1_preds=input_df3['base_forecast']

                    ############################################### Create features- labels ###############################################
                    labels=pd.Series(input_df3['FEWS_CS'])
                    features=input_df3.drop(['FEWS_CS','lead', 'base_forecast', 'county', 'cluster'], axis=1)

                    feature_list = list(features.columns)

                    ############################################### Training-test split ###############################################            
                    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.20,shuffle=False) #25% of the data used for testing (38 time steps)   random_state = 42. if baseline2 state is not fixed, performance is different each time the code is run.
                    print ('training years = %s-%s, testing years= %s-%s' %(train_features.index.year[0],train_features.index.year[-1],test_features.index.year[0],test_features.index.year[-1]))
                    
                    

                    ############################################### Correlation analysis ###################################   
                    # correlations= features_l.drop(features_l.columns[-2:], axis=1).corr()

                    # mask = []
                    # for i in range(len(correlations.columns)):
                    #     mask_i = []
                    #     for j in range(len(correlations.columns)):
                    #         if i<j:
                    #             mask_i.append(True)
                    #         else: 
                    #             mask_i.append(False)
                    #     mask.append(mask_i)
                    # mask = np.array(mask)

                    # plt.figure(figsize=(20, 20), facecolor='w', edgecolor='k')
                    # sns.set(font_scale=1.2)
                    # sns.heatmap(correlations,cmap='coolwarm',
                    #             center = 0, 
                    #             annot=True,
                    #             fmt='.1g',
                    #             annot_kws={"fontsize":8},
                    #             mask=mask)
                    
                    # plt.title('Correlation matrix for %s months lead, county= %s'%(lead, county))
                    


                    

                    ############ create original fews_base values in shape of train_features and test_features ############# 
                    # fews_base_original=features['FEWS_base']

                    
                    # # original fews_base of the months present in train/test features 
                    # fews_base_original_train=fews_base_original.iloc[:len(train_features)] 
                    # fews_base_original_test=fews_base_original.iloc[-len(test_features):]

                    ############ create dates for training and testing set #############
                    train_dates=pd.DataFrame({'day':train_features.index.day, 'month':train_features.index.month, 'year':train_features.index.year})
                    test_dates=pd.DataFrame({'day':test_features.index.day, 'month':test_features.index.month, 'year':test_features.index.year})

                    ############ convert to numpy arrays #############
                    train_features_np = np.array(train_features)
                    test_features_np = np.array(test_features)
                    train_labels_np = np.array(train_labels)
                    test_labels_np = np.array(test_labels)
                    #fews_base_original_np=np.array(fews_base_original)
                    # fews_base_original_l_np=np.array(fews_base_original_l) ### CHECK: NOT USED
                    #fews_base_original_train_np=np.array(fews_base_original_train) ### CHECK: NOT USED
                    #fews_base_original_test_np=np.array(fews_base_original_test)
                    train_dates_np=np.array(train_dates) #real original time stamps 
                    test_dates_np=np.array(test_dates) #real original time stamps 

                    
                    ############################################### Baseline models ###############################################


                    ############################################### Baseline 1: use base_forecast as prediction ###############################################
                    # extract the base_forecast column from the test_features dataframe
                    base1_preds=base1_preds.iloc[-len(test_features):]



                    # plt.plot(base1_preds, label='Predictions')
                    # plt.plot(base1_truth, label='Truth')
                    # plt.plot(base1_errors, label='Errors')
                    # plt.legend()
                    # plt.show()

                    ############################################### Baseline 2: use train set seasonality as baseline ###############################################
                    # take training values of fews, for index with years < 2016. Calculate seasonality based on that (after 2015 there is a change in the months included in the fews datset)
                    
                    if aggregation=='all' or 'cluster': 
                        # reset index 
                        train_labels_means=train_labels.reset_index()
                        # calculate mean over same values in the 'index' column
                        train_labels_means=train_labels_means.groupby('index').mean()
                        train_labels_means=train_labels_means['FEWS_CS']
                    else: 
                        train_labels_means=train_labels.copy() 

                    train_set_seasonality=train_labels_means.loc[train_labels_means.index.year<train_features.index.year[-1]]
                    seasonality=train_set_seasonality.groupby(train_set_seasonality.index.month).mean()
                    
                    # reindex the seasonality dataframe to the months of the year
                    seasonality=seasonality.reindex([1,2,3,4,5,6,7,8,9,10,11,12,1])

                    # interpolate the values using interpolation 
                    seasonality=seasonality.interpolate(method='linear', axis=0)
                    # make seasonality one row shorter (remove the last row!)
                    seasonality=seasonality.iloc[:-1]

                    # Assign these monthly values (from seasonality) to the test set, based on the month of the index
                    seasonality=seasonality.reindex(test_labels.index.month) #CHECK THIS LINE

                    # reset the index of the seasonality dataframe  
                    seasonality=seasonality.reset_index(drop=True)
                    # set the index of the seasonality dataframe to the index of the fews_base_original_test dataframe
                    seasonality.index=test_labels.index
                    

                    base2_preds= seasonality.copy()
                    base2_preds=pd.Series(base2_preds)



                    ############################################### Linear Regression ###############################################
                    regr = LinearRegression()
                    regr.fit(train_features_np, train_labels)
                    # Make predictions using the testing set
                    lr_preds = regr.predict(test_features_np)
                    # The coefficients
                    regr.score(test_features_np, test_labels)
                    regr.coef_
                    
                    ############################################### Random Forest ###############################################
                    # MIN SAMPLES LEAF =1 IS DANGEROUS! MAX DEPTH PRESENT OVERFITTING. MAX DEPTH=5 can still overfit (See lecture)
                    rf = RandomForestRegressor(n_estimators = 700,max_features='sqrt', n_jobs=-1, max_depth=5, min_samples_leaf=2, min_samples_split=5, random_state=40) # joris also uses max features  max_features='sqrt'
                    rf.get_params()

                    # Train the model on training data
                    model=rf.fit(train_features_np, train_labels)
                    
                    # Use the forest's predict method on the test data
                    predictions = rf.predict(test_features) ## rf predictions for the months with lead ahead 
                    
                    
                    ########################################## Visualize trees --> use dtreeviz package in future?  ###################################
                
                    #Pull out one tree from the forest
                    os.chdir(TREE_FOLDER)
                    tree = rf.estimators_[5]# extracts 1 random tree from the forest
                    export_graphviz(tree, out_file = 'tree_L%s_%s.dot'%(lead,county), feature_names = feature_list, rounded = True, precision = 3)# Use dot file to create a graph
                    (graph, ) = pydot.graph_from_dot_file('tree_L%s_%s.dot'%(lead,county))# Write dot file to a png file
                    graph.write_png('tree_L%s_%s.png'%(lead,county)) 
                    # remove the dot file
                    os.remove('tree_L%s_%s.dot'%(lead,county))


                    ############################# feature importances #############################
                    ############# SCI-Kit Learn ################

                    # Get feature importances -->  computed as the mean and standard deviation of accumulation of the impurity decrease within each tree.
                    importances = list(rf.feature_importances_)# List of tuples with variable and importance
                    
                    std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)

                    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]# Sort the feature importances by most important first
                    
                    # append featuer importances to df
                    # make a dataframe with 3 columns: importance, county, lead. Importances is a np.array. Attach the county and lead to the array and then make a df from it.
                    append= pd.DataFrame(importances, index=feature_list, columns=['importance'])
                    append['county']=county
                    append['lead']=lead
                    features_df_full=pd.concat([features_df_full, append], axis=0)

                    # sort feature importances and print most important feature
                    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
                    main_feature= feature_importances[0][0]
                    feature_imp= feature_importances[0][1]

                    ########### SHAP ################ Lundberg and Lee (2016) 
                    #Since we want the global importance, we average the absolute Shapley values per feature across the data:
                    explainer = shap.Explainer(model) # or TreeExplainer? Gives same result?
                    shap_explainer = explainer(train_features) # input what is used to fit the model 
                    shap_vals=explainer.shap_values(train_features)


                    ############################################### Plotting section ###############################################
                    os.chdir(PLOT_FOLDER)
                    ################# Plot SHAP values #################
                    
                    # summarize plot 
                    #shap.summary_plot(shap_values, features=features_l, feature_names=features_l.columns, plot_type='bar',title="Feature importance for %s, L%s"%(county,lead), show=False)
                    #plt.savefig(PLOT_FOLDER+'\\shap_summary_plot_L%s_%s.png'%(lead,county), bbox_inches='tight')
                    #plt.close()

                    # dependence plot
                    ranks=[0,1,2,3,4]# Rank 1,2,3,4
                    
                    for rank in ranks:
                        shap.dependence_plot('rank(%s)'%rank, shap_vals, train_features, show=False)
                        plt.title('Dependence plot for %s, L%s'%(county,lead))
                        plt.savefig('shap_dependence_plot_L%s_%s_rank%s.png'%(lead,county,rank), bbox_inches='tight')
                        plt.close()

                    # beeswarm plot
                    shap.plots.beeswarm(shap_explainer, show=False)
                    plt.title('Beeswarm plot for %s, L%s'%(county,lead))
                    plt.savefig('shap_beeswarm_plot_L%s_%s.png'%(lead,county), bbox_inches='tight')
                    plt.close()

                    ################# Plot feature importances #################

                    ############################################### Change DIR to PLOT FOLDER ###############################################
                    

                    #  Variable importances
                    forest_importances = pd.Series(importances, index=feature_list)
                    # sort importances
                    forest_importances.sort_values(ascending=False, inplace=True)

                    # keep only top 15 features
                    forest_importances=forest_importances[:20]

                    fig, ax = plt.subplots()
                    forest_importances.plot.bar(ax=ax) # yerr=std
                    
                    
                    ax.set_title("Feature importances using MDI for %s, L%s"%(county,lead))
                    ax.set_ylabel("Mean decrease in impurity")
                    ax.set_ylim(0,0.5)
                    fig.tight_layout()
                    plt.savefig('Variable_Importances_%s_L%s.png'%(county,lead), dpi=300, bbox_inches='tight')
                    plt.show()
                    plt.close()

                    ################# reconstruct dataframe with predictions. Datestamps retrieved are "real" original datestampes from features df #################           
                    ## attach county_df as column to train_features
                    train_features['county']= county_df.iloc[:len(train_features)] # iloc works on a slice of the dataframe based on position! 
                    test_features['county']= county_df.iloc[-len(test_features):] # iloc works on a slice of the dataframe based on position! 
                    
                    test_features.reset_index(inplace=True) # reset index to get the original datestamps
                    train_features.reset_index(inplace=True) # reset index to get the original datestamps
                    test_labels=pd.DataFrame(test_labels)
                    test_labels.reset_index(inplace=True) # reset index to get the original datestamps
                    train_labels=pd.DataFrame(train_labels)
                    train_labels.reset_index(inplace=True) # reset index to get the original datestamps

                    labels=pd.DataFrame(labels)
                    labels.reset_index(inplace=True) # reset index to get the original datestamps
                    
                    # predictions 
                    predictions= pd.DataFrame(predictions)
                    lr_preds= pd.DataFrame(lr_preds)
                    
                    # base predictions 
                    base1_preds= pd.DataFrame(base1_preds).reset_index().drop('index', axis=1)
                    base2_preds= pd.DataFrame(base2_preds).reset_index().drop('index', axis=1)
                    
                    # counties 
                    county_df2=pd.DataFrame(county_df) # counties from original input_df 
                    county_df2.reset_index(inplace=True) # reset index to get the original datestamps
                    # base predictions
                    
                    ############################################### Filter predictions per county  ###############################################
                    for c in list(county_df.unique()):
                        ####################### select county data  ########################### 
                        # labels 
                        labels_county=labels.iloc[county_df2[county_df2['county']==c].index] # select county labels 
                        test_labels_county=test_labels.iloc[test_features[test_features['county']==c].index].set_index('index') # select county test labels
                        
                        # features with lead
                        features_county=features.iloc[county_df2[county_df2['county']==c].index] # select county features
                        test_features_county=test_features.iloc[test_features[test_features['county']==c].index].set_index('index') # select county test features
                        


                        
                        
                        


                        # predictions 
                        predictions_county=predictions.iloc[test_features[test_features['county']==c].index] # select county predictions
                        lr_preds_county=lr_preds.iloc[test_features[test_features['county']==c].index]# select county predictions
                        
                        # Base predictions 
                        base1_preds_county=base1_preds.iloc[test_features[test_features['county']==c].index] # select county base predictions
                        base2_preds_county=base2_preds.iloc[test_features[test_features['county']==c].index] # select county base predictions                

                        
                        predictions_data = pd.DataFrame(data = {'date': test_labels_county.index, 'prediction': predictions_county.values.flatten(),'lr':lr_preds_county.values.flatten(), 'base1_preds':base1_preds_county.values.flatten(), 'base2_preds':base2_preds_county.values.flatten()}) # Dataframe with predictions and dates

                        ######################### Evaluation #############################
                
                        # note: test_labels are for X months ahead, according to the lead
                        test_values=test_labels_county.values.flatten() 
                        ############################# general errors #############################
                        base1_errors = abs(predictions_data['base1_preds'] - test_values)
                        base2_errors = abs(predictions_data['base2_preds'] - test_values)        
                        lr_errors = abs(predictions_data['lr'] - test_values)
                        errors = abs(predictions_data['prediction'] - test_values)
                                    
                        ############################# accuracy #############################
                        # Calculate mean absolute percentage error (MAPE) 
                        mape = 100 * (errors / test_values) 
                        mape_baseline1= 100 * (base1_errors / test_values)
                        mape_baseline2= 100 * (base2_errors / test_values)
                        mape_lr= 100 * (lr_errors / test_values)
                        
                        # Calculate and display accuracy
                        accuracy = 100 - np.mean(mape)
                        accuracy_baseline= 100 - np.mean(mape_baseline1)
                        accuracy_baseline2= 100 - np.mean(mape_baseline2)
                        accuracy_lr= 100 - np.mean(mape_lr)

                        ############################# r2 score #############################

                        # Explained variance score: 1 is perfect prediction
                        var_score=explained_variance_score(test_values, predictions_data['prediction'])
                        var_score_baseline= explained_variance_score(test_values, predictions_data['base1_preds'])
                        var_score_baseline2= explained_variance_score(test_values, predictions_data['base2_preds'])
                        var_score_lr= explained_variance_score(test_values, predictions_data['lr'])

                        ############################# mean absolute error #############################
                        mae = mean_absolute_error(test_values, predictions_data['prediction'])
                        mae_baseline= mean_absolute_error(test_values, predictions_data['base1_preds'])
                        mae_baseline2= mean_absolute_error(test_values, predictions_data['base2_preds'])
                        mae_lr= mean_absolute_error(test_values, predictions_data['lr'])

                        ############################# mean squared error #############################
                        mse = mean_squared_error(test_values, predictions_data['prediction'])
                        mse_baseline= mean_squared_error(test_values, predictions_data['base1_preds'])
                        mse_baseline2= mean_squared_error(test_values, predictions_data['base2_preds'])
                        mse_lr= mean_squared_error(test_values, predictions_data['lr'])

                        ############################# root mean squared error ############################# 
                        rmse = np.sqrt(mean_squared_error(test_values, predictions_data['prediction']))
                        rmse_baseline= np.sqrt(mean_squared_error(test_values, predictions_data['base1_preds']))
                        rmse_baseline2= np.sqrt(mean_squared_error(test_values, predictions_data['base2_preds']))
                        rmse_lr= np.sqrt(mean_squared_error(test_values, predictions_data['lr']))


                        ############################### Evaluation stats per county ########################################
                        var_list= [c,round(accuracy, 2), round(accuracy_baseline, 2), round(accuracy_baseline2, 2), round(accuracy_lr,2), round(var_score, 2), round(var_score_baseline, 2),round(var_score_baseline2, 2), round(var_score_lr,2), round(mse,2), round(mse_baseline,2), round(mse_baseline2,2), round(mse_lr,2), round(mae,2), round(mae_baseline,2), round(mae_baseline2,2), round(mae_lr,2),lead]         
                        stats_df.loc[len(stats_df), :] = var_list

                        ############################################### Plot predictions ###############################################

                        #load target obs
                        target_obs=labels_county.set_index('index').copy()
                        #load feature obs
                        #extract names from 2 most important features from feature importance tuples 

                        fig, ax = plt.subplots(figsize=(10, 5))
                        # Plot the actual values
                        plt.plot(target_obs.index,target_obs['FEWS_CS'], 'b-', label = 'Observed FEWS class')# Plot the predicted values
                        
                        # plot base predictions
                        plt.plot(predictions_data['date'], predictions_data['base1_preds'], 'go', label = 'base1 prediction (future based on last observed value)')
                        plt.plot(predictions_data['date'], predictions_data['base2_preds'], 'yo', label = 'base2 prediction (future based on train data seasonality)')
                        
                        # plot rf predictions
                        plt.plot(predictions_data['date'], predictions_data['prediction'], 'ro', label = 'Random Forest prediction')

                        #plot lr predictions
                        #plt.plot(predictions_data['date'], predictions_data['lr'], 'mo', label = 'Linear regression prediction')
                        
                        plt.xticks(rotation = '60'); 
                        plt.xlabel('Date'); plt.ylabel('FEWS IPC class'); plt.title('FEWS observations vs RF predictions for L=%s, county= %s. Accuracy=%s'%(lead,c,round(accuracy, 2)));
                        plt.legend(loc='best')
                        plt.savefig('TS_%s_L%s.png'%(c,lead), dpi=300, bbox_inches='tight')
                        plt.show() 
                        plt.close()

                        #Select features --> CHECK
                        feature1, feature2=feature_importances[0][0], feature_importances[1][0]
                        
                        # lead features
                        feature_lead=features_county[[feature1,feature2]]
                        feature_lead=feature_lead.loc[feature_lead.index>=target_obs.index[0]]

                        # observed features
                        features_plot= input_df[input_df['county']==c]
                        features_plot= features_plot[features_plot['lead']==0]
                        features_plot=features_plot[[feature1,feature2]]
                        features_plot=features_plot.loc[features_plot.index>=target_obs.index[0]]

                        ################# Plot explanatory line graph #################
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax2=ax.twinx()
                        
                        # Plot the actual values
                        ax.plot(target_obs.index,target_obs['FEWS_CS'], 'b-', label = 'Observed FEWS class')# Plot the predicted values
                        # plot rf predictions 
                        ax.plot(predictions_data['date'], predictions_data['prediction'], 'ro', label = 'random Forest prediction')
                        
                        # plot base predictions         

                        # plot lead features
                        #ax2.plot(feature_lead.index, feature_lead[feature1], 'o-', color='orange', label = feature1+'_lead')# Plot the predicted values
                        #ax2.plot(feature_lead.index, feature_lead[feature2], 'o-', color='black', label = feature2+'+lead')# Plot the predicted values
                        
                        # plot observed features
                        ax2.plot(features_plot.index, features_plot[feature1], 'o-', color='purple', label = feature1+'_obs')# Plot the predicted values
                        ax2.plot(features_plot.index, features_plot[feature2], 'o-', color='blue', label = feature2+'_obs')# Plot the predicted values
                        
                        plt.xticks(rotation = '60'); 
                        plt.xlabel('Date'); plt.ylabel('Magnitude of features'); plt.title('Explanatory_plot for %s,L=%s. Accuracy=%s'%(c,lead,round(accuracy, 2)));
                        # axis label on ax axes
                        ax.set_ylabel('FEWS IPC class')
                        plt.legend()
                        plt.savefig('Explanatory_plot_%s_L%s.png'%(c,lead), dpi=300, bbox_inches='tight')
                        
                        plt.show() 
                        plt.close()
                



    ################################################# feature plots ###############################################


    ################### feature importance plot per lead time (averaged over counties) ###################

    fig=plt.figure(figsize=(10, 5)) 
    # plot feature importance per lead time
    for lead in leads:
        # get feature importance for lead time
        features_df_lead=features_df_full[features_df_full['lead']==lead]
        # reset index 
        features_df_lead=features_df_lead.reset_index()
        # get mean feature importance per feature
        means=features_df_lead.groupby('index').mean()
        # rename importance column to mean
        means.rename(columns={'importance':'mean'}, inplace=True)
        # sort values
        means=means.sort_values(by='mean', ascending=False)
        
        if aggregation=='county':
            # get stdev of feature importance over counties (stdev indicates spread between countries)
            stdev=features_df_lead.groupby('index').std()
            # rename importance column to stdev
            stdev.rename(columns={'importance':'stdev'}, inplace=True)

            # merge stdev and mean on column with index as name 
            means=means.merge(stdev['stdev'], left_index=True, right_index=True)
        else: 
            means['stdev']=0
        # plot feature importance per lead time
        plt.figure(figsize=(10, 5))
        plt.bar(means.index, means['mean'], yerr=means['stdev'], align='center', label='L=%s'%(lead))
        plt.xticks(rotation = '90');
        plt.xlabel('Feature'); plt.ylabel('Feature importance'); plt.title('Feature importance for lead: %s'%(lead));
        
        # x limite x-axis to 0 and 0.5 
        plt.ylim(0, 0.6)
        plt.legend(loc='best')
        plt.savefig('feature_importance_lead_%s.png'%(lead), dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()


    ############################################## evaluation plots ####################################
    # convert all negative values in the df to zero 
    column_names_eval = ['accuracy', 'accuracy_baseline', 'accuracy_baseline2', 'accuracy_lr', 'var_score', 'var_score_baseline', 'var_score_baseline2', 'var_score_lr', 'mse', 'mse_baseline', 'mse_baseline2', 'mse_lr', 'mae', 'mae_baseline', 'mae_baseline2', 'mae_lr']

    for column in column_names_eval:
        stats_df.loc[stats_df[column]<0, column]=0

    var_vars=['var_score', 'var_score_baseline', 'var_score_baseline2', 'var_score_lr']
    mse_vars=['mse', 'mse_baseline', 'mse_baseline2', 'mse_lr']
    mae_vars=['mae', 'mae_baseline', 'mae_baseline2', 'mae_lr']
    accuracy_vars=['accuracy', 'accuracy_baseline', 'accuracy_baseline2', 'accuracy_lr']

    # plot var scores (var_vars) for all counties for all leads 
    for lead in leads:
        # subset stats_df for lead
        stats_df_lead=stats_df.loc[stats_df['lead']==lead]
        # melt the df to get a df with county, variable and var_score
        stats_df_lead_melt=pd.melt(stats_df_lead, id_vars=['county'], value_vars=var_vars, var_name='variable', value_name='var_score')
        # plot the df with seaborn
        fig=plt.figure(figsize=(10, 5))
        sns.barplot(x='county', y='var_score', hue='variable', data=stats_df_lead_melt)
        plt.title('Variance explained for lead %s'%(lead))
        plt.xticks(rotation=90)
        plt.tight_layout()
        # save plot in plots folder
        plt.savefig('R2_lead_%s.png'%(lead), dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

    # plot mse scores (mse_vars) for all counties for all leads
    for lead in leads:
        # subset stats_df for lead
        stats_df_lead=stats_df.loc[stats_df['lead']==lead]
        # melt the df to get a df with county, variable and mse
        stats_df_lead_melt=pd.melt(stats_df_lead, id_vars=['county'], value_vars=mse_vars, var_name='variable', value_name='mse')
        # plot the df with seaborn
        fig=plt.figure(figsize=(10, 5))
        sns.barplot(x='county', y='mse', hue='variable', data=stats_df_lead_melt)
        plt.title('Mean squared error for lead %s'%(lead))
        plt.xticks(rotation=90)
        plt.tight_layout()
        # save plot in plots folder
        plt.savefig('MSE_lead_%s.png'%(lead), dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()


    # plot var_vars for all leads averaged over all counties 

    # melt the df to get a df with lead, variable and var_score
    stats_df_melt=pd.melt(stats_df, id_vars=['lead'], value_vars=var_vars, var_name='variable', value_name='var_score')
    # plot the df with seaborn, lead on x-axis
    fig=plt.figure(figsize=(10, 5))
    sns.barplot(x='lead', y='var_score', hue='variable', data=stats_df_melt)
    plt.title('Variance explained')
    plt.xticks(rotation=90)
    plt.ylim(0, 0.6)
    plt.tight_layout()
    # save plot in plots folder
    plt.savefig('R2_all_leads.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # plot mse_vars for all leads averaged over all counties

    # melt the df to get a df with lead, variable and var_score
    stats_df_melt=pd.melt(stats_df, id_vars=['lead'], value_vars=mae_vars, var_name='variable', value_name='mae_score')
    # plot the df with seaborn, lead on x-axis
    fig=plt.figure(figsize=(10, 5))
    sns.barplot(x='lead', y='mae_score', hue='variable', data=stats_df_melt)
    plt.title('MAE')
    plt.xticks(rotation=90)
    plt.ylim(0,1)
    plt.tight_layout()
    # save plot in plots folder
    plt.savefig('MAE_all_leads.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()



    # spatial plot 
    # for i,lead in enumerate(leads): 
    #     # get lead 2 var scores
    #     stats_df_lead2=stats_df.loc[stats_df['lead']==lead][['var_score','county']]

    #     # add the geometries to the stats_df_lead2 df based on the county name
    #     gdf_scores=pd.merge(stats_df_lead2, county_sf, on='county')[['county', 'var_score', 'geometry']]

    #     # convert to geodataframe
    #     gdf_scores=gpd.GeoDataFrame(gdf_scores, geometry='geometry')
    #     gdf_scores=gdf_scores.set_crs(epsg=4326)

    #     # plot the geodataframe, with a fixed colorbar
    #     fig, ax = plt.subplots(1, figsize=(10, 10))
    #     gdf_scores.plot(column='var_score', ax=ax, legend=True, vmin=0,vmax=0.5, cmap='OrRd')
    #     ax.set_title('Variance explained for lead %s'%(lead))
    #     ax.set_axis_off()
    #     plt.tight_layout()
    #     plt.show()
    #     # save plot in plots folder


    #%% save results
    #make new folder for results
    os.chdir(RESULT_FOLDER)

    #save results in new folder     
    stats_df.to_csv('stats_df2_%s.csv'%(aggregation))
    #features_df.to_csv('features_df2.csv')




# ####################################################################################### SHAP ###############################################################
# (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)


# explainer = shap.Explainer(model)
# shap_values = explainer(features)

# # visualize the first prediction's explanation
# shap.plots.waterfall(shap_values[0])

# # create a dependence scatter plot to show the effect of a single feature across the whole dataset
# shap.plots.scatter(shap_values[:,"SPI6"], color=shap_values)

# # summarize the effects of all the features
# shap.plots.beeswarm(shap_values)
    
# shap.plots.scatter(shap_values[:,"FEWS_CS_lag1"])

# shap.summary_plot(shap_values, features=features, feature_names=features.columns, plot_type="bar")



# # convert to list
# values=series.values
# values=list(values)
# #series = pd.read_csv('daily-total-female-births.csv', header=0, index_col=0)

# # transform the time series data into supervised learning
# # n_in=Number of lag observations as input (X). Values may be between [1..len(data)] Optional. Defaults to 1. 
# # n_out: Number of observations as output (y). Values may be between [0..len(data)-1]. Optional. Defaults to 1.
# data2 = series_to_supervised(values, 2,2) 
# # evaluate
# mae, y, yhat = walk_forward_validation(data, 12)
# n_test=12
# print('MAE: %.3f' % mae)
# # plot expected vs predicted
# pyplot.plot(y, label='Expected')
# pyplot.plot(yhat, label='Predicted')
# pyplot.legend()
# pyplot.show()





    # Grid specs draft part

    # if grid_search==True: 
    #     tss = TimeSeriesSplit(n_splits = 5, test_size=5)# 10-fold cross validation

    #     for split, (train_index, test_index) in enumerate(tss.split(features_l)):
    #         # features
    #         train_features, test_features = features_l.iloc[train_index, :],features_l.iloc[test_index,:]
    #         # labels 
    #         train_labels, test_labels = labels_l.iloc[train_index], labels_l.iloc[test_index]
    #         # time 
    #         train_time, test_time = features_date_l.iloc[train_index, :],features_date_l.iloc[test_index,:]
    #         # convert to numpy array
    #         # features
    #         feature_list2=feature_list.copy()
    #         features_np = np.array(features_l)
    #         train_features_np=np.array(train_features)
    #         test_features_np=np.array(test_features)
    #         # time
    #         features_date_np=np.array(features_date_l)
    #         train_time_np=np.array(train_time)
    #         test_time_np=np.array(test_time)


    # from sklearn.model_selection import GridSearchCV
    # import warnings
    # warnings.filterwarnings("ignore") #ignore all warnings in this cell

    # parametergrid = { 
    #     'n_estimators': [25, 50, 100],
    #     'max_features': ['sqrt'],
    #     'max_depth' : [5,7,9],
    #     'random_state' : [18]
    # }

    # ## Grid Search function
    # CV_rf = GridSearchCV(estimator=RandomForestRegressor(), param_grid=parametergrid, cv= 4, scoring = 'neg_mean_squared_error')
    # CV_rf.fit(x_train, y_train)
    
    #rf = RandomForestClassifier(criterion = 'entropy', random_state = 42)
    
    
    #labels_stack=pd.concat([labels,labels_stack], axis=0)

    # drop county_Garissa and aridity_arid columns
    # corr=features.drop(['county_Garissa','aridity_arid'], axis=1)
    # corr=corr.corr()
    # plt.figure(figsize=(20,20))
    # sns.heatmap(corr,annot=False, fmt='.2f', cmap='coolwarm', xticklabels=corr.columns, yticklabels=corr.columns)
    # plt.title('Correlation matrix')
    # #save in plots folder in results
    # plt.savefig(RESULT_FOLDER + '/plots/correlation_matrix.png')

    # plt.show()



      
    ############################################### Feature selection ###############################################


    # # feature selection with selectkbest test
    #features_kbest = SelectKBest(f_regression, k=10).fit_transform(features, labels)
    #features_kbest = pd.DataFrame(features_kbest)
    
    # this function will take in X, y variables 
    # with criteria, and return a dataframe
    # with most important columns
    # based on that criteria
    # def featureSelect_dataframe(X, y, criteria, k):

    #     # initialize our function/method
    #     reg = SelectKBest(criteria, k=k).fit(X,y)
        
    #     # transform after creating the reg (so we can use getsupport)
    #     X_transformed = reg.transform(X)

    #     # filter down X based on kept columns
    #     X = X[[val for i,val in enumerate(X.columns) if reg.get_support()[i]]]

    #     # return that dataframe
    #     return X

    # features = featureSelect_dataframe(features, labels, mutual_info_regression, 10)





# #%% scriopt henrique 
# def calibration(X_origin,y_origin,type_of_model='RF', test_year = None):
#     X, y = shuffle(X_origin, y_origin, random_state=0)
#     if test_year == None:
#         # Shuffle and Split data
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#     else:
#         X_train, y_train = X.loc[X.index.get_level_values('time') != test_year], y.loc[y.index.get_level_values('time') != test_year]
#         X_test, y_test = X.loc[X.index.get_level_values('time') == test_year], y.loc[y.index.get_level_values('time') == test_year]
       
#     if type_of_model == 'RF':
#         model_rf = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1,
#                                           max_depth = 20, max_features = 'auto',
#                                           min_samples_leaf = 1, min_samples_split=2)
        
#         full_model_rf = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1,
#                                           max_depth = 20, max_features = 'auto',
#                                           min_samples_leaf = 1, min_samples_split=2)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
#     elif type_of_model == 'lightgbm':
#         model_rf = Pipeline([
#             ('scaler', StandardScaler()),
#             ('estimator', lgb.LGBMRegressor(linear_tree= True, max_depth = 20, num_leaves = 50, min_data_in_leaf = 100, 
#                                             random_state=0, learning_rate = 0.01, n_estimators = 1000 ) )
#         ])
        
        
#         full_model_rf = Pipeline([
#             ('scaler', StandardScaler()),
#             ('estimator', lgb.LGBMRegressor(linear_tree= True, max_depth = 20, num_leaves = 50, min_data_in_leaf = 100, 
#                                             random_state=0, learning_rate = 0.01, n_estimators = 1000 ) )
#         ])
    
#     elif type_of_model == 'DNN':
#     #     	extra layer	batch	epoch	nodes	lr	dropout_value	best_epoch	R2	MAE	RMSE
#     #   31	True	          256	700	512	0.01	0.2	514	0.651	0.24641	0.34128
#     #   1	False	     1024	700	512	0.01	0.2	564	0.648	0.24794	0.34267
#     #   30	True	          256	700	512	0.005	0.2	635	0.648	0.24502	0.34266
#     # number of epochs 529 or 431
#         epochs_train = 441 #390
#         batch_size_train = 1024
#         nodes_size = 512
#         learning_rate_train = 0.01
#         dropout_train = 0.2
#         regul_value = 0
#         # =============================================================================
#         #      #   TRAIN model 
#         # =============================================================================
#         def create_model():
#             train_model = Sequential()
#             train_model.add(Dense(nodes_size, input_dim=len(X_train.columns), kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(regul_value))) 
#             train_model.add(BatchNormalization())
#             train_model.add(Activation('relu'))
#             train_model.add(Dropout(dropout_train))
    
#             train_model.add(Dense(nodes_size, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(regul_value)))
#             train_model.add(BatchNormalization())
#             train_model.add(Activation('relu'))
#             train_model.add(Dropout(dropout_train))
    
#             train_model.add(Dense(nodes_size, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(regul_value)))
#             train_model.add(BatchNormalization())
#             train_model.add(Activation('relu'))
#             train_model.add(Dropout(dropout_train))
           
#             train_model.add(Dense(nodes_size, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(regul_value)))
#             train_model.add(BatchNormalization())
#             train_model.add(Activation('relu'))
#             train_model.add(Dropout(dropout_train))
            
#             train_model.add(Dense(nodes_size, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(regul_value)))
#             train_model.add(BatchNormalization())
#             train_model.add(Activation('relu'))
#             train_model.add(Dropout(dropout_train))
            
#             train_model.add(Dense(1, activation='linear'))
            
#             # compile the keras model
#             train_model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate_train), metrics=['mean_squared_error','mean_absolute_error'])
#             return train_model
        
#         # Callbacks to monitor the performance of the optimization of the model and if there is any overfitting
#         # callback_model = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience= 100, restore_best_weights=True)
#         # mc = ModelCheckpoint('best_model_test.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)
        
#         model_rf = Pipeline([
#             ('scaler', StandardScaler()),
#             ('estimator', KerasRegressor(model=create_model(), epochs= epochs_train, random_state = 0, batch_size=batch_size_train, verbose=0)) ]) #, callbacks=callback_model #, validation_split= 0.1, callbacks=[callback_model,mc]
        
#         model_fit = model_rf.fit(X_train, y_train)
        
#         # =============================================================================
#         #         # Entire full set model
#         # =============================================================================
#         full_model_rf = Pipeline([
#             ('scaler', StandardScaler()),
#             ('estimator', KerasRegressor(model=create_model(), epochs = epochs_train, random_state = 0, batch_size = batch_size_train, verbose=0)) ]) # validation_split= 0.1, callbacks=[callback_model_full, mc_full]
        
#         model_fit_full = full_model_rf.fit(X, y)

        
#     if type_of_model == 'DNN':
#         # Wrap up and plot graphs
#         model = model_fit
        
#         full_model = model_fit_full
        
#     else:
    
#         model = model_rf.fit(X_train, y_train)
        
#         full_model = full_model_rf.fit(X, y)
    
#     # Test performance
#     y_pred = model.predict(X_test)
#     df_y_pred = pd.DataFrame(y_pred, index = y_test.index, columns = [y_test.name]) 
    
#     # report performance
#     print(f'Results for model: {type_of_model}')
#     print("R2 on test set:", round(r2_score(y_test, y_pred),2))
#     print("Var score on test set:", round(explained_variance_score(y_test, y_pred),2))
#     print("MAE on test set:", round(mean_absolute_error(y_test, y_pred),5))
#     print("RMSE on test set:",round(mean_squared_error(y_test, y_pred, squared=False),5))
#     print("______")
    
#     y_pred_total = full_model.predict(X)
#     df_y_pred_total = pd.DataFrame(y_pred_total, index = y.index, columns = [y.name]) 
#     df_y_pred_total = df_y_pred_total.sort_index()
#     # Display error
#     plt.figure(figsize=(5,5), dpi=250) #plot clusters
#     plt.scatter(y_test, y_pred)
#     plt.plot(y_test, y_test, color = 'black', label = '1:1 line')
#     plt.ylabel('Predicted yield')
#     plt.xlabel('Observed yield')
#     plt.title('Scatter plot - test set')
#     plt.legend()
#     # plt.savefig('paper_figures/???.png', format='png', dpi=500)
#     plt.show()
    
#     # Display error
#     plt.figure(figsize=(5,5), dpi=250) #plot clusters
#     plt.scatter(y, y_pred_total)
#     plt.plot(y, y, color = 'black', label = '1:1 line')
#     plt.ylabel('Predicted yield')
#     plt.xlabel('Observed yield')
#     plt.title('Scatter plot - total set')
#     plt.legend()
#     # plt.savefig('paper_figures/???.png', format='png', dpi=500)
#     plt.show()
   
#     return df_y_pred, df_y_pred_total, model, full_model 

###################################################################################################################################
########################################################## HP TUNING ##########################################################
###################################################################################################################################
    # if hp_tuning==True: 
    #     for i,county in enumerate(counties): 
    #         print (county)
    #         print(i)
    #         features= df[df['county']==county]#['FEWS_CS']#[~df['FEWS_CS'].isnull()]
            
    #         #Fill na for indicators (debatable!)
    #         NDMA_indicators=['MPR', 'CP', 'MP', 'HDW', 'LDW', 'MUAC'] #'GP'
    #         if with_NDMA==False: 
    #             features.drop(NDMA_indicators, axis=1, inplace=True)


    #         ########################## merge SST data with features df ##########################
    #         if with_TC==True: 
    #             features=features.merge(WVG, how='left', left_index=True, right_index=True)# merge WVG to features df
    #             features=features.merge(MEI, how='left', left_index=True, right_index=True)# merge MEI to features df
    #             features=features.merge(NINA34, how='left', left_index=True, right_index=True)# merge NINA34 to features df
    #             features=features.merge(IOD, how='left', left_index=True, right_index=True)# merge IOD to features df


    #         ############################################# Merge food prices #############################################
            

    #         # merge food prices for a county with the features dataframe
    #         food_prices_county=food_prices[food_prices['county']==county]
    #         # sort index 
    #         food_prices_county.sort_index(inplace=True)
    #         # mean values for same datetime index 
    #         food_prices_county=food_prices_county.groupby(food_prices_county.index).mean()
    #         # print list of missing months in the index
    #         #print(food_prices_county.index[~food_prices_county.index.isin(features.index)])
            
            
    #         # add food prices to features as a new column, for the same datetime index. Keep all rows in features, even if there is no food price data for that month
    #         features=features.merge(food_prices_county, how='left', left_index=True, right_index=True)




    #         ############################################# NAN values processing #############################################

            
    #         # linear imputation FEWS in between the not nan values 
    #         #
            
    #         if fill_nans_target==True:
    #             #features['FEWS_CS']=features['FEWS_CS'].interpolate(method='time',limit_area='inside')
    #             features['FEWS_CS']=features['FEWS_CS'].fillna(method="ffill")
            
            
    #         # create dataframe column called 'FEWS_CS_baseline2' with baseline2 float numbers (based on min and max of FEWS_CS)
    #         #features['FEWS_CS_baseline2']= np.baseline2.uniform(features['FEWS_CS'].min(), features['FEWS_CS'].max(), size=len(features))
            



    #         #make new SE_indicators variable based on whether there is SE data for the county or not
    #         if food_prices_county.empty:
    #             SE_indicators= []
    #         else:
    #             SE_indicators= ['maize_price']
            
    #         all_SE_indicators=NDMA_indicators+SE_indicators

    #         if with_NDMA==False: 
    #             all_SE_indicators=SE_indicators.copy()

    #         for ind in all_SE_indicators:
    #             features[ind].fillna((features[ind].mean()), inplace=True) # CHECK




    #         #################################################### feature engineering ####################################################
        

    #         if feature_engineering==True:
    #             # add rolling mean for 4 months and 12 months
    #             #     
    #             features['NDVI_roll_mean']=features['NDVI'].rolling(window=4).mean().shift(1)
    #             features['NDVI_range_roll_mean']=features['NDVI_range'].rolling(window=4).mean().shift(1)
    #             features['NDVI_crop_roll_mean']=features['NDVI_crop'].rolling(window=4).mean().shift(1)
    #             features['wd_roll_mean']=features['wd'].rolling(window=4).mean().shift(1)
    #             features['ds_roll_mean']=features['ds'].rolling(window=4).mean().shift(1)
    #             features['maize_price_roll_mean']=features['maize_price'].rolling(window=4).mean().shift(1)

    #             features['NDVI_roll_mean_12']=features['NDVI'].rolling(window=12).mean().shift(1)
    #             features['NDVI_range_roll_mean_12']=features['NDVI_range'].rolling(window=12).mean().shift(1)
    #             features['NDVI_crop_roll_mean_12']=features['NDVI_crop'].rolling(window=12).mean().shift(1)
    #             features['wd_roll_mean_12']=features['wd'].rolling(window=12).mean().shift(1)
    #             features['ds_roll_mean_12']=features['ds'].rolling(window=12).mean().shift(1)
                
    #             # fill nan values that came out of rolling
    #             features['NDVI_roll_mean'].fillna((features['NDVI_roll_mean'].mean()), inplace=True) # check: fillna with mean is incorrect! 
    #             features['NDVI_range_roll_mean'].fillna((features['NDVI_range_roll_mean'].mean()), inplace=True)
    #             features['NDVI_crop_roll_mean'].fillna((features['NDVI_crop_roll_mean'].mean()), inplace=True)
    #             features['wd_roll_mean'].fillna((features['wd_roll_mean'].mean()), inplace=True)
    #             features['ds_roll_mean'].fillna((features['ds_roll_mean'].mean()), inplace=True)

    #             features['NDVI_roll_mean_12'].fillna((features['NDVI_roll_mean_12'].mean()), inplace=True)
    #             features['NDVI_range_roll_mean_12'].fillna((features['NDVI_range_roll_mean_12'].mean()), inplace=True)
    #             features['NDVI_crop_roll_mean_12'].fillna((features['NDVI_crop_roll_mean_12'].mean()), inplace=True)
    #             features['wd_roll_mean_12'].fillna((features['wd_roll_mean_12'].mean()), inplace=True)
    #             features['ds_roll_mean_12'].fillna((features['ds_roll_mean_12'].mean()), inplace=True)



    #             ######################### rolling operations for maize prices ######################### CHECK: NO ROLLING FOR NDMA IND? 
    #             # maize prices 
    #             if food_prices_county.empty:
    #                 pass
    #             else:
    #                 features['maize_price_roll_mean_12']=features['maize_price'].rolling(window=12).mean().shift(1)
                    
    #                 # fill nan values that came out of rolling for maize price
    #                 features['maize_price_roll_mean'].fillna((features['maize_price_roll_mean'].mean()), inplace=True)# check: incorrect nan filling. Fill with obs 
    #                 features['maize_price_roll_mean_12'].fillna((features['maize_price_roll_mean_12'].mean()), inplace=True)

    #             # rolling operations for SST indicators (WVG, IOD, MEI, NINA34)
    #             if with_TC==True: 
    #                 features['WVG_roll_mean']=features['WVG'].rolling(window=4).mean().shift(1)
    #                 features['IOD_roll_mean']=features['IOD'].rolling(window=4).mean().shift(1)
    #                 features['MEI_roll_mean']=features['MEI'].rolling(window=4).mean().shift(1)
    #                 features['NINA34_roll_mean']=features['NINA34'].rolling(window=4).mean().shift(1)

    #                 features['WVG_roll_mean_12']=features['WVG'].rolling(window=12).mean().shift(1)
    #                 features['IOD_roll_mean_12']=features['IOD'].rolling(window=12).mean().shift(1)
    #                 features['MEI_roll_mean_12']=features['MEI'].rolling(window=12).mean().shift(1)
    #                 features['NINA34_roll_mean_12']=features['NINA34'].rolling(window=12).mean().shift(1)

    #                 # fill nan values that came out of rolling
    #                 features['WVG_roll_mean'].fillna((features['WVG_roll_mean'].mean()), inplace=True)
    #                 features['IOD_roll_mean'].fillna((features['IOD_roll_mean'].mean()), inplace=True)
    #                 features['MEI_roll_mean'].fillna((features['MEI_roll_mean'].mean()), inplace=True)
    #                 features['NINA34_roll_mean'].fillna((features['NINA34_roll_mean'].mean()), inplace=True)

    #                 features['WVG_roll_mean_12'].fillna((features['WVG_roll_mean_12'].mean()), inplace=True)
    #                 features['IOD_roll_mean_12'].fillna((features['IOD_roll_mean_12'].mean()), inplace=True)
    #                 features['MEI_roll_mean_12'].fillna((features['MEI_roll_mean_12'].mean()), inplace=True)
    #                 features['NINA34_roll_mean_12'].fillna((features['NINA34_roll_mean_12'].mean()), inplace=True)



    #         # drop nan values when whole column is nan (CROP column)
    #         features.dropna(axis=1, how='all', inplace=True)



    #         #################################################### Extract MAM and OND seasons ####################################################
    #         features['month']=features.index.month
    #         features['day']=features.index.day
    #         features['year']=features.index.year

    #         # add OND and MAM rainy season
    #         # OND --> months 10,11,12, flag in a new column with boolean values
    #         features['OND']=((features['month']==10) | (features['month']==11) | (features['month']==12))
    #         # MAM --> months 3,4,5, flag in a new column with boolean values
    #         features['MAM']=((features['month']==3) | (features['month']==4) | (features['month']==5))

    #         # make a seperate dataframe with day, month, year columns 
    #         features_date=features[['day','month','year']]

    #         # drop day, month, year columns from features
    #         features.drop(['day','month','year'], axis=1, inplace=True)


    #         if feature_engineering==True:
    #             #################################################### feature engineering for FEWS ####################################################
    #             #create a forward-filled column for FEWS_CS
    #             features['FEWS_CS_FF']=features['FEWS_CS'].fillna(method='ffill')
                
    #             # create a baseline ini 
    #             base_ini=features['FEWS_CS_FF']
    #             # FEWS_CS lags, and fill nan values with mean of the column
    #             features['FEWS_CS_lag1']=features['FEWS_CS_FF'].shift(1)
    #             features['FEWS_CS_lag1'].fillna((features['FEWS_CS_lag1'].mean()), inplace=True)

    #             features['FEWS_CS_lag2']=features['FEWS_CS_FF'].shift(2)
    #             features['FEWS_CS_lag2'].fillna((features['FEWS_CS_lag2'].mean()), inplace=True)

    #             features['FEWS_CS_lag3']=features['FEWS_CS_FF'].shift(3)
    #             features['FEWS_CS_lag3'].fillna((features['FEWS_CS_lag3'].mean()), inplace=True)
                
    #             features['FEWS_CS_lag4']=features['FEWS_CS_FF'].shift(4) # CHECK: FILL NANS WITH MEAN IS INCORRECT? --> FILL MEANS WITH JUST THE FEWS OBS COLUMN! 
    #             features['FEWS_CS_lag4'].fillna((features['FEWS_CS_lag4'].mean()), inplace=True)    

    #             # create new variables from rolling mean of existing features, where the preceding 4 months are used to calculate the mean. Do not include the current month 
    #             features['FEWS_CS_roll_mean']=features['FEWS_CS_FF'].rolling(window=4).mean().shift(1)
    #             features['FEWS_CS_roll_mean'].fillna((features['FEWS_CS_roll_mean'].mean()), inplace=True)
                

    #         # One-hot encode the data using pandas get_dummies
    #         features = pd.get_dummies(features) 
        





    #         # save target 
    #         labels=features['FEWS_CS'].dropna() # drop nan values in labels

    #         # Remove the target from the features
    #         features= features.drop('FEWS_CS', axis = 1)# axis 1 refers to the columns
    #         features=features.drop('FEWS_CS_FF', axis=1)


    #         ################################### saving features ###################################  
    #         feature_list = list(features.columns)

    #         if forecast==True: 
    #             # implement leads of 1,4 and 8, 12 months for all features, by shifting features X rows down based on lead 
                
    #             for lead in leads: 

    #                 features_l= features.copy()
                    
    #                 #cols_shift=list(features_l.columns)
    #                 #cols_shift.remove('FEWS_base')
                    
    #                 # Shift features by the lead (1,4,8,12) months. This trains and tests the model with features from the past, creating a lag to predict the future.
    #                 features_l=features_l.shift(lead)
                    
    #                 #remove the last X rows based on the lead
    #                 if lead!=0:
    #                     features_l=features_l.drop(features_l.index[:lead])

    #                 # after shifting, keep only the rows with index values that are in the labels dataframe
    #                 features_l=features_l.loc[labels.index]
                    

    #                 # explore the data and correlations with some fancy plots 

    #                 # pairplot with axis labels all around, and only the plots with highest correlations 
    #                 # sns.pairplot(features_l, kind="reg", diag_kind="kde", plot_kws={'line_kws':{'color':'red'}, 'scatter_kws': {'alpha': 0.1}})
            

    #                 #################################################### split data into training and testing sets ####################################################
    #                 train_features, test_features, train_labels, test_labels = train_test_split(features_l, labels, test_size = 0.25,shuffle=False) #25% of the data used for testing (38 time steps)   random_state = 42. if baseline2 state is not fixed, performance is different each time the code is run.
    #                 feature_list2=feature_list.copy()


    #                 #################################################### HYPER-PARAMETER TUNING ####################################################
    #                 counties_hp= [0,5,15,20] # selection of counties to perform hyper-parameter tuning on.
    #                 if i in counties_hp: 
    #                     if lead==4 or lead==8:
    #                         ############################################### Randomized Search CV ###############################################
    #                         # Number of trees in random forest
    #                         n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    #                         # Number of features to consider at every split
    #                         max_features = ['auto', 'sqrt']
    #                         # Maximum number of levels in tree
    #                         max_depth = [int(x) for x in np.linspace(5, 110, num = 11)]
    #                         max_depth.append(None)
    #                         # Minimum number of samples required to split a node
    #                         min_samples_split = [2, 5, 10]
    #                         # Minimum number of samples required at each leaf node
    #                         min_samples_leaf = [1, 2, 4]
    #                         # Method of selecting samples for training each tree
    #                         bootstrap = [True, False]
                            
    #                         # Create the random grid
    #                         random_grid = {'n_estimators': n_estimators, #add criterion? 
    #                                     'max_features': max_features, # max features to consider for each split 
    #                                     'max_depth': max_depth,
    #                                     'min_samples_split': min_samples_split,
    #                                     'min_samples_leaf': min_samples_leaf,
    #                                     'bootstrap': bootstrap}
                            



    #                         # Use the random grid to search for best hyperparameters
    #                         # First create the base model to tune
    #                         rf = RandomForestRegressor()
    #                         # Random search of parameters, using 10 fold cross validation, search across 100 different combinations, and use all available cores
    #                         rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 10, verbose=2, random_state=42, n_jobs = -1)
    #                         # Fit the random search model
    #                         model2=rf_random.fit(train_features, train_labels)
    #                         model2.best_params_

    #                         # save the best parameters in a dictionary
    #                         best_params= model2.best_params_
    #                         # append lead to the best params dictionary
    #                         best_params= {**best_params, **{'lead':lead}}

    #                         #append the best_params dictionary to the best_params_df dataframe using pd.concat
    #                         best_params_df=pd.concat([best_params_df, pd.DataFrame(best_params, index=[county])], axis=0)
    #                         print(best_params_df)

    #                         # save the best_params_df dataframe to a csv file to the ML_results folder
    #                         best_params_df.to_csv(RESULT_FOLDER+'\\best_params_df_%s.csv'%(aggregation))


