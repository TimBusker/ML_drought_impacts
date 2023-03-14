# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 15:33:24 2023

@author: tbr910
"""


#%% 
import os
import random as python_random
import numpy as np
import matplotlib.pyplot as plt
import datetime 
import pandas as pd
import seaborn as sns
os.chdir('C:\\Users\\tbr910\\Documents\\Forecast_action_analysis')

#with rasterio.open(os.path.join(DATA_FOLDER, 'data.tif'), 'r') as src
from ML_functions import *

#%% Machine learning model training
#import tensorflow as tf`
from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

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


#%% Define paths 
DATA_FOLDER = 'C:\\Users\\tbr910\\Documents\\Forecast_action_analysis\\'
ML_FOLDER = 'C:\\Users\\tbr910\\Documents\\ML\\'
SE_FOLDER= 'C:\\Users\\tbr910\\Documents\\ML\\SE_data\\'
RESULT_FOLDER = 'C:\\Users\\tbr910\\Documents\\ML\\ML_results'

os.chdir(ML_FOLDER)
#%% define master variables 
feature_engineering=True #without feature engineering the errors are much higher (almost 2x)   
forecast=True
fill_nans_target=True
leads=[1,2,3,4,8,12] # check: 12 lead creates quite some nan's in the training data 
random_split=False
#%% Define counties & read data
counties=['Garissa','Isiolo','Mandera','Marsabit','Samburu','Tana River','Turkana','Wajir','Baringo','Kajiado','Kilifi','Kitui','Laikipia','Makueni','Meru','Taita Taveta','Tharaka','West Pokot','Lamu','Nyeri','Narok']
df=pd.read_excel('input_data.xlsx', index_col=0)
df.drop('GP',axis=1, inplace=True)












# for county in counties: 
#     county=county

#     # select isiolo as county 
#     df_county=df[df['county']==county]

#     # linear imputation FEWS in between the not nan values 
#     df_county['FEWS_CS_INT']=df_county['FEWS_CS'].interpolate(method='time',limit_area='inside')

#     # plot interpolated FEWS_CS for one county

#     fig=plt.figure(figsize=(10,5))

#     # scatterplot of FEWS_CS and FEWS_CS_INT
#     df_scatter=df_county[['FEWS_CS','FEWS_CS_INT']]
#     plt.scatter(df_scatter.index,df_scatter['FEWS_CS_INT'], color='blue')
#     plt.scatter(df_scatter.index,df_scatter['FEWS_CS'], color='red')
#     plt.show() 




stats_df=pd.DataFrame(columns=('county', 'accuracy', 'accuracy_baseline', 'accuracy_random','accuracy_lr', 'var_score','var_score_baseline','var_score_random','var_score_lr', 'mae', 'mae_baseline', 'mae_random','mae_lr', 'mse', 'mse_baseline', 'mse_random', 'mse_lr', 'lead'))
features_df=pd.DataFrame(columns=('county', 'feature', 'feature_imp', 'lead'))             
features_df_full=pd.DataFrame()

#%% check of data is stationary? https://www.analyticsvidhya.com/blog/2021/06/random-forest-for-time-series-forecasting/
result = adfuller(df.FEWS_CS.dropna())
#print('p-value: %f' % result[1])# <0.05 is ok


############################################# get extra socio economic data #############################################

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


######################  population ###################### https://data.humdata.org/dataset/kenya-population-statistics-2019

# population= pd.read_excel('ken_admpop_2019.xlsx',sheet_name='ken_admpop_ADM1_2019',index_col=0, header=0)
# total_pop= population[['ADM1_NAME','T_TL']].set_index('ADM1_NAME')


# extract the feature names from the dataframe and convert to dataframe with nice names in new column 
feature_names=pd.DataFrame(df.columns, columns=['feature'])



#%% loop over counties  

for county in counties[0:6]: 



    print(county)
    # construct the dataset 
    
    county=county
    features= df[df['county']==county]#['FEWS_CS']#[~df['FEWS_CS'].isnull()]

    ############################################# Merge extra SE data #############################################
    

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



    ############################################# NAN values processing #############################################

    
    # linear imputation FEWS in between the not nan values 
    #
    
    if fill_nans_target==False:
        features=features.dropna(subset=['FEWS_CS'])

    else:
        #features['FEWS_CS']=features['FEWS_CS'].interpolate(method='time',limit_area='inside')
        features['FEWS_CS']=features['FEWS_CS'].fillna(method="ffill")
    

    # create dataframe column called 'FEWS_CS_random' with random float numbers (based on min and max of FEWS_CS)
    #features['FEWS_CS_random']= np.random.uniform(features['FEWS_CS'].min(), features['FEWS_CS'].max(), size=len(features))
    

    # fill na for indicators (debatable!)
    NDMA_indicators=['MPR', 'CP', 'MP', 'HDW', 'LDW', 'MUAC'] #'GP'

    #make new SE_indicators variable based on whether there is SE data for the county or not
    if food_prices_county.empty:
        SE_indicators= []
    else:
        SE_indicators= ['maize_price']

    all_SE_indicators=NDMA_indicators+SE_indicators


    for ind in all_SE_indicators:
        features[ind].fillna((features[ind].mean()), inplace=True)

    # explore the data and correlations with some fancy plots 

    # pairplot with axis labels all around, and only the plots with highest correlations 
    # sns.pairplot(features, kind="reg", diag_kind="kde", plot_kws={'line_kws':{'color':'red'}, 'scatter_kws': {'alpha': 0.1}})
    


    #################################################### feature engineering ####################################################
   

    if feature_engineering==True:
        # add rolling mean for 4 months and 12 months
        #     
        features['NDVI_roll_mean']=features['NDVI'].rolling(window=4).mean().shift(1)
        features['NDVI_range_roll_mean']=features['NDVI_range'].rolling(window=4).mean().shift(1)
        features['NDVI_crop_roll_mean']=features['NDVI_crop'].rolling(window=4).mean().shift(1)
        features['wd_roll_mean']=features['wd'].rolling(window=4).mean().shift(1)
        features['ds_roll_mean']=features['ds'].rolling(window=4).mean().shift(1)
        features['maize_price_roll_mean']=features['maize_price'].rolling(window=4).mean().shift(1)

        features['NDVI_roll_mean_12']=features['NDVI'].rolling(window=12).mean().shift(1)
        features['NDVI_range_roll_mean_12']=features['NDVI_range'].rolling(window=12).mean().shift(1)
        features['NDVI_crop_roll_mean_12']=features['NDVI_crop'].rolling(window=12).mean().shift(1)
        features['wd_roll_mean_12']=features['wd'].rolling(window=12).mean().shift(1)
        features['ds_roll_mean_12']=features['ds'].rolling(window=12).mean().shift(1)
        
        # fill nan values that came out of rolling
        features['NDVI_roll_mean'].fillna((features['NDVI_roll_mean'].mean()), inplace=True)
        features['NDVI_range_roll_mean'].fillna((features['NDVI_range_roll_mean'].mean()), inplace=True)
        features['NDVI_crop_roll_mean'].fillna((features['NDVI_crop_roll_mean'].mean()), inplace=True)
        features['wd_roll_mean'].fillna((features['wd_roll_mean'].mean()), inplace=True)
        features['ds_roll_mean'].fillna((features['ds_roll_mean'].mean()), inplace=True)

        features['NDVI_roll_mean_12'].fillna((features['NDVI_roll_mean_12'].mean()), inplace=True)
        features['NDVI_range_roll_mean_12'].fillna((features['NDVI_range_roll_mean_12'].mean()), inplace=True)
        features['NDVI_crop_roll_mean_12'].fillna((features['NDVI_crop_roll_mean_12'].mean()), inplace=True)
        features['wd_roll_mean_12'].fillna((features['wd_roll_mean_12'].mean()), inplace=True)
        features['ds_roll_mean_12'].fillna((features['ds_roll_mean_12'].mean()), inplace=True)



        ######################### rolling operations for maize prices #########################
        # maize prices 
        if food_prices_county.empty:
            pass
        else:
            features['maize_price_roll_mean_12']=features['maize_price'].rolling(window=12).mean().shift(1)
            
            # fill nan values that came out of rolling for maize price
            features['maize_price_roll_mean'].fillna((features['maize_price_roll_mean'].mean()), inplace=True)
            features['maize_price_roll_mean_12'].fillna((features['maize_price_roll_mean_12'].mean()), inplace=True)


    # drop nan values when whole column is nan (CROP column)
    features.dropna(axis=1, how='all', inplace=True)

    # drop nan values (values outside of the date range of the FEWS data)
    features=features[~features['FEWS_CS'].isna()] #keep only not nan values 


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
        features['FEWS_CS_lag1']=features['FEWS_CS'].shift(1)
        features['FEWS_CS_lag1'].fillna((features['FEWS_CS_lag1'].mean()), inplace=True)


        # FEWS_CS lags

        features['FEWS_CS_lag2']=features['FEWS_CS'].shift(2)
        features['FEWS_CS_lag3']=features['FEWS_CS'].shift(3)
        features['FEWS_CS_lag4']=features['FEWS_CS'].shift(4)
        features['FEWS_CS_lag5']=features['FEWS_CS'].shift(5)
        features['FEWS_CS_lag6']=features['FEWS_CS'].shift(6)

        # create new variables from rolling mean of existing features, where the preceding 4 months are used to calculate the mean. Do not include the current month 
        features['FEWS_CS_roll_mean']=features['FEWS_CS'].rolling(window=4).mean().shift(1)

        # fill nan values which are created by the rolling mean in the first 4 months with the mean of the whole column
        features['FEWS_CS_roll_mean'].fillna((features['FEWS_CS_roll_mean'].mean()), inplace=True)
        # same for the lags 
        
        features['FEWS_CS_lag2'].fillna((features['FEWS_CS_lag2'].mean()), inplace=True)
        features['FEWS_CS_lag3'].fillna((features['FEWS_CS_lag3'].mean()), inplace=True)
        features['FEWS_CS_lag4'].fillna((features['FEWS_CS_lag4'].mean()), inplace=True)
        features['FEWS_CS_lag5'].fillna((features['FEWS_CS_lag5'].mean()), inplace=True)    
        features['FEWS_CS_lag6'].fillna((features['FEWS_CS_lag6'].mean()), inplace=True)


    # One-hot encode the data using pandas get_dummies
    features = pd.get_dummies(features) 
  
    # save labels 
    labels=features['FEWS_CS']  
    features['FEWS_base']=features['FEWS_CS']

    # Remove the labels from the features
    features= features.drop('FEWS_CS', axis = 1)# axis 1 refers to the columns


    
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
    
    
    




    ################################### saving features ###################################  
    feature_list = list(features.columns)

    if forecast==True: 
        # implement leads of 1,4 and 8, 12 months for all features except for the FEWS_base
        # shift the features by 1,4 and 8 months, but not for the FEWS_base column
        # list of features except for FEWS_base
        
        for lead in leads: 
            features_l= features.copy()
            cols_shift=list(features_l.columns)
            cols_shift.remove('FEWS_base')
            # Shift all columns in cols_shift by 3 months 
            features_l[cols_shift]=features_l[cols_shift].shift(lead)
            
            # drop first X rows 
            features_l=features_l.drop(features_l.index[:lead])
            labels_l=labels.drop(labels.index[:lead])
            features_date_l=features_date.drop(features_date.index[:lead])
            


            #################################################### split data into training and testing sets ####################################################

            

                
            # Convert to numpy array
            features_np = np.array(features_l)
            train_features, test_features, train_labels, test_labels = train_test_split(features_l, labels_l, test_size = 0.25) #25% of the data used for testing (38 time steps)   random_state = 42. if random state is not fixed, performance is different each time the code is run.
            feature_list2=feature_list.copy()
            
            #convert dataframes to numpy arrays
            features_np = np.array(features_l)


            # time
            # make train_time and test_time variables by extract time from features_date_l based on length of train_features and test_features
            train_time=features_date_l.iloc[:len(train_features)]
            test_time=features_date_l.iloc[len(train_features):]

            




            features_date_np=np.array(features_date_l)
            train_time_np=np.array(train_time)
            test_time_np=np.array(test_time)
            train_features_np=np.array(train_features)
            test_features_np=np.array(test_features)

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

            ############################################### Baseline models ###############################################

            # baseline 1: use the last FEWS_base value as prediction for all time steps in the test set
            
            # get last FEWS_base value from train_features_np array and substitute all FEWS_base in test_features_np array with this value
            last_FEWS_base = train_features_np[-1, feature_list2.index('FEWS_base')]
            
            # make a 1D array with the last FEWS_base value and with same length as the first axis of the test_features_np array
            base1_preds= np.zeros(len(test_features_np)) + last_FEWS_base

            # Baseline errors, and display average baseline error
            base1_errors = abs(base1_preds - test_labels)


            # baseline 3 random FEWS_base value between min and max of FEWS_base in train_features_np array
            # get min and max of FEWS_base in train_features_np array
            min_FEWS_base = np.min(train_features_np[:, feature_list2.index('FEWS_base')])
            max_FEWS_base = np.max(train_features_np[:, feature_list2.index('FEWS_base')])
            # generate random FEWS_base value between min and max of FEWS_base in train_features_np array
            random_FEWS_base = np.random.uniform(min_FEWS_base, max_FEWS_base, len(test_features_np))
            
            # baseline 2: get the month from the train_features_np, calculate the grouped average over these months, use this dataframe to append a new column to the test_features_np array for each specific month as assigned in the test_features_np array
            if fill_nans_target==True: 
                # get the month from the train_features_np
                month = train_time_np[:, list(features_date_l.columns).index('month')]
                # calculate the grouped average over these months on the numpy array of fews_base
                grouped = pd.DataFrame(train_features_np[:, feature_list2.index('FEWS_base')], index=month).groupby(month).mean()

                # attach these averages per month in a 1D numpy array with same length as test_features_np array, based on the month assigned for that row in the test_features_np array (first axis)
                base2_preds= np.zeros(len(test_features_np)) + np.squeeze(grouped.loc[test_time_np[:, list(features_date_l.columns).index('month')]].values)
                base2_errors = abs(base2_preds - test_labels)

            else: 
                base2_preds=random_FEWS_base
                base2_errors = abs(base2_preds - test_labels)

            # add this random FEWS_base value to a new array with the length of test_features_np array
            base3_preds= np.zeros(len(test_features_np)) + random_FEWS_base
            base3_errors = abs(base3_preds - test_labels)

            # print (test_features_np[:,test_features_np.shape[1]-1]) # print the last column of the test_features_np array

            # drop FEWS_base variable from the arrays and df's (including feature list)
            train_features_np = np.delete(train_features_np, feature_list2.index('FEWS_base'), axis=1)
            test_features_np = np.delete(test_features_np, feature_list2.index('FEWS_base'), axis=1)
            feature_list2.remove('FEWS_base')
        



            ############################################### Linear Regression ###############################################
            regr = LinearRegression()
            regr.fit(train_features_np, train_labels)
            # Make predictions using the testing set
            lr_preds = regr.predict(test_features_np)
            lr_errors = abs(lr_preds - test_labels)

            ############################################### Random Forest ###############################################

            # Import the model we are using 
            rf = RandomForestRegressor(n_estimators = 1000,max_features='auto', n_jobs=-1, max_depth=5, min_samples_leaf=1, min_samples_split=2, random_state=40) # joris also uses max features  max_features='sqrt'
            
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
            # Train the model on training data
            model=rf.fit(train_features_np, train_labels)
            param_search = {'n_estimators' : [10, 100]}

            #gsearch = GridSearchCV(estimator=rf, cv=tss, param_grid=param_search)
            #gsearch.fit(train_features_np, train_labels)
            # Use the forest's predict method on the test data
            predictions = rf.predict(test_features_np)


            ############################################### Evaluate ###############################################

            # Calculate the absolute errors
            errors = abs(predictions - test_labels)

            # Calculate mean absolute percentage error (MAPE) 
            mape = 100 * (errors / test_labels)
            mape_baseline= 100 * (base2_errors / test_labels)
            mape_random= 100 * (base3_errors / test_labels)
            mape_lr= 100 * (lr_errors / test_labels)
            # Calculate and display accuracy
            accuracy = 100 - np.mean(mape)
            accuracy_baseline= 100 - np.mean(mape_baseline)
            accuracy_random= 100 - np.mean(mape_random)
            accuracy_lr= 100 - np.mean(mape_lr)

            # calculate other scores to evaluate continous variables 

            # Explained variance score: 1 is perfect prediction
            var_score = r2_score(test_labels, predictions)
            var_score_baseline= r2_score(test_labels, base2_preds)
            var_score_random= r2_score(test_labels, base3_preds)
            var_score_lr= r2_score(test_labels, lr_preds)

            # Mean absolute error
            mae = mean_absolute_error(test_labels, predictions)
            mae_baseline= mean_absolute_error(test_labels, base2_preds)
            mae_random= mean_absolute_error(test_labels, base3_preds)
            mae_lr= mean_absolute_error(test_labels, lr_preds)

            # Mean squared error
            mse = mean_squared_error(test_labels, predictions)
            mse_baseline= mean_squared_error(test_labels, base2_preds)
            mse_random= mean_squared_error(test_labels, base3_preds)
            mse_lr= mean_squared_error(test_labels, lr_preds)

            # Root mean squared error   
            rmse = np.sqrt(mean_squared_error(test_labels, predictions))
            rmse_baseline= np.sqrt(mean_squared_error(test_labels, base2_preds))
            rmse_random= np.sqrt(mean_squared_error(test_labels, base3_preds))
            rmse_lr= np.sqrt(mean_squared_error(test_labels, lr_preds))


            
            # Visualize trees
        
            #Pull out one tree from the forest
            # os.chdir('C:\\Users\\tbr910\\Documents\\ML\\trees')
            # tree = rf.estimators_[5]# Export the image to a dot file
            # export_graphviz(tree, out_file = 'tree_%s.dot'%(county), feature_names = feature_list2, rounded = True, precision = 1)# Use dot file to create a graph
            # (graph, ) = pydot.graph_from_dot_file('tree_%s.dot'%(county))# Write graph to a png file
            # graph.write_png('tree_%s.png'%(county))
            

            
            # Get feature importances -->  computed as the mean and standard deviation of accumulation of the impurity decrease within each tree.
            importances = list(rf.feature_importances_)# List of tuples with variable and importance
            std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
            feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list2, importances)]# Sort the feature importances by most important first
            
            # append featuer importances to df
            test=pd.DataFrame(importances, index=feature_list2, columns=['importance_%s'%(county)])
            features_df_full=pd.concat([features_df_full, test], axis=1)
            
            # sort feature importances and print most important feature
            feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
            main_feature= feature_importances[0][0]
            feature_imp= feature_importances[0][1]

            
            ############################################### Plotting section ###############################################
            
            # reconstruct dataframe with true values 
            months = features_date_np[:, 1]
            days = features_date_np[:, 0]
            years = features_date_np[:, 2]
            
            # List and then convert to datetime object
            dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
            dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]
            
            # Dataframe with true values and dates
            true_data = pd.DataFrame(data = {'date': dates, 'FEWS_CS': labels_l})# Dates of predictions


            # reconstruct dataframe with test values 
            months = test_time_np[:, 1]
            days = test_time_np[:, 0]
            years = test_time_np[:, 2]# Column of dates
            test_dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]# Convert to datetime objects
            test_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in test_dates]
            
            # reconstruct dataframe with predictions
            predictions_data = pd.DataFrame(data = {'date': test_dates, 'prediction': predictions, 'base1':base1_preds, 'base2': base2_preds, 'base3': base3_preds, 'lr':lr_preds}) # Dataframe with predictions and dates
            

            ##### plots #####
            # Make dir
            if not os.path.exists(RESULT_FOLDER+'\\plots'):
                os.makedirs(RESULT_FOLDER+'\\plots')

            os.chdir(RESULT_FOLDER+'\\plots')
            #  Variable importances
            forest_importances = pd.Series(importances, index=feature_list2)

            fig, ax = plt.subplots()
            forest_importances.plot.bar(yerr=std, ax=ax)

            
            ax.set_title("Feature importances using MDI for %s, L%s"%(county,lead))
            ax.set_ylabel("Mean decrease in impurity")
            ax.set_ylim(0,1)    
            fig.tight_layout()
            plt.savefig('Variable_Importances_%s_L%s.png'%(county,lead), dpi=300, bbox_inches='tight')
            plt.close()
            # save plot
            #Make dir

            fig, ax = plt.subplots(figsize=(10, 5))
            ax2=ax.twinx()
            # Plot the actual values
            ax.plot(true_data['date'], true_data['FEWS_CS'], 'b-', label = 'Observed FEWS class')# Plot the predicted values
            
            # plot features 
            
            # plot rf predictions
            ax.plot(predictions_data['date'], predictions_data['prediction'], 'ro', label = 'Random Forest prediction')
            # plot base predictions
            #plt.plot(predictions_data['date'], predictions_data['base1'], 'go', label = 'base1 prediction (current = future)')
            #ax.plot(predictions_data['date'], predictions_data['base2'], 'yo', label = 'baseline prediction (past seasonality= future)')
            #plt.plot(predictions_data['date'], predictions_data['base3'], 'ko', label = 'base3 prediction (future is random)')
            #plot lr predictions
            #ax.plot(predictions_data['date'], predictions_data['lr'], 'mo', label = 'Linear regression prediction')
            plt.xticks(rotation = '60'); 
            plt.xlabel('Date'); plt.ylabel('FEWS IPC class'); plt.title('FEWS observations vs RF predictions for %s. Accuracy=%s'%(county,round(accuracy, 2)));
            #plt.savefig('TS_%s_S%s_L%s.png'%(county,split,lead), dpi=300, bbox_inches='tight')
            
            
            # plot features
            true_data['SPI6'] = features_np[:, feature_list2.index('SPI6')]
            true_data['SPI12'] = features_np[:, feature_list2.index('SPI12')]
            ax2.plot(true_data['date'], true_data['SPI6'], 'o-', color='orange', label = 'SPI6')# Plot the predicted values
            ax2.plot(true_data['date'], true_data['SPI12'], 'o-', color='black', label = 'SPI12')# Plot the predicted values
            plt.legend()# Graph labels
            plt.show() 
            plt.close()

            # plot ROC curve 
            # Calculate the false positive rates and true positive rates
            # fpr, tpr, _ = roc_curve(test_labels, predictions)# Plot of a ROC curve for a specific class
            # plt.figure()
            # plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
            # plt.plot([0, 1], [0, 1], 'k--')# Random guesses should fall in the middle
            # plt.xlim([0.0, 1.0])# Limit the graph
            # plt.ylim([0.0, 1.05])# Label the graph
            # plt.xlabel('False Positive Rate')
            # plt.ylabel('True Positive Rate')
            # plt.title('ROC curve for %s'%(county))
            # plt.legend(loc="lower right")
            # plt.savefig('ROC_%s.png'%(county))
            # plt.show()

            # make a evaluation dataframe (with baseline being the prediction based on observed clim)
            var_list= [county,round(accuracy, 2), round(accuracy_baseline, 2), round(accuracy_random, 2), round(accuracy_lr,2), round(var_score, 2), round(var_score_baseline, 2),round(var_score_random, 2), round(var_score_lr,2), round(mse,2), round(mse_baseline,2), round(mse_random,2), round(mse_lr,2), round(mae,2), round(mae_baseline,2), round(mae_random,2), round(mae_lr,2),lead]         
            stats_df.loc[len(stats_df), :] = var_list

            # make a features df
            var_list=[county, main_feature, feature_imp, lead]
            features_df.loc[len(features_df), :] = var_list

#stats_df=pd.read_csv(RESULT_FOLDER+'\\stats_df2.csv')
# feature_df=pd.read_csv(RESULT_FOLDER+'\\features_df2.csv')



################################################# plots ###############################################

############### feature plots ####################
# keep only columns in feature_df_full from which the column name contains a zero (0)
# get column names of features_df_full
col_names=features_df_full.columns

# keep only columns in features_df_full that contain a zero (0)
features_df_full2=features_df_full[col_names]
# delete all rows with full nan values 
features_df_full2=features_df_full2.dropna(axis=0, how='any')


# plot features_df_full as bar plot with stdev as error bars 
means=features_df_full2.mean(axis=1)
# keep only top 10 features (index) based on mean values 
means=means.sort_values(ascending=False).head(10)
# get the stdev of the top 10 features in means 
stdev=features_df_full2.loc[means.index].std(axis=1)


# bar plot of every feature (index of features_df_fu with means and stdev as error bars
fig=plt.figure(figsize=(10, 5)) 
plt.bar(means.index, means, yerr=stdev, align='center', alpha=0.5, ecolor='black', capsize=10)
plt.xticks(means.index, rotation='vertical')
plt.ylabel('Feature importance')

plt.title('Feature importance for all counties')
plt.tight_layout()
# save plot in plots folder
plt.savefig(RESULT_FOLDER+'\\plots\\Feature_importances_all_counties.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()



############################################## evaluation plots ####################################
# convert all negative values in the df to zero 
column_names_eval = ['accuracy', 'accuracy_baseline', 'accuracy_random', 'accuracy_lr', 'var_score', 'var_score_baseline', 'var_score_random', 'var_score_lr', 'mse', 'mse_baseline', 'mse_random', 'mse_lr', 'mae', 'mae_baseline', 'mae_random', 'mae_lr']

for column in column_names_eval:
    stats_df.loc[stats_df[column]<0, column]=0

var_vars=['var_score', 'var_score_baseline', 'var_score_random', 'var_score_lr']
mse_vars=['mse', 'mse_baseline', 'mse_random', 'mse_lr']
mae_vars=['mae', 'mae_baseline', 'mae_random', 'mae_lr']
accuracy_vars=['accuracy', 'accuracy_baseline', 'accuracy_random', 'accuracy_lr']

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
    plt.savefig(RESULT_FOLDER+'\\plots\\Variance_explained_lead_%s.png'%(lead), dpi=300, bbox_inches='tight')
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
    plt.savefig(RESULT_FOLDER+'\\plots\\MSE_lead_%s.png'%(lead), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()













#%% save results
#make new folder for results
os.chdir(RESULT_FOLDER)
if not os.path.exists(RESULT_FOLDER):
    os.makedirs(RESULT_FOLDER)
#save results in new folder     
stats_df.to_csv('stats_df2.csv')
features_df.to_csv('features_df2.csv')




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