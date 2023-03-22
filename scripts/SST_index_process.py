

import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr


wd='C:\\Users\\tbr910\\Documents\\ML\\SE_data'
os.chdir(wd)


# open meiv2 txt file with pandas. Columns represent months in the year (1-12) and rows represent years (1979-2023). seperator is a space


# import pandas and numpy
import pandas as pd
import numpy as np
# define a function to read the meiv2.txt file
def read_NOAA(txt_file,variable):
    # read the txt file into a pandas dataframe
    # the file is separated by spaces, so use sep='\s+'
    # there is no header, so use header=None
    # skip the first row, which is a header, so use skiprows=1
    meiv2=pd.read_csv(txt_file, sep='\s+', header=None, skiprows=1)

    # make a datetime index by using the year in column 0, and the months in columns 1-12
    # make a list of the months
    months=[1,2,3,4,5,6,7,8,9,10,11,12]
    # make a list of the years
    years=meiv2[0].values

    # make a list of the dates
    dates=[str(year)+'-'+str(month) for year in years for month in months]
    # convert the list of dates to a datetime index
    dates=pd.to_datetime(dates, format='%Y-%m')

    # get the values from the dataframe 
    values=meiv2.iloc[:,1:].values.flatten()

    # make a dataframe with the datetime index and the values
    meiv2=pd.DataFrame(values, index=dates, columns=[variable])
    return meiv2


# Multivariate ENSO Index Version 2 (MEI.v2)
# https://www.psl.noaa.gov/enso/mei
# Row values are 2 month seasons (YEAR DJ JF FM MA AM MJ JJ JA AS SO ON ND)

df=read_NOAA('meiv2.txt','MEI')
# add the dataframe to a new tab in SST_index.xlsx excel file 
with pd.ExcelWriter('SST_index.xlsx', mode='a') as writer:
    df.to_excel(writer, sheet_name='MEI')

#  Nino Anom 3.4 Index  using ersstv5 from CPC
# https://psl.noaa.gov/data/climateindices/list/for info
df=read_NOAA('nina34.txt','NINA34')
# add the dataframe to a new tab in SST_index.xlsx excel file
with pd.ExcelWriter('SST_index.xlsx', mode='a') as writer:
    df.to_excel(writer, sheet_name='NINA34')

# read the SST_index.xlsx excel file, tab WVG
df=pd.read_excel('SST_index.xlsx', sheet_name='WVG',skiprows=10)
df.drop('Unnamed: 2', axis=1, inplace=True)

# make a list of monthly datetimes using the year column. month column doesnt exist
dates=[str(year)+'-'+str(month) for year in df['year'] for month in range(1,13)]

# convert the list of dates to a datetime index
dates=pd.to_datetime(dates, format='%Y-%m')

# repeat every value in the WVG column 12 times to match the length of the datetime index
values=np.repeat(df['MAM WVG Values'].values,12)
df=pd.DataFrame(values, index=dates, columns=['WVG'])

# set all months except march-april-may to NaN
df.loc[(df.index.month!=3)&(df.index.month!=4) & (df.index.month!=5),'WVG']=np.nan

# add the dataframe to a new tab in SST_index.xlsx excel file
with pd.ExcelWriter('SST_index.xlsx', mode='a') as writer:
    df.to_excel(writer, sheet_name='WVG_processed')
    


