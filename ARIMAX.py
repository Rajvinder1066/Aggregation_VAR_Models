# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 10:54:57 2022

@author: Rajvinder.Kaur
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from datetime import datetime
from pmdarima.arima import auto_arima
from statsmodels.tsa.arima_model import ARIMA
import pmdarima as pm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})



df = pd.read_csv(
        'Input/ARIMAX/10yTreasury.csv',
        parse_dates=True,
        index_col=0)
Shadow_Rate_forecasts = pd.read_csv(
        'Input/ARIMAX/Shadow_Rate_forecasts.csv')


xmodel = pm.auto_arima(df[['10YTreasuryNotesRate']], exogenous=df[['Shadow_Rate']],
                           start_p=1, start_q=1,
                           test='adf',
                           max_p=3, max_q=3,
                           start_P=0, seasonal=False,
                           d=None, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True,stationary=True)

xmodel.summary()
n_periods = 14
fitted, confint = xmodel.predict(n_periods = n_periods, exogenous = Shadow_Rate_forecasts,
                      return_conf_int=True)
fitted_series = pd.DataFrame(fitted)
index_of_fc = pd.read_csv(
        'Input/ARIMAX/Shadow_Rate_forecasts_dates.csv')['Quarter']
index_of_fc = pd.DataFrame(index_of_fc)


new_forecast = index_of_fc.reset_index(drop=True).join(fitted_series)
new_forecast = new_forecast.rename(columns = {0:'10YTreasuryNotesRate'})
new_forecast
#new_forecast.to_csv('Input/ARIMAX/10YTreasuryNotesRate.csv')
new_forecast.to_csv('Input/ARIMAX/10YTreasuryNotesRate_forecast.csv',index=False) 


df1 = pd.read_csv(
        'Input/ARIMAX/3MONTERESTRATE.csv',
        parse_dates=True,
        index_col=0)
Shadow_Rate_forecasts = pd.read_csv(
        'Input/ARIMAX/Shadow_Rate_forecasts.csv')


xmodel = pm.auto_arima(df1[['3MInterestRates']], exogenous=df1[['Shadow_Rate']],
                           start_p=1, start_q=1,
                           test='adf',
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=False,
                           d=None, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)

xmodel.summary()
n_periods = 16
fitted, confint = xmodel.predict(n_periods = n_periods, exogenous = Shadow_Rate_forecasts,
                      return_conf_int=True)
fitted_series = pd.DataFrame(fitted)
index_of_fc = pd.read_csv(
        'Input/ARIMAX/Shadow_Rate_forecasts_dates.csv')['Quarter']
index_of_fc = pd.DataFrame(index_of_fc)


new_forecast = index_of_fc.reset_index(drop=True).join(fitted_series)
new_forecast = new_forecast.rename(columns = {0:'3MInterestRates'})
new_forecast
#new_forecast.to_csv('Input/ARIMAX/3MInterestRates.csv')
new_forecast.to_csv('Input/ARIMAX/3MInterestRates_forecast.csv',index=False)    


df2 = pd.read_csv(
        'Input/ARIMAX/30YFixedMortgageRate.csv',
        parse_dates=True,
        index_col=0)
Shadow_Rate_forecasts = pd.read_csv(
        'Input/ARIMAX/Shadow_Rate_forecasts.csv')
df2
Shadow_Rate_forecasts
xmodel = pm.auto_arima(df2[['30YFixedMortgageRate']], exogenous=df2[['Shadow_Rate']],
                           start_p=1, start_q=1,
                           test='adf',
                           max_p=2, max_q=2,
                           start_P=0, seasonal=False,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)

xmodel.summary()
n_periods = 14
fitted, confint = xmodel.predict(n_periods = n_periods, exogenous = Shadow_Rate_forecasts,
                      return_conf_int=True)
fitted_series = pd.DataFrame(fitted)
index_of_fc = pd.read_csv(
        'Input/ARIMAX/Shadow_Rate_forecasts_dates.csv')['Quarter']
index_of_fc = pd.DataFrame(index_of_fc)


new_forecast = index_of_fc.reset_index(drop=True).join(fitted_series)
new_forecast = new_forecast.rename(columns = {0:'30YFixedMortgageRate'})
new_forecast
new_forecast.to_csv('Input/ARIMAX/30YFixedMortgageRate_forecast.csv',index=False) 



