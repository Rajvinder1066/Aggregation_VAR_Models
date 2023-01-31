# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 23:17:13 2021

@author: gaurav.tibude
"""

import aggregation_helper_functions as ahf
import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats as stats
from matplotlib import pyplot as plt
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.diagnostic import acorr_lm, acorr_breusch_godfrey
import warnings
warnings.filterwarnings("ignore")

## Model 1 Code

# Start of main code

def Premodel_1(last_iteration=16,recomputation=False):

    # This reads csv file with input data into model_1_input_raw_data
    model_1_input_raw_data = pd.read_csv(
        'Input/PreModel 1/DesiredModel1Data_12Aug.csv',
        parse_dates=True,
        index_col=0)
    
    covid_event = pd.DataFrame(model_1_input_raw_data[['Real GDP', 'GDP Deflator']])
    covid_event = covid_event.rename(columns={'Real GDP':'EVENT',
                                              'Real PCE':'FIN_CRISIS08'})
    covid_event['EVENT'] = 0.00000000
    #covid_event.loc['2020-01-01']['EVENT'] = -0.109
    # covid_event.loc['2020-04-01']['EVENT'] = -0.356
    # covid_event.loc['2020-07-01']['EVENT'] = 0.358
    # covid_event.loc['2020-10-01']['EVENT'] = -0.043
    
    #covid_event.loc['2020-01-01']['EVENT'] = -0.075
    covid_event.loc['2020-04-01']['EVENT'] = -0.328
    covid_event.loc['2020-07-01']['EVENT'] = 0.107
    covid_event.loc['2020-10-01']['EVENT'] = 0.022
    #covid_event.loc['2021-01-01']['EVENT'] = 1 - 0.66386991
    
    covid_event['FIN_CRISIS08'] = 0.00000000
    covid_event.loc['2008-10-01']['FIN_CRISIS08'] = 1
    covid_event.loc['2009-01-01']['FIN_CRISIS08'] = 0.37
    covid_event.loc['2009-04-01']['FIN_CRISIS08'] = 0.1369
    
    ## Variation 1 all log first diff
    log_diff_var_list = ['Real GDP','GDP Deflator','Shadow Rate','Real Exchange Rate']
    diff_var_list = []
    log_list = []
    var_dict ={'dlRealGDP': 'Real GDP', 'dlGDPDeflator': 'GDP Deflator',
                'ShadowRate': 'Shadow Rate',
                'RealExchangeRate': 'Real Exchange Rate'}
    
    short_names_dict = {'dlReal GDP': 'dlRealGDP',
                        'dlGDP Deflator':'dlGDPDeflator',
                        'Shadow Rate':'ShadowRate',
                        'Real Exchange Rate':'RealExchangeRate'}
    
    log_trans = [True,True,False,False]
    first_diff= [True,True,False,False]
    
 
    initial_training_end_date = '2022-04-01'
    
    # This creates model_1_data after tranformation
    model_1_data = ahf.preprocessing(model_1_input_raw_data,
                                 log_diff_var_list,
                                 diff_var_list,
                                 log_list,
                                 short_names_dict,
                                 initial_training_end_date)  
    lag_order = 8
    
    initial_training_start_date = '1994-04-01'
    #initial_training_end_date = '2021-01-01'
    initial_training_end_date = '2022-04-01'
    #forecast_start_date = '2021-04-01'
    forecast_start_date = '2022-07-01'
    forecast_end_date = '2025-10-01'
        
    results,forecasts_trans_df, forecasts_new  = ahf.generate_out_of_sample_forecasts(
                                initial_training_start_date,
                                initial_training_end_date,
                                forecast_end_date,
                                model_1_data,
                                model_1_input_raw_data,
                                covid_event,
                                log_trans,
                                first_diff,
                                _lag_order=lag_order)
    
    n_steps = len(forecasts_trans_df)
    n_variables = len(forecasts_trans_df.columns)
    _irf = results.irf(n_steps)
    with pd.ExcelWriter('Output/PreModel 1/Model 1 Run 1 Lag 2 Covid Irfs.xlsx') as writer:  # doctest: +SKIP
        for i in range(n_steps):
            pd.DataFrame(_irf.orth_irfs[i]).to_excel(
                writer, startrow=(n_variables+2)*i, header=False)
    results.params.drop('EVENT').to_csv('Output/PreModel 1/Model 1 Lag 1 Run 1 Covid Coeffs.csv')
    pd.DataFrame(model_1_data.loc[initial_training_end_date]).transpose(
        ).to_csv('Output/PreModel 1/Model 1 Run 1 Covid Event Data bef Forecast.csv')
    pd.DataFrame(model_1_input_raw_data.loc[initial_training_end_date]).transpose(
        ).to_csv('Output/PreModel 1/Model 1 Run 1 Covid Event Data bef Forecast Orig Scale.csv')
    forecasts_trans_df.to_csv(
        'Output/PreModel 1/Model 1 Covid Event Initial Forecasts Transformed Form Lag 2 Run 1.csv')
    forecast_columns = [col for col in forecasts_new.columns\
                    if 'forecast' in col]
    forecasts_new.to_csv('Output/PreModel 1/Model 1 forecast new df.csv')
    
    ## Code for adjusting forecasts
    initial_forecasts = forecasts_trans_df.copy()
    initial_forecasts = np.array(initial_forecasts)
    
    irfs = results.irf(n_steps)
    irfs = irfs.orth_irfs
    
    coeffs = results.params.drop(['EVENT','FIN_CRISIS08'])
    #coeffs = results.params.drop()
    
    coeffs = np.array(coeffs)
    
    #lag_order = 2
    
    # upstream_forecasts = pd.read_csv(
    #     'Input/Model 1/OE Real GDP Log first diff.csv',
    #     index_col=0, parse_dates=True)
    upstream_forecasts = pd.read_csv(
        'Input/PreModel 1/IO Model Real GDP Log first diff.csv',
        index_col=0, parse_dates=True)
    upstream_forecasts = np.array(upstream_forecasts)  
    
    col_loc_adj_col = [0]
    
    last_loc_bef_forecast = model_1_data.index.get_loc(initial_training_end_date)
    if lag_order == 2:
        bef_forecast_values = pd.DataFrame(model_1_data.loc[initial_training_end_date])\
        .transpose()
    else:
        bef_forecast_values = pd.DataFrame(
            model_1_data.iloc[last_loc_bef_forecast-(lag_order-2):\
                              last_loc_bef_forecast+1])
        
    bef_forecast_values = np.array(bef_forecast_values)  
    
    if not recomputation:
        coeffs=None
    
    new_forecasts = ahf.adjust_forecasts(
                        initial_forecasts,
                        irfs,
                        coeffs,
                        lag_order,
                        upstream_forecasts,
                        col_loc_adj_col,
                        bef_forecast_values,
                        last_iteration,
                        output_location='Output/PreModel 1/shocks')
    new_forecasts_df = pd.DataFrame(new_forecasts)
    
    
    non_forecast_columns = [col for col in forecasts_new.columns\
                    if 'forecast' not in col]
    
    new_forecasts_df.columns=non_forecast_columns
    
    #new_adj_forecasts.index = forecasts_new.index
    
    bef_for_orig_scale = pd.DataFrame(model_1_input_raw_data.loc[
        initial_training_end_date]).transpose(
        )
    
    new_adj_forecasts_orig_scale = ahf.invert_transformation(
                          bef_for_orig_scale,
                          new_forecasts_df,
                          log_trans,
                          first_diff)
    
    forecast_columns = [col for col in new_adj_forecasts_orig_scale.columns\
                        if 'forecast' in col]
    
    new_adj_forecasts_orig_scale = new_adj_forecasts_orig_scale[forecast_columns]
    
    new_adj_forecasts_orig_scale.index = forecasts_new.index
    
    # forecast_columns = [col for col in forecasts_new.columns\
    #                     if 'forecast' in col]
    
    # forecasts_new = forecasts_new[forecast_columns]
    
    forecasts_new_with_adj = forecasts_new.join(new_adj_forecasts_orig_scale, rsuffix='_adj')
    
    for column in non_forecast_columns:
        temp_dict = dict()
        temp_dict['OE_forecast'] = forecasts_new_with_adj[column]
        temp_dict['initial_forecast'] = forecasts_new_with_adj[column+'_forecast']
        temp_dict['new_adj_forecast'] = forecasts_new_with_adj[column+'_forecast_adj']
        ax = pd.DataFrame(temp_dict,index=forecasts_new.index).plot(title=column)
        #ax.set_ylim(0, np.array(pd.DataFrame(temp_dict,index=forecasts_new.index)).max()*1.5)
        fig = ax.get_figure()
        fig.savefig('Output/Variations/Variant 1/PreModel 1 '+column+'.png')
        pd.DataFrame(temp_dict,index=forecasts_new.index).to_csv(
            'Output/PreModel 1/Adj Forecasts/'+column+\
                '_adj_forecasts_model_1_covid.csv')
        pd.DataFrame(temp_dict,index=forecasts_new.index).to_csv(
            'Output/Variations/Variant 1/Adj Forecasts/PreModel 1/'+column+\
                '_adj_forecasts_model_1_covid.csv')
    

    
    for column in results.resid.columns:
        fig = sm.qqplot(results.resid[column],stats.norm, fit=True, line='45')
        plt.show()
        print("Results for", column)
        print(jarque_bera(results.resid[column]))
        print("----------------------")
        
    print("--- LM Test for autocorrelation")
    for column in results.resid.columns:
        if acorr_lm(results.resid[column],nlags=lag_order+1)[1] > .05:
            print('Null Fail to reject for',column)
        else:
            print('Null rejected for',column)
 
    
