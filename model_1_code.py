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
from statsmodels.tsa.vector_ar.vecm import VECM, select_order
warnings.filterwarnings("ignore")

## Model 1 Code

# Start of main code


    
def model_1_with_HoursWorked(last_iteration=14,recomputation=False):

    # This reads csv file with input data into model_1_input_raw_data
    model_1_input_raw_data = pd.read_csv(
        'Input/Model 1/with Hours Worked/DesiredModel1Data_6Oct with hours.csv',
        parse_dates=True,
        index_col=0)
    
    # model_1_input_raw_data = pd.read_csv(
    #     'Input/Model 1/with CPI/DesiredModel1Data_12Aug qtrly 10YTNR.csv',
    #     parse_dates=True,
    #     index_col=0)
    
    # model_1_input_raw_data = pd.read_csv('DesiredModel1Data_12Aug.csv',
    #                                       parse_dates=True,
    #                                       index_col=0)
    
    covid_event = pd.DataFrame(model_1_input_raw_data[['Real GDP', 'Real PCE']])
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
    
    # decay_factor = 0.37
    # crisis_period_start = '2008-10-01'
    # crisis_period_end = '2009-04-01'
    # covid_period_start_index = covid_event.index.get_loc(covid_period_start)
    # covid_period_end_index = covid_event.index.get_loc(covide_period_end)
    
    # for i in range(covid_period_end_index-covid_period_start_index+1):
    #       covid_event.iloc[covid_period_start_index+i]['EVENT'] = decay_factor**(i)
    
    
    # decay_factor = 0.37
    # covid_period_start = '2020-04-01'
    # covide_period_end = '2021-01-01'
    # covid_period_start_index = covid_event.index.get_loc(covid_period_start)
    # covid_period_end_index = covid_event.index.get_loc(covide_period_end)
    
    # for i in range(covid_period_end_index-covid_period_start_index+1):
    #       covid_event.iloc[covid_period_start_index+i]['EVENT'] = decay_factor**(i)
    
    # Following list is a list of variables which are transformed during
    # pre-processing step in difference log format
    
    # log_diff_var_list = ['Real GDP', 'Real PCE', 'Real Investment',
    #             'Real Disposable Income', 'UnemploymentRate',
    #             'Prices','10YTreasuryNotesRate']
    
    # # log_diff_var_list = ['Real GDP', 'Real PCE', 'Real Investment',
    # #             'Real Disposable Income',
    # #             'Consumer price index']
    
    # # log_diff_var_list = ['Real GDP', 'Real PCE', 'Real Investment',
    # #             'Real Disposable Income',
    # #             'Prices','10YTreasuryNotesRate']
    
    # diff_var_list = []
    # #diff_var_list = ['UnemploymentRate']
    # log_list = []
    # #log_list = ['UnemploymentRate']
    
    # var_dict ={'dlRealGDP': 'Real GDP', 'dlConsumptionExpenditure': 'Real PCE',
    #            'dlInvestment': 'Real Investment',
    #            'dlRealDisposableIncome': 'Real Disposable Income',
    #            'dlUnemploymentRate': 'UnemploymentRate',
    #            'dlPrices': 'Prices',
    #            'dl10YTreasuryNotesRate':'10YTreasuryNotesRate'}
    
    # short_names_dict = {'dlReal GDP': 'dlRealGDP',
    #                     'dlReal PCE':'dlConsumptionExpenditure',
    #                     'dlReal Investment':'dlInvestment',
    #                     'dlReal Disposable Income':'dlRealDisposableIncome',
    #                     'dlUnemploymentRate': 'dlUnemploymentRate',
    #                     'dlPrices':'dlPrices',
    #                     'dl10YTreasuryNotesRate':'dl10YTreasuryNotesRate'}
    
    ## Variation 1 all log first diff
    log_diff_var_list = ['Real GDP', 'Real PCE', 'Real Investment',
                'Real Disposable Income','HoursWorked',
                'Consumer price index','10YTreasuryNotesRate']
    diff_var_list = []
    log_list = []
    var_dict ={'dlRealGDP': 'Real GDP', 'dlConsumptionExpenditure': 'Real PCE',
                'dlInvestment': 'Real Investment',
                'dlRealDisposableIncome': 'Real Disposable Income',
                'dlHoursWorked': 'HoursWorked',
                'dlPrices': 'Consumer price index',
                'dl10YTreasuryNotesRate':'10YTreasuryNotesRate'}
    
    short_names_dict = {'dlReal GDP': 'dlRealGDP',
                        'dlReal PCE':'dlConsumptionExpenditure',
                        'dlReal Investment':'dlInvestment',
                        'dlReal Disposable Income':'dlRealDisposableIncome',
                        'dlHoursWorked': 'dlHoursWorked',
                        'dlConsumer price index':'dlPrices',
                        'dl10YTreasuryNotesRate':'dl10YTreasuryNotesRate'}
    
    log_trans = [True,True,True,True,True,True,True]
    first_diff= [True,True,True,True,True,True,True]
    
    # ## Variation 2 all log first diff except 10YTreasuryNotesRate
    # log_diff_var_list = ['Real GDP', 'Real PCE', 'Real Investment',
    #             'Real Disposable Income','UnemploymentRate',
    #             'Consumer price index']
    # diff_var_list = []
    # log_list = []
    # var_dict ={'dlRealGDP': 'Real GDP', 'dlConsumptionExpenditure': 'Real PCE',
    #             'dlInvestment': 'Real Investment',
    #             'dlRealDisposableIncome': 'Real Disposable Income',
    #             'dlUnemploymentRate': 'UnemploymentRate',
    #             'dlPrices': 'Consumer price index',
    #             '10YTreasuryNotesRate':'10YTreasuryNotesRate'}
    
    # short_names_dict = {'dlReal GDP': 'dlRealGDP',
    #                     'dlReal PCE':'dlConsumptionExpenditure',
    #                     'dlReal Investment':'dlInvestment',
    #                     'dlReal Disposable Income':'dlRealDisposableIncome',
    #                     'dlUnemploymentRate': 'dlUnemploymentRate',
    #                     'dlConsumer price index':'dlPrices',
    #                     '10YTreasuryNotesRate':'10YTreasuryNotesRate'}
    # log_trans = [True,True,True,True,True,True,False]
    # first_diff= [True,True,True,True,True,True,False]
    
    #initial_training_end_date = '2021-01-01'
    initial_training_end_date = '2022-04-01'
    
    # This creates model_1_data after tranformation
    model_1_data = ahf.preprocessing(model_1_input_raw_data,
                                 log_diff_var_list,
                                 diff_var_list,
                                 log_list,
                                 short_names_dict,
                                 initial_training_end_date)
    

    model_1_data.to_csv('Input/Model 1/with Hours Worked/for Model 1.csv')
    # # Block to generate rolling forecasts for RMSE analysis
    # initial_training_start_date = '1980-04-01'
    # initial_training_end_date = '2010-10-01'
    
    # forecast_end_date = '2019-10-01'
    
    # forecast_horizons_list = ['1q', '2q', '4q', '8q', '12q', '24q']
    
    lo = select_order( model_1_data,maxlags = 6)
    lo.selected_orders
    
    
    lag_order = 4
    
    # forecasts_dict, rmse_dict = ahf.rolling_forecasts_var_model(model_1_data,
    #                             model_1_input_raw_data,
    #                             covid_event,
    #                             initial_training_start_date,
    #                             initial_training_end_date,
    #                             forecast_end_date,
    #                             forecast_horizons_list,
    #                             var_dict,
    #                             log_diff_var_list,
    #                             diff_var_list,
    #                             _lag_order=lag_order)
    
    
    
    initial_training_start_date = '1980-04-01'
    #initial_training_end_date = '2021-01-01'
    initial_training_end_date = '2022-04-01'
    #forecast_start_date = '2021-04-01'
    forecast_start_date = '2022-07-01'
    forecast_end_date = '2025-10-01'
    
    # log_trans = [True,True,True,True,True,True,True]
    # first_diff= [True,True,True,True,False,True,True]
    
    # log_trans = [True,True,True,True,False,True,True]
    # first_diff= [True,True,True,True,False,True,True]
    
    # log_trans = [True,True,True,True,False,True,True]
    # first_diff= [True,True,True,True,True,True,True]
    
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
    with pd.ExcelWriter('Output/Model 1/with Hours Worked/Model 1 Irfs.xlsx') as writer:  # doctest: +SKIP
        for i in range(n_steps):
            pd.DataFrame(_irf.orth_irfs[i]).to_excel(
                writer, startrow=(n_variables+2)*i, header=False)
    results.params.drop('EVENT').to_csv('Output/Model 1/with Hours Worked/Model 1 Coeffs.csv')
    pd.DataFrame(model_1_data.loc[initial_training_end_date]).transpose(
        ).to_csv('Output/Model 1/with Hours Worked/Model 1 Data bef Forecast.csv')
    pd.DataFrame(model_1_input_raw_data.loc[initial_training_end_date]).transpose(
        ).to_csv('Output/Model 1/with Hours Worked/Model 1 Data bef Forecast Orig Scale.csv')
    forecasts_trans_df.to_csv(
        'Output/Model 1/with Hours Worked/Model 1 Initial Forecasts Transformed Form.csv')
    forecast_columns = [col for col in forecasts_new.columns\
                    if 'forecast' in col]
    forecasts_new.to_csv('Output/Model 1/with Hours Worked/Model 1 forecast new df.csv')
    
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
        'Input/Model 1/IO Model Real GDP Log first diff.csv',
        index_col=0, parse_dates=True)
    upstream_forecasts = np.array(upstream_forecasts)  
    
    col_loc_adj_col = [0,3,4]
    
  
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
                        output_location='Output/Model 1/with Hours Worked/Shocks')
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
        fig.savefig('Output/Variations/Variant 1/Model 1 '+column+'.png')
        pd.DataFrame(temp_dict,index=forecasts_new.index).to_csv(
            'Output/Model 1/with Hours Worked/Adj Forecasts/'+column+\
                '_adj_forecasts_model_1.csv')
        pd.DataFrame(temp_dict,index=forecasts_new.index).to_csv(
            'Output/Variations/Variant 1/Adj Forecasts/Model 1/'+column+\
                '_adj_forecasts_model_1.csv')
    
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
            
