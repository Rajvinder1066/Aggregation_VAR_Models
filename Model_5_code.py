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
import os
import warnings
warnings.filterwarnings("ignore")


#############################################################################

# Start of main code
def model_5_with_HW(last_iteration=16, output_folder='Variant 1'
                                   ,recomputation=False):
    # This reads csv file with input data into model_1_input_raw_data
    model_1_input_raw_data = pd.read_csv(
        'Input/Model 5/with Hours Worked/Model 5 Desired Model Input Data with Unemployment Rate.csv',
                                         parse_dates=True,
                                         index_col=0)
    
    # model_1_input_raw_data = pd.read_csv('DesiredModel1Data_12Aug.csv',
    #                                      parse_dates=True,
    #                                      index_col=0)
    
    covid_event = pd.DataFrame(model_1_input_raw_data[['Real GDP', 'Housing Starts']])
    covid_event = covid_event.rename(columns={'Real GDP':'EVENT',
                                              'Housing Starts':'FIN_CRISIS08'})
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
    # Following list is a list of variables which are transformed during
    # pre-processing step in difference log format
    
    log_diff_var_list = ['Housing Starts','Real GDP','Housing Price Index',
                         'Unemployment Rate','3MInterestRate']
    
    diff_var_list = []
    log_list = []
    var_dict ={'dlHousingStarts': 'Housing Starts',
               'dlRealGDP': 'Real GDP',
               'dlHousingPriceIndex': 'Housing Price Index',
               'dlUnemploymentRate': 'Unemployment Rate',
               'dl3MInterestRate': '3MInterestRate'
               }
    
    short_names_dict = {'dlHousing Starts':'dlHousingStarts',
                        'dlReal GDP': 'dlRealGDP',
                        'dlHousing Price Index': 'dlHousingPriceIndex',
                        'dlUnemployment Rate':'dlUnemploymentRate',
                        'dl3MInterestRate': 'dl3MInterestRate'}
    
    # log_diff_var_list = ['Real GDP', 'Real PCE', 'Real Investment',
    #            'Real Disposable Income',
    #            'Consumer price index']
    # diff_var_list = ['UnemploymentRate','10YTreasuryNotesRate']
    
    # var_dict ={'dlRealGDP': 'Real GDP', 'dlConsumptionExpenditure': 'Real PCE',
    #            'dlInvestment': 'Real Investment',
    #            'dlRealDisposableIncome': 'Real Disposable Income',
    #            'dUnemploymentRate': 'UnemploymentRate',
    #            'dlPrices': 'Consumer price index',
    #            '10YTreasuryNotesRate':'10YTreasuryNotesRate'}
    
    # short_names_dict = {'dlReal GDP': 'dlRealGDP',
    #                     'dlReal PCE':'dlConsumptionExpenditure',
    #                     'dlReal Investment':'dlInvestment',
    #                     'dlReal Disposable Income':'dlRealDisposableIncome',
    #                     'dUnemploymentRate': 'dUnemploymentRate',
    #                     'dlConsumer price index':'dlPrices',
    #                     'd10YTreasuryNotesRate':'d10YTreasuryNotesRate'}
    
    #initial_training_end_date = '2021-01-01'
    initial_training_end_date = '2022-04-01'
    
    # This creates model_1_data after tranformation
    model_1_data = ahf.preprocessing(model_1_input_raw_data,
                                 log_diff_var_list,
                                 diff_var_list,
                                 log_list,
                                 short_names_dict,
                                 initial_training_end_date)
    lag_order = 3
    
    # # Block to generate rolling forecasts for RMSE analysis
    # initial_training_start_date = '1982-04-01'
    # initial_training_end_date = '2010-10-01'
    
    # forecast_end_date = '2019-10-01'
    
    # forecast_horizons_list = ['1q', '2q', '4q', '8q', '12q', '24q']
    
    # lag_order = 2
    
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
    
    
    
    initial_training_start_date = '1982-04-01'
    #initial_training_end_date = '2021-01-01'
    initial_training_end_date = '2022-04-01'
    #forecast_start_date = '2021-04-01'
    forecast_start_date = '2022-07-01'
    forecast_end_date = '2025-10-01'
    
#    log_trans = [True,True,True,True,True]
#    first_diff= [True,True,True,True,True]
    log_trans = [True,True,True,True,True]
    first_diff= [True,True,True,True,True]
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
    with pd.ExcelWriter('Output/Model 5/with Hours Worked/Model 5 Run 1 Lag 2 Covid Irfs.xlsx') as writer:  # doctest: +SKIP
        for i in range(n_steps):
            pd.DataFrame(_irf.orth_irfs[i]).to_excel(
                writer, startrow=(n_variables+2)*i, header=False)
    results.params.drop('EVENT').to_csv('Output/Model 5/with Hours Worked/Model 5 Lag 1 Run 1 Covid Coeffs.csv')
    pd.DataFrame(model_1_data.loc[initial_training_end_date]).transpose(
        ).to_csv('Output/Model 5/with Hours Worked/Model 5 Run 1 Covid Event Data bef Forecast.csv')
    pd.DataFrame(model_1_input_raw_data.loc[initial_training_end_date]).transpose(
        ).to_csv('Output/Model 5/with Hours Worked/Model 5 Run 1 Covid Event Data bef Forecast Orig Scale.csv')
    forecasts_trans_df.to_csv(
        'Output/Model 5/with Hours Worked/Model 5 Covid Event Initial Forecasts Transformed Form Lag 2 Run 1.csv')
    forecast_columns = [col for col in forecasts_new.columns\
                    if 'forecast' in col]
    forecasts_new.to_csv('Output/Model 5/with Hours Worked/Model 5 forecast new df.csv')
    
    ## Code for adjusting forecasts
    initial_forecasts = forecasts_trans_df.copy()
    initial_forecasts = np.array(initial_forecasts)
    
    irfs = results.irf(n_steps)
    irfs = irfs.orth_irfs
    
    #irfs.to_csv('Output/Model 5/with Unemployment Rate/Model 5 irfs.csv')
    
    #coeffs = results.params.drop('EVENT')
    coeffs = results.params.drop(['EVENT','FIN_CRISIS08'])
    coeffs = np.array(coeffs)
    
    #lag_order = 2
    
    # upstream_forecasts = pd.read_csv(
    #     'Model 1 Output Transformed log first diff.csv',
    #     index_col=0, parse_dates=True)
    upstream_forecasts = pd.read_csv(
    'Input/Model 5/with Unemployment Rate/Model 1 Transformed Output for Model 5.csv',
    index_col=0)
    
    upstream_forecasts = np.array(upstream_forecasts)  
    
    col_loc_adj_col = [1,3]
    #col_loc_adj_col = [1,4]
    
    bef_forecast_values= pd.DataFrame(model_1_data.loc[initial_training_end_date])\
        .transpose()
        
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
                        output_location='Output/Model 5/with Hours Worked/Shocks')
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
    
    path = 'Output/Variations/'+output_folder
    if not os.path.exists(path):
        os.makedirs(path)
    path = 'Output/Variations/'+output_folder+'/Adj Forecasts/Model 5'
    if not os.path.exists(path):
        os.makedirs(path)
    
    for column in non_forecast_columns:
        temp_dict = dict()
        temp_dict['OE_forecast'] = forecasts_new_with_adj[column]
        temp_dict['initial_forecast'] = forecasts_new_with_adj[column+'_forecast']
        temp_dict['new_adj_forecast'] = forecasts_new_with_adj[column+'_forecast_adj']
        ax = pd.DataFrame(temp_dict,index=forecasts_new.index).plot(title=column)
        #ax.set_ylim(0)
        fig = ax.get_figure()
        #fig.savefig('Output/Variations/Variant 1/Model 5'+column+'.png')
        fig.savefig('Output/Variations/'+output_folder+'/Model 5'+column+'.png')
        pd.DataFrame(temp_dict,index=forecasts_new.index).to_csv(
            'Output/Model 5/with Hours Worked/Adj Forecasts/'+column+'_adj_forecasts_model_5_covid.csv')
        #pd.DataFrame(temp_dict,index=forecasts_new.index).to_csv(
        #    'Output/Variations/Variant 1/Adj Forecasts/Model 5/'+column+\
        #        '_adj_forecasts_model_5_covid.csv')
        pd.DataFrame(temp_dict,index=forecasts_new.index).to_csv(
            'Output/Variations/'+output_folder+'/Adj Forecasts/Model 5/'+column+\
                '_adj_forecasts_model_5_covid.csv')
    
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
    
    # # code for adjusting forecasts through IRFs
    # #i_o_model_gdp_forecasts = pd.DataFrame(model_1_input_raw_data.loc[
    #    # forecast_start_date:,['Real GDP']]
    # i_o_model_gdp_forecasts= pd.read_csv('Model1 Output.csv',
    #                                      parse_dates=True,
    #                                      index_col=0)
    # new_adj_forecasts = adjust_forecasts(model_1_data, model_1_input_raw_data,
    #                                      covid_event,
    #                       i_o_model_gdp_forecasts,initial_training_start_date,
    #                       initial_training_end_date,log_trans,first_diff,lag_order,
    #                       False)
    # new_adj_forecasts.to_csv('new_adj_forecasts_1Variable_model_5_covid_2.csv')
    # forecasts_new_with_adj = forecasts_new.join(new_adj_forecasts, rsuffix='_adj')
    
    # for column in model_1_input_raw_data.columns:
    #     temp_dict = dict()
    #     temp_dict['OE_forecast'] = forecasts_new_with_adj[column]
    #     temp_dict['initial_forecast'] = forecasts_new_with_adj[column+'_forecast']
    #     temp_dict['new_adj_forecast'] = forecasts_new_with_adj[column+'_adj']
    #     pd.DataFrame(temp_dict,index=new_adj_forecasts.index).plot(title=column)
    #     pd.DataFrame(temp_dict,index=new_adj_forecasts.index).to_csv(column+'_adj_forecasts_model_5_covid.csv')
        
    # # end of code for adjusting forecasts through IRFs
    
    # # Model Diagnostics at different lags
    # training_sample = model_1_data.loc[
    #         initial_training_start_date:initial_training_end_date]
    # keys = ['Lagrange Multiplier statistic:', 'LM test\'s p-value:', 'F-statistic:', 'F-test\'s p-value:']
    # for lag_number in range(1,10,1):
    #     print("Results for lag order",lag_number)
    #     model = VAR(training_sample)
    #     results = model.fit(lag_number)
    #     print(results.test_normality())
    #     print(results.test_whiteness())
    #     print(results.is_stable())
        
    #     exog_var = training_sample.copy()
    #     for i in range(lag_number):
    #         exog_var = exog_var.join(training_sample.shift(i+1),
    #                                  rsuffix='lag_'+str(i+1))
        
    #     lag_columns = [col for col in exog_var.columns\
    #                             if 'lag' in col]
    #     lag_columns.insert(0,'const')
    #     exog_var['const'] = np.ones(len(exog_var))
    #     exog_var = exog_var[lag_columns]
    #     exog_var = exog_var.dropna()
    #     exog_var = np.array(exog_var)
        
    #     for column in results.resid.columns:
    #         het_res = het_breuschpagan(results.resid[column], exog_var)
    #         print(lzip(keys,het_res))
    #     print("-----------------------------------")
    
def model_5_with_M2(last_iteration=17, output_folder='Variant 1'):
    # This reads csv file with input data into model_1_input_raw_data
    model_1_input_raw_data = pd.read_csv(
        'Input/Model 5/with M2/Model 5 Desired Model Input Data with M2.csv',
                                         parse_dates=True,
                                         index_col=0)
    
    # model_1_input_raw_data = pd.read_csv('DesiredModel1Data_12Aug.csv',
    #                                      parse_dates=True,
    #                                      index_col=0)
    
    covid_event = pd.DataFrame(model_1_input_raw_data[['Real GDP', 'Housing Starts']])
    covid_event = covid_event.rename(columns={'Real GDP':'EVENT',
                                              'Housing Starts':'FIN_CRISIS08'})
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
    # Following list is a list of variables which are transformed during
    # pre-processing step in difference log format
    
    log_diff_var_list = ['Housing Starts','Real GDP','Housing Price Index','M2',
                         '3MInterestRate']
        
    diff_var_list = []
    log_list = []
    var_dict ={'dlHousingStarts': 'Housing Starts',
               'dlRealGDP': 'Real GDP',
               'dlHousingPriceIndex': 'Housing Price Index',
               'dlM2': 'M2',
               'dl3MInterestRate': '3MInterestRate'
               }
    
    short_names_dict = {'dlHousing Starts':'dlHousingStarts',
                        'dlReal GDP': 'dlRealGDP',
                        'dlHousing Price Index': 'dlHousingPriceIndex',
                        'dlM2':'dlM2',
                        'dl3MInterestRate': 'dl3MInterestRate'}
    
    # log_diff_var_list = ['Real GDP', 'Real PCE', 'Real Investment',
    #            'Real Disposable Income',
    #            'Consumer price index']
    # diff_var_list = ['UnemploymentRate','10YTreasuryNotesRate']
    
    # var_dict ={'dlRealGDP': 'Real GDP', 'dlConsumptionExpenditure': 'Real PCE',
    #            'dlInvestment': 'Real Investment',
    #            'dlRealDisposableIncome': 'Real Disposable Income',
    #            'dUnemploymentRate': 'UnemploymentRate',
    #            'dlPrices': 'Consumer price index',
    #            '10YTreasuryNotesRate':'10YTreasuryNotesRate'}
    
    # short_names_dict = {'dlReal GDP': 'dlRealGDP',
    #                     'dlReal PCE':'dlConsumptionExpenditure',
    #                     'dlReal Investment':'dlInvestment',
    #                     'dlReal Disposable Income':'dlRealDisposableIncome',
    #                     'dUnemploymentRate': 'dUnemploymentRate',
    #                     'dlConsumer price index':'dlPrices',
    #                     'd10YTreasuryNotesRate':'d10YTreasuryNotesRate'}
    
    #initial_training_end_date = '2021-01-01'
    initial_training_end_date = '2021-07-01'
    # This creates model_1_data after tranformation
    model_1_data = ahf.preprocessing(model_1_input_raw_data,
                                 log_diff_var_list,
                                 diff_var_list,
                                 log_list,
                                 short_names_dict,
                                 initial_training_end_date)
    
    lag_order = 2
    # # Block to generate rolling forecasts for RMSE analysis
    # initial_training_start_date = '1982-04-01'
    # initial_training_end_date = '2010-10-01'
    
    # forecast_end_date = '2019-10-01'
    
    # forecast_horizons_list = ['1q', '2q', '4q', '8q', '12q', '24q']
    
    # lag_order = 2
    
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
    
    
    
    initial_training_start_date = '1982-04-01'
    #initial_training_end_date = '2021-01-01'
    initial_training_end_date = '2021-07-01'
    #forecast_start_date = '2021-04-01'
    forecast_start_date = '2021-10-01'
    forecast_end_date = '2025-10-01'
    
    log_trans = [True,True,True,True,True]
    first_diff= [True,True,True,True,True]
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
    with pd.ExcelWriter('Output/Model 5/with M2/Model 5 Run 1 Lag 2 Covid Irfs.xlsx') as writer:  # doctest: +SKIP
        for i in range(n_steps):
            pd.DataFrame(_irf.orth_irfs[i]).to_excel(
                writer, startrow=(n_variables+2)*i, header=False)
    results.params.drop('EVENT').to_csv('Output/Model 5/with Unemployment Rate/Model 5 Lag 1 Run 1 Covid Coeffs.csv')
    pd.DataFrame(model_1_data.loc[initial_training_end_date]).transpose(
        ).to_csv('Output/Model 5/with M2/Model 5 Run 1 Covid Event Data bef Forecast.csv')
    pd.DataFrame(model_1_input_raw_data.loc[initial_training_end_date]).transpose(
        ).to_csv('Output/Model 5/with M2/Model 5 Run 1 Covid Event Data bef Forecast Orig Scale.csv')
    forecasts_trans_df.to_csv(
        'Output/Model 5/with M2/Model 5 Covid Event Initial Forecasts Transformed Form Lag 2 Run 1.csv')
    forecast_columns = [col for col in forecasts_new.columns\
                    if 'forecast' in col]
    forecasts_new.to_csv('Output/Model 5/with M2/Model 5 forecast new df.csv')
    
    ## Code for adjusting forecasts
    initial_forecasts = forecasts_trans_df.copy()
    initial_forecasts = np.array(initial_forecasts)
    
    irfs = results.irf(n_steps)
    irfs = irfs.orth_irfs
    
    #coeffs = results.params.drop('EVENT')
    coeffs = results.params.drop(['EVENT','FIN_CRISIS08'])
    coeffs = np.array(coeffs)
    
    #lag_order = 2
    
    # upstream_forecasts = pd.read_csv(
    #     'Model 1 Output Transformed log first diff.csv',
    #     index_col=0, parse_dates=True)
    upstream_forecasts = pd.read_csv(
    'Input/Model 5/with M2/Model 1 Transformed Output for Model 5.csv',
    index_col=0)
    
    upstream_forecasts = np.array(upstream_forecasts)  
    
    col_loc_adj_col = [1,3]
    
    bef_forecast_values= pd.DataFrame(model_1_data.loc[initial_training_end_date])\
        .transpose()
        
    bef_forecast_values = np.array(bef_forecast_values)  
    
    new_forecasts = ahf.adjust_forecasts(
                        initial_forecasts,
                        irfs,
                        coeffs,
                        lag_order,
                        upstream_forecasts,
                        col_loc_adj_col,
                        bef_forecast_values,
                        last_iteration)
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
    
    path = 'Output/Variations/'+output_folder
    if not os.path.exists(path):
        os.makedirs(path)
    path = 'Output/Variations/'+output_folder+'/Adj Forecasts/Model 5'
    if not os.path.exists(path):
        os.makedirs(path)
    
    for column in non_forecast_columns:
        temp_dict = dict()
        temp_dict['OE_forecast'] = forecasts_new_with_adj[column]
        temp_dict['initial_forecast'] = forecasts_new_with_adj[column+'_forecast']
        temp_dict['new_adj_forecast'] = forecasts_new_with_adj[column+'_forecast_adj']
        ax = pd.DataFrame(temp_dict,index=forecasts_new.index).plot(title=column)
        #ax.set_ylim(0)
        fig = ax.get_figure()
        fig.savefig('Output/Variations/'+output_folder+'/Model 5'+column+'.png')
        pd.DataFrame(temp_dict,index=forecasts_new.index).to_csv(
            'Output/Model 5/with M2/Adj Forecasts/'+column+'_adj_forecasts_model_5_covid.csv')
        pd.DataFrame(temp_dict,index=forecasts_new.index).to_csv(
            'Output/Variations/'+output_folder+'/Adj Forecasts/Model 5/'+column+\
                '_adj_forecasts_model_5_covid.csv')
    
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
    
    # # code for adjusting forecasts through IRFs
    # #i_o_model_gdp_forecasts = pd.DataFrame(model_1_input_raw_data.loc[
    #    # forecast_start_date:,['Real GDP']]
    # i_o_model_gdp_forecasts= pd.read_csv('Model1 Output.csv',
    #                                      parse_dates=True,
    #                                      index_col=0)
    # new_adj_forecasts = adjust_forecasts(model_1_data, model_1_input_raw_data,
    #                                      covid_event,
    #                       i_o_model_gdp_forecasts,initial_training_start_date,
    #                       initial_training_end_date,log_trans,first_diff,lag_order,
    #                       False)
    # new_adj_forecasts.to_csv('new_adj_forecasts_1Variable_model_5_covid_2.csv')
    # forecasts_new_with_adj = forecasts_new.join(new_adj_forecasts, rsuffix='_adj')
    
    # for column in model_1_input_raw_data.columns:
    #     temp_dict = dict()
    #     temp_dict['OE_forecast'] = forecasts_new_with_adj[column]
    #     temp_dict['initial_forecast'] = forecasts_new_with_adj[column+'_forecast']
    #     temp_dict['new_adj_forecast'] = forecasts_new_with_adj[column+'_adj']
    #     pd.DataFrame(temp_dict,index=new_adj_forecasts.index).plot(title=column)
    #     pd.DataFrame(temp_dict,index=new_adj_forecasts.index).to_csv(column+'_adj_forecasts_model_5_covid.csv')
        
    # # end of code for adjusting forecasts through IRFs
    
    # # Model Diagnostics at different lags
    # training_sample = model_1_data.loc[
    #         initial_training_start_date:initial_training_end_date]
    # keys = ['Lagrange Multiplier statistic:', 'LM test\'s p-value:', 'F-statistic:', 'F-test\'s p-value:']
    # for lag_number in range(1,10,1):
    #     print("Results for lag order",lag_number)
    #     model = VAR(training_sample)
    #     results = model.fit(lag_number)
    #     print(results.test_normality())
    #     print(results.test_whiteness())
    #     print(results.is_stable())
        
    #     exog_var = training_sample.copy()
    #     for i in range(lag_number):
    #         exog_var = exog_var.join(training_sample.shift(i+1),
    #                                  rsuffix='lag_'+str(i+1))
        
    #     lag_columns = [col for col in exog_var.columns\
    #                             if 'lag' in col]
    #     lag_columns.insert(0,'const')
    #     exog_var['const'] = np.ones(len(exog_var))
    #     exog_var = exog_var[lag_columns]
    #     exog_var = exog_var.dropna()
    #     exog_var = np.array(exog_var)
        
    #     for column in results.resid.columns:
    #         het_res = het_breuschpagan(results.resid[column], exog_var)
    #         print(lzip(keys,het_res))
    #     print("-----------------------------------")