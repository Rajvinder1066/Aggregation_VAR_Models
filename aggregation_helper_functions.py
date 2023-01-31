# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 12:08:11 2021

@author: Avijit.Nandy
"""
import numpy as np
import pandas as pd
from pandas import ExcelWriter

from statsmodels.tsa.stattools import adfuller, kpss

from statsmodels.tsa.api import VAR
from statsmodels.tools.eval_measures import rmse
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.compat import lzip
from matplotlib import pyplot as plt
import scipy.stats as stats
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.diagnostic import acorr_lm, acorr_breusch_godfrey
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings("ignore")

#import os
import statsmodels.api as sm
     
def UnitRootAdfTest(colname, data, con_interval):
    print('Augmented Dickey-Fuller Test Results of ' + colname)
    res = adfuller(data,autolag='AIC')
    labels = ['ADF_Test_Stat','P_Value','Number_of_Lags',
              'Number_of_Observations']
    output = pd.Series(res[0:4], index=labels)


    for k,v in res[4].items():
        output[f'critical value({k})'] = v

    print(output.to_string())
    if res[1] <= 1 - con_interval:
        print('Null rejected at '+str(con_interval)+' confidence interval')
        print('No unit root')
    else:
        print('Null accepted at '+str(con_interval)+' confidence interval')
        print('Unit root Exist')
    print('\n')


def UnitRootkpss(colname, data):
    con_interval = .95
    print('KPSS Test Results of ' + colname)
    print('\n')
    res = kpss(data)
    labels = ['KPSS_Stat','P_Value','Number_of_Lags']
    output = pd.Series(res[0:3], index=labels)

    for k,v in res[3].items():
            output[f'critical value({k})'] = v
    print(output.to_string())
    if res[1] <= 1 - con_interval:
        print('Null rejected at '+str(con_interval)+' confidence interval')
        print('Unit Root Exist')
    else:
        print('Null accepted at '+str(con_interval)+' confidence interval')
        print('No Unit Root')

def kpss_test(series, **kw):    
    statistic, p_value, n_lags, critical_values = kpss(series, **kw)
    # Format Output
    print(f'KPSS Statistic: {statistic}')
    print(f'p-value: {p_value}')
    print(f'num lags: {n_lags}')
    print('Critial Values:')
    for key, value in critical_values.items():
        print(f'   {key} : {value}')
    print(f'Result: The series is {"not " if p_value < 0.05 else ""}stationary')
    print('\n')

def Casualty_test(PassTheDataFrame, PasssTheModel):
    
    df2 = PassTheDataFrame
    Model_res = PasssTheModel
    causRes = pd.DataFrame(columns=['Null H0','Caused','Causing','Conclusion'])

    for col in df2.columns:
        for col2 in df2.columns:
            causRes = causRes.append({'Null H0': Model_res.test_granger_causality(caused=col, causing = col2).h0,
                                      'Caused': col, 'Causing': col2, 'Conclusion':Model_res.test_granger_causality(caused=col, causing = col2).conclusion}, ignore_index=True)
    return causRes

def AutoCorrtest(PassTheModel,maximumlags):
    Model_res = PassTheModel
    lg = maximumlags
    AutoCorrTestRes = pd.DataFrame(columns=['Null H0','Lags','Conclusion'])
    for i in range(lg):
        AutoCorrTestRes = AutoCorrTestRes.append({'Null H0':Model_res.test_whiteness(nlags=i).h0,'Lags':i,'Conclusion':Model_res.test_whiteness(nlags=i).conclusion},ignore_index=True) 
    return AutoCorrTestRes 

def difftrans(PassTheDataFrame,col_list = [1,1,1,1,1,1,1]):
    res = pd.DataFrame()
    df2 = PassTheDataFrame
    for i in range(len(df2.columns)):
        if(col_list[i] == 1):
            new = pd.DataFrame(np.diff(df2.iloc[:,i]))
            new.columns = [df2.columns[i]]
            res = pd.concat([res.reset_index(drop=True), new.reset_index(drop=True)], axis=1)
        else:
            new2 = pd.DataFrame(df2.iloc[1:,i])
            new2.columns = [df2.columns[i]]
            res = pd.concat([res.reset_index(drop=True), new2.reset_index(drop=True)], axis=1)

    return res

def results_irf(passIRF):
    temp_irf = np.zeros((int(len(passIRF)/len(passIRF.columns)),len(passIRF.columns),len(passIRF.columns)))
    counter = 0
    for i in range(int(len(passIRF)/len(passIRF.columns))):
        for r in range(len(passIRF.columns)):
            for c in range(len(passIRF.columns)):
                temp_irf[i][r][c] = passIRF.iat[counter,c]
            counter = counter + 1               
    return temp_irf

def gen_forecasts(_coeffs,_values):
    #_forecasts = np.zeros(_values.shape)
    n_cols = _coeffs.shape[1]
    _forecasts = np.zeros(n_cols)
    for i in range(n_cols):
        _forecasts[i] = _coeffs[0][i]+ np.sum(np.multiply(
            _coeffs[1:,i].reshape(_values.shape),_values))
    return _forecasts

def transformation(original_data,
                   log_trans,
                   first_diff,
                   data_bef_forecast,
                   multiplier=1):
    """

    Parameters
    ----------
    original_data : TYPE
        DESCRIPTION.
    log_trans : list
        List of flags indicating whether a variable is log transformed.
    first_diff : list
        List of flags indicating whether a variable is first differenced.

    Returns
    -------
    transformed_data.

    """
    transformed_data = original_data.copy()
    if transformed_data.ndim > 1:
        dim = transformed_data.shape[1]
    else:
        dim = 1
        n_rows = transformed_data.shape[0]
        #transformed_data = transformed_data.reshape((n_rows,dim))
    
    
    if True in first_diff:
        original_data = np.insert(original_data,0,
                                  data_bef_forecast)
    original_data = original_data.reshape(original_data.shape[0],dim)   
    transformed_data = transformed_data.reshape(transformed_data.shape[0],dim)   
    for i in range(dim):
        if log_trans[i] and first_diff[i]:
            # transformed_data[:,i] = np.insert(transformed_data[:,i],0,
            #                                   data_bef_forecast[:,i])
            transformed_data[:,i] = multiplier*(np.diff(np.log(original_data[:,i])))
        elif log_trans[i] and not first_diff[i]:
            transformed_data[:,i] = np.log(original_data[:,i])
        elif not log_trans[i] and first_diff[i]:
            transformed_data[:,i] = np.diff(original_data[:,i])
        else:
            transformed_data[:,i] = original_data[:,i]
    return transformed_data

def invert_transformation(df_train, df_forecast, log_trans,first_diff):
    """Revert back the differencing to get the forecast to original scale."""
    df_fc = df_forecast.copy()
    columns = df_forecast.columns
    train_df_columns = df_train.columns
    for i in range(len(columns)):
        col = columns[i]
        orig_col = train_df_columns[i]

        if log_trans[i] and first_diff[i]:
            df_fc[str(col)+'_forecast'] = df_train[orig_col].iloc[-1] *\
                np.exp((df_fc[str(col)].cumsum())/100)
        elif not log_trans[i] and first_diff[i]:
            df_fc[str(col)+'_forecast'] = df_train[orig_col].iloc[-1] + \
                df_fc[str(col)].cumsum()
        elif log_trans[i] and not first_diff[i]:
            df_fc[str(col)+'_forecast'] = np.exp(df_fc[str(col)])
        else:
            df_fc[str(col)+'_forecast'] = df_fc[str(col)]
    return df_fc

def adjust_forecasts(
        initial_forecasts,
        irfs,
        coeffs,
        lag_order,
        upstream_forecasts,
        col_loc_adj_col,
        bef_forecast_values,
        last_iteration=4,
        results=None,
        output_location=None):
    """ This function is used to generate adjusted forecasts.
        initial_forecasts: initial forecasts produced by the model
                           shape - no of steps X no of variables
        irfs: irfs from the model
              shape - no of steps X no of variables X no of variables
        coeffs: coeffs of the model equations
                shape - [(lag order X no of variables) + 1] X no of variables
        lag_order: Lag order of the model - integer
        upstream_forecasts: forecasts from the parent model
                            shape - no of steps X no of upstream forecasts
        col_loc_adj_col: locations of the columns in initial_forecasts which
                         will be equated with upstream forecasts
        bef_forecast_values: values in the same scale as initial_forecasts
                             to be used in case of lag Order>=2 for calculating
                             new forecasts based on updated current period data
                             Should have lag_order - 1 steps before forecast
                             data
    """
    # Fit the model
    #_i_o_model_gdp_forecasts = this is forecast from the macrofile
    #Generate Forecasts
    n_steps = min(len(upstream_forecasts),last_iteration)
                
    # Prepare data needed for adjustments
    n_variables = initial_forecasts.shape[1]
    n_forecast_cols = upstream_forecasts.shape[1]
    
    new_forecasts = initial_forecasts.copy()
    cur_period_arr = np.zeros((n_steps,n_forecast_cols))
    
    #midx = pd.MultiIndex.from_product([np.array(range(n_forecast_cols)),
    #                                  np.array(range(n_variables))])
    #Loop to go over each forecast from Upstream Model
    shocks_adj_cols_df_list = []
    for i in range(n_steps):
        current_shock = np.zeros(irfs[0,:,0].shape)
        prev_period_shocks_df = pd.DataFrame(
            columns=np.array(range(
            n_forecast_cols)),
            index=np.array(range(n_variables))).fillna(0)
        for col_no in range(n_forecast_cols):
            col_loc = col_loc_adj_col[col_no]
            irfs_arr = np.zeros((n_variables,i))
            # Loop to adjust forecast at each IRF
            for j in range(i,-1,-1):
                if j > 0:
                    irfs_arr[:,i-j] = irfs[j,:,col_loc]        
                elif j==0:
                    if i > 0:
                        ones_vector = np.ones((1,n_variables))
                        err_upto_cur_period = cur_period_arr[
                            :i,col_no].reshape(i,1)
                        cur_err_mat = np.matmul(err_upto_cur_period,
                                                ones_vector)
                        shocks_mat = np.matmul(irfs_arr,cur_err_mat)
                        current_shock = current_shock + np.diag(shocks_mat)
                        prev_period_shocks_df[col_no] = np.diag(shocks_mat)
                        
        new_forecasts[i] = new_forecasts[i] + current_shock

        for col_no in range(n_forecast_cols):
            col_loc = col_loc_adj_col[col_no]
            cur_upstream_cur_col_forecast = upstream_forecasts[
                i, col_no]
            cur_initial_cur_col_forecast = new_forecasts[
                i,col_loc]
            # cur_initial_cur_col_forecast = initial_forecasts[
            #     i,col_loc]
            irf_cur_col_to_self = irfs[j][col_loc][col_loc]
            if col_no == 0:
                cur_err = (
                    cur_upstream_cur_col_forecast\
                        - cur_initial_cur_col_forecast\
                         )/irf_cur_col_to_self
            elif col_no == 1:
                irf_cur_col_to_prev_col = irfs[j][col_loc][prev_col_loc]
                cur_err = (cur_upstream_cur_col_forecast \
                           - cur_initial_cur_col_forecast \
                           - prev_err*irf_cur_col_to_prev_col \
                          )/irf_cur_col_to_self
            cur_period_arr[i,col_no] = cur_err
            prev_col_loc = col_loc
            prev_err = cur_err
        
        #new_forecasts[i] = new_forecasts[i] + current_shock
        
        current_period_shocks_df = pd.DataFrame(
            columns=np.array(range(
            n_forecast_cols)),
            index=np.array(range(n_variables))).fillna(0)
        
        current_period_shocks = np.zeros(irfs[0,:,0].shape)
        for col_no in range(n_forecast_cols):
            col_loc = col_loc_adj_col[col_no]
            cur_err = cur_period_arr[i,col_no]
            current_period_shocks = current_period_shocks \
            + np.dot(cur_err, irfs[0,:,col_loc])
            current_period_shocks_df[col_no] = np.dot(cur_err,
                                                      irfs[0,:,col_loc])
        total_shocks_df = prev_period_shocks_df + current_period_shocks_df
        consolidated_shocks_df = prev_period_shocks_df.join(
            current_period_shocks_df, rsuffix='_current', lsuffix='_prev')
        consolidated_shocks_df = pd.concat(
            [consolidated_shocks_df, total_shocks_df],axis=1)
        new_forecasts[i] = new_forecasts[i] + current_period_shocks
        
        shocks_adj_cols_df_list.append(consolidated_shocks_df)
        if coeffs is not None:
            if i<n_steps-1:
                lagged_values_array = np.zeros(n_variables*lag_order)
                for lag_iter in range(lag_order):
                    if i - lag_iter>=0:
                        lagged_values_array[
                            (lag_iter*n_variables):(lag_iter+1)*n_variables] = \
                            new_forecasts[i-lag_iter]
                    else:
                        lagged_values_array[
                            (lag_iter*n_variables):(lag_iter+1)*n_variables] = \
                            bef_forecast_values[lag_order - (lag_iter+1)]
                if coeffs is not None:
                    new_forecasts[i+1] = gen_forecasts(coeffs,lagged_values_array)
    
    # Code block to write each step output to different sheet
    if output_location is not None:
        with ExcelWriter(output_location+'/shocks.xlsx') as writer:
            for n, df in enumerate(shocks_adj_cols_df_list):
                df.to_excel(writer,'sheet%s' % n)
        writer.save()
        
        pd.DataFrame(cur_period_arr).to_csv(output_location+'/errors_adj.csv')
        
    return new_forecasts
 
def preprocessing(_model_1_input_raw_data, _log_diff_var_list,_diff_var_list, 
                  _log_list,
                  _short_names_dict,_initial_training_end_date):
    """ This function is used for data pre processing and checking unit roots.
        Currently, this function only supports difference log transforms.
        _log_diff_var_list is a list of variabls which will be transformed.
    """
    _model_1_data = _model_1_input_raw_data.copy()
    for i in range(len(_log_diff_var_list)):
        _model_1_data['dl' + _log_diff_var_list[i]] = 100*(np.log(
            _model_1_data[_log_diff_var_list[i]]).diff())
    for i in range(len(_diff_var_list)):
        _model_1_data['d' + _diff_var_list[i]] = _model_1_data[
            _diff_var_list[i]].diff()
    for i in range(len(_log_list)):
        _model_1_data['l' + _log_list[i]] = np.log(_model_1_data[
            _log_list[i]]) 
    #model_1_data['lHoursWorked'] = 100*(np.log(model_1_data['HoursWorked']))
    

    _model_1_data.rename(columns=_short_names_dict, inplace=True)
    
    _model_1_data = _model_1_data.loc[:,_short_names_dict.values()].dropna()
    for (columnName, columnData) in _model_1_data.loc[:
                                _initial_training_end_date].iteritems():
        UnitRootAdfTest(columnName, columnData,.95)
        kpss_test(columnData)
    return _model_1_data

def generate_rmse(_model_1_data,_forecasts_dict,_forecast_horizons_list):
    """ This function is used for creating rmse values.
        It creates a nested dictionary with column at first level,
        horizon at second level and stores actual and forecast values
        for calculating rmse values.
    """
    rmse_dict = {}
    for col_name in _model_1_data.columns:
        rmse_dict[col_name] = {}
        for horizon in _forecast_horizons_list:
            rmse_dict[col_name][horizon] = {}
            rmse_dict[col_name][horizon] = rmse(
                _forecasts_dict[col_name][horizon]['actual'],
                _forecasts_dict[col_name][horizon]['forecast'])
    return rmse_dict

def generate_forecast_dict(_model_1_data,
                           _forecast_horizons_list,
                           _initial_training_start_date,
                           _initial_training_end_date,
                           _forecast_end_date,
                           _lag_order,
                           _model_1_input_raw_data,
                           _covid_event,
                           _var_dict,
                           _log_diff_var_list,
                           _diff_var_list,
                           _transformed=False):
    """ This function is used for creating forecasts in nested dictionaries.
        _forecast_horizons_list is a list of horizons for which forecasts
        are generated.
        _transformed is a flag which is used to compare actual vs forecasts
        in transformed form.
        Output is a nested dictionary with columns at first level,
        horizon at second level and stores forecasts and actual values.
    """
    forecast_index_start = _model_1_data.index.get_loc(
        _initial_training_end_date)
    forecast_index_end = _model_1_data.index.get_loc(_forecast_end_date)
    
    # Folllwing loop creates an empty dictionary which is used for storing
    # actual vs forecasted values for each column across different
    # horizons
    forecasts_dict = {}
    for col_name in _model_1_data.columns:
        forecasts_dict[col_name] = {}
        for horizon in _forecast_horizons_list:
            forecasts_dict[col_name][horizon] = {}
            forecasts_dict[col_name][horizon]['actual'] = []
            forecasts_dict[col_name][horizon]['forecast'] = []
    
    # This loop runs from forecast start to end
    # Expands the training in each iteration.
    # Fits a new model with expanded window in each iteration
    for date in _model_1_data.index[forecast_index_start:\
                                    forecast_index_end+1]:
        training_sample = _model_1_data.loc[_initial_training_start_date:date]
        #exog_var = _covid_event.loc[_initial_training_start_date:date]['EVENT']
        exog_var = _covid_event.loc[_initial_training_start_date:date]
        exog_last_loc = exog_var.index.get_loc(date)
        model = VAR(training_sample,exog=exog_var)
        results_lag_1 = model.fit(_lag_order)
        
        # This inner for loop runs for each column
        # Generates and stores forecast for each horizon for that column
        # Stores corresponding actual value for rmse calculation
        for i in range(len(_model_1_data.columns)):
            col_name = _model_1_data.columns[i]
            for horizon in _forecast_horizons_list:
                forecast_look_ahead = int(horizon[:-1])-1
                
                # _exog_future = _covid_event.iloc[exog_last_loc+1:exog_last_loc\
                #                 +forecast_look_ahead+2]['EVENT']
                _exog_future = _covid_event.iloc[exog_last_loc+1:exog_last_loc\
                                +forecast_look_ahead+2]
                forecasts = results_lag_1.forecast(
                    training_sample.values[-_lag_order:], forecast_look_ahead+1,
                    exog_future=_exog_future)
                forecast_location = _model_1_data.index.get_loc(date) + 1
                if forecast_location+forecast_look_ahead < \
                    len(_model_1_data):
                    if _transformed:
                        actual_value = _model_1_data.iloc[
                            forecast_location+forecast_look_ahead][
                                col_name]
                    else:
                        actual_value = _model_1_input_raw_data.iloc[
                            forecast_location+forecast_look_ahead+1][
                                _var_dict[col_name]]
    
                    if  _var_dict[col_name] in _log_diff_var_list and\
                        not _transformed:
                        forecast_value = _model_1_input_raw_data.iloc[
                            forecast_location+forecast_look_ahead][
                                _var_dict[col_name]] * np.exp(
                                    forecasts[forecast_look_ahead][i]/100)
                    elif  _var_dict[col_name] in _diff_var_list and\
                        not _transformed:
                        forecast_value = _model_1_input_raw_data.iloc[
                            forecast_location+forecast_look_ahead][
                                _var_dict[col_name]] + forecasts[
                                    forecast_look_ahead][i]
                    else:
                        forecast_value = forecasts[forecast_look_ahead][i]
                        
                    forecasts_dict[col_name][horizon]['forecast'].append(
                        forecast_value)
                    forecasts_dict[col_name][horizon]['actual'].append(
                        actual_value)
    return forecasts_dict

def rolling_forecasts_var_model(_model_1_data,
                                _model_1_input_raw_data,
                                _covid_event,
                                _initial_training_start_date,
                                _initial_training_end_date,
                                _forecast_end_date,
                                _forecast_horizons_list,
                                _var_dict,
                                _log_diff_var_list,
                                _diff_var_list,
                                _lag_order=1):


    """ This function is a wrapper function to generate rolling forecasts
    at a given frequency with expanding window and call other helper functions
    to generate forecast_dict which stores the forecasts for each column
    for each horizon.
    Then finally plots the forecasts vs actual values at each horizon
    """
    
    # This creates forecast dictionary in transformed form
    forecasts_dict = generate_forecast_dict(_model_1_data,
                               _forecast_horizons_list,
                               _initial_training_start_date,
                               _initial_training_end_date,
                               _forecast_end_date,
                               _lag_order,
                               _model_1_input_raw_data,
                               _covid_event,
                               _var_dict,
                               _log_diff_var_list,
                               _diff_var_list,
                               True)
    # This creates rmse dictionary in transformed form
    rmse_dict = generate_rmse(_model_1_data,
                              forecasts_dict,
                              _forecast_horizons_list)
    
    pd.DataFrame(rmse_dict).to_csv(
        'DesirableModel1_rmse_till2019_transformed_lag_order_'+\
            str(_lag_order) +'.csv')
    
    # This creates forecast dictionary in raw/level form
    forecasts_dict = generate_forecast_dict(_model_1_data,
                               _forecast_horizons_list,
                               _initial_training_start_date,
                               _initial_training_end_date,
                               _forecast_end_date,
                               _lag_order,
                               _model_1_input_raw_data,
                               _covid_event,
                               _var_dict,
                               _log_diff_var_list,
                               _diff_var_list,
                               False)
    # This creates rmse dictionary in raw/level form
    rmse_dict = generate_rmse(_model_1_data,
                              forecasts_dict,
                              _forecast_horizons_list)
    
    pd.DataFrame(rmse_dict).to_csv(
        'DesirableModel1_rmse_till2019_original_scale_lag_order_'+\
            str(_lag_order) +'.csv')
    
    # This for loop iterates through each horizon.
    # The inner for loop iterates through each column
    # A temp_dict is created which extracts corresponding actual
    # and forecast values from forecast_dict
    # A plot is generated comparing actuval vs forecasted values.
    for horizon in _forecast_horizons_list:
        for column in _var_dict:    
            forecast_look_ahead = int(horizon[:-1])-1
            first_forecast_location = _model_1_input_raw_data.index.\
                get_loc(_initial_training_end_date) + 1 +\
                    forecast_look_ahead
            last_forecast_location = _model_1_input_raw_data.index.\
                get_loc(_forecast_end_date)
            end_of_forecasts_location = _model_1_input_raw_data.index.\
                get_loc(_forecast_end_date) - first_forecast_location
            temp_dict = dict()
            temp_dict['actual'] = forecasts_dict[column][horizon][
                'actual'][:end_of_forecasts_location]
            temp_dict['forecast'] = forecasts_dict[column][horizon][
                'forecast'][:end_of_forecasts_location]
            pd.DataFrame(temp_dict,
                     index=_model_1_input_raw_data.iloc[
                         first_forecast_location:last_forecast_location].\
                         index).plot(title=_var_dict[column] + '_' +\
                                     horizon)    
    return forecasts_dict, rmse_dict
                                     
def generate_out_of_sample_forecasts(_initial_training_start_date,
                                     _initial_training_end_date,
                                     _forecast_end_date,
                                     _model_1_data,
                                     _model_1_input_raw_data,
                                     _covid_event,
                                     _log_trans,
                                     _first_diff,
                                     _lag_order=1):                                     

    """ This function is used to generate out of sample forecasts
        after _initial_training_end_date till _forecast_end_date.
        It also plots the forecast values vs out of sample forecasts (OE)
        from _model_1_input_raw_data
    """
    training_sample = _model_1_data.loc[
        _initial_training_start_date:_initial_training_end_date]
    # exog_var = _covid_event.loc[_initial_training_start_date:\
    #                            _initial_training_end_date]['EVENT']
    exog_var = _covid_event.loc[_initial_training_start_date:\
                               _initial_training_end_date]
    exog_last_loc = exog_var.index.get_loc(_initial_training_end_date)
    model = VAR(training_sample,exog=exog_var)
    results_lag_1 = model.fit(_lag_order)
    
    irf = results_lag_1.irf(30)
    irf.plot(orth=True)
    print("Lag order is",_lag_order)
    print(results_lag_1.summary())
    print(results_lag_1.test_normality())
    print(results_lag_1.is_stable(verbose=True))
    print(results_lag_1.test_whiteness())
       
    oe_forecast_start_date = _model_1_input_raw_data.index.get_loc(
            _initial_training_end_date)+1
    oe_forecast_end_date = _model_1_input_raw_data.index.get_loc(
        _forecast_end_date)+1
    indx = _model_1_input_raw_data.iloc[
        oe_forecast_start_date:oe_forecast_end_date].index
    # _exog_future = _covid_event.iloc[exog_last_loc+1:exog_last_loc+\
    #             oe_forecast_end_date - oe_forecast_start_date+1]['EVENT']
    _exog_future = _covid_event.iloc[exog_last_loc+1:exog_last_loc+\
                oe_forecast_end_date - oe_forecast_start_date+1]
    forecasts = results_lag_1.forecast(
                    training_sample.values[-_lag_order:],
                    oe_forecast_end_date - oe_forecast_start_date,
                    exog_future=_exog_future)
    
    forecasts_df = pd.DataFrame(forecasts,
                                columns=_model_1_input_raw_data.columns,
                                index=indx)
    
    forecasts_new = invert_transformation(_model_1_input_raw_data.loc[
                        :_initial_training_end_date],
                          forecasts_df,
                          _log_trans,
                          _first_diff)
    
    forecast_columns = [col for col in forecasts_new.columns\
                        if 'forecast' in col]
    
    forecasts_new = forecasts_new[forecast_columns].join(
        _model_1_input_raw_data)
    
    for column in _model_1_input_raw_data.columns:       
        forecasts_new.loc[:,[column+'_forecast',
                             column]].plot(title=column)        
    # forecasts_new.loc[:,['Real GDP_forecast',
    #                      'Real GDP']].plot(title='Real GDP')
    # forecasts_new.loc[:,['Real PCE_forecast',
    #                      'Real PCE']].plot(title='Real PCE')
    # forecasts_new.loc[:,['Real Investment_forecast',
    #                      'Real Investment']].plot(
    #                          title='Real Investment')
    # forecasts_new.loc[:,['Real Disposable Income_forecast',
    #                      'Real Disposable Income']].plot(
    #                          title='Real Disposable Income')
    # forecasts_new.loc[:,['UnemploymentRate_forecast',
    #                      'UnemploymentRate']].plot(
    #                          title='Unemployment Rate')
    # forecasts_new.loc[:,['Consumer price index_forecast',
    #                      'Consumer price index']].plot(
    #                          title='Consumer Price Index')
    # forecasts_new.loc[:,['10YTreasuryNotesRate_forecast',
    #                      '10YTreasuryNotesRate']].plot(
    #                          title='10Y Treasury Note Rate')
    return results_lag_1,forecasts_df, forecasts_new

# end of functions



def create_upstream_forecasts_for_model_5_with_unemployment_rate(folder='with CPI'):
    upstream_forecasts_from_model_1 = pd.read_csv(
    'Output/Model 1/'+folder+'/Adj Forecasts/Real GDP_adj_forecasts_model_1_covid.csv', index_col=0,parse_dates=True)
    col_no = 2
    upstream_forecasts_from_model_1 = upstream_forecasts_from_model_1.iloc[:,col_no]
    original_data = upstream_forecasts_from_model_1.copy()
    
    log_trans=[True]
    first_diff=[True]
    model_1_input_raw_data = pd.read_csv(
        'Input/Model 1/with CPI/DesiredModel1Data_12Aug.csv',
        parse_dates=True,
        index_col=0)
    col_loc = 0
    data_bef_forecast = np.array(model_1_input_raw_data.iloc[model_1_input_raw_data.index.get_loc(original_data.index[0]) -1,col_loc])
    original_data = np.array(original_data)
    real_gdp_transformed = pd.DataFrame(transformation(original_data,
                       log_trans,
                       first_diff,
                       data_bef_forecast,
                       multiplier=100))
    
    upstream_forecasts_from_model_1 = pd.read_csv(
    'Output/Model 1/'+folder+'/Adj Forecasts/UnemploymentRate_adj_forecasts_model_1_covid.csv', index_col=0,parse_dates=True)
    col_no = 2
    upstream_forecasts_from_model_1 = upstream_forecasts_from_model_1.iloc[:,col_no]
    original_data = upstream_forecasts_from_model_1.copy()
    
    log_trans=[True]
    first_diff=[True]
    model_1_input_raw_data = pd.read_csv(
        'Input/Model 1/with CPI/DesiredModel1Data_12Aug.csv',
        parse_dates=True,
        index_col=0)
    col_loc = 4
    data_bef_forecast = np.array(model_1_input_raw_data.iloc[model_1_input_raw_data.index.get_loc(original_data.index[0]) -1,col_loc])
    original_data = np.array(original_data)
    unemp_rate_transformed = pd.DataFrame(transformation(original_data,
                       log_trans,
                       first_diff,
                       data_bef_forecast,
                       multiplier=100))
    
    real_gdp_transformed.join(unemp_rate_transformed,rsuffix='unemp_rate').to_csv('Input/Model 5/with Unemployment Rate/Model 1 Transformed Output for Model 5.csv')

def create_upstream_forecasts_for_model_5_with_M2(folder='with GDP deflator'):
    upstream_forecasts_from_model_1 = pd.read_csv(
    'Output/Model 1/'+folder+'/Adj Forecasts/Real GDP_adj_forecasts_model_1_covid.csv', index_col=0,parse_dates=True)
    col_no = 2
    upstream_forecasts_from_model_1 = upstream_forecasts_from_model_1.iloc[:,col_no]
    original_data = upstream_forecasts_from_model_1.copy()
    
    log_trans=[True]
    first_diff=[True]
    model_1_input_raw_data = pd.read_csv(
        'Input/Model 1/with GDP Deflator/DesiredModel1Data_25Aug_with_GDP_deflat.csv',
        parse_dates=True,
        index_col=0)
    col_loc = 0
    data_bef_forecast = np.array(model_1_input_raw_data.iloc[model_1_input_raw_data.index.get_loc(original_data.index[0]) -1,col_loc])
    original_data = np.array(original_data)
    pd.DataFrame(transformation(original_data,
                       log_trans,
                       first_diff,
                       data_bef_forecast,
                       multiplier=100)).to_csv('Input/Model 5/with M2/Model 1 Transformed Output for Model 5.csv')


def create_upstream_forecasts_for_model_1(initial_training_end_date='2021-10-01'):
    io_gdp = pd.read_csv(
    'Input/Model 1/IO Model Input Data.csv',
    index_col=0, parse_dates=True)

    #decompose_data = seasonal_decompose(io_gdp['Nominal GDP'].dropna(), model="additive")
    decompose_data = seasonal_decompose(io_gdp['Nominal GDP'], model="additive")
    io_gdp['Seasonally Adjusted Nominal GDP'] = io_gdp['Nominal GDP'] - \
        decompose_data.seasonal
    io_gdp['Seasonally Adjusted Real GDP'] = io_gdp[
        'Seasonally Adjusted Nominal GDP'] / (io_gdp['GDP Deflator']/100)
    io_gdp = io_gdp.dropna()
    io_gdp['QoQ Real GDP Growth Rate'] = io_gdp['Seasonally Adjusted Real GDP'].pct_change()
    io_gdp['intermediate_growth'] = 1+io_gdp['QoQ Real GDP Growth Rate']
    io_gdp['Quarterly Final Real GDP'] = io_gdp['intermediate_growth'].cumprod()*io_gdp.iloc[0,2]  
    gdp_data = io_gdp.loc[initial_training_end_date:].copy()
    gdp_data.at[initial_training_end_date,'Quarterly Final Real GDP'] = gdp_data.loc[initial_training_end_date,'Quarterly OE Real GDP']
    gdp_data.at[initial_training_end_date,'intermediate_growth'] = 1
    gdp_data['Quarterly Final Real GDP'] = gdp_data['intermediate_growth'].cumprod()*gdp_data.iloc[0,2]
    gdp_data['transformed_real_gdp'] = 100*(np.log(
            gdp_data['Quarterly Final Real GDP']).diff())
    gdp_data['transformed_real_gdp'].dropna().to_csv(
        'Input/Model 1/IO Model Real GDP Log first diff.csv')
    
    gdp_data.to_csv('gdp data calculation sea decomp.csv')
    io_gdp.to_csv('io gdp data calculation sea decomp.csv')
    
def create_x13_upstream_forecasts_for_model_1(
        initial_training_end_date='2022-04-01'):
    io_gdp = pd.read_csv(
    'Input/Model 1/IO Model Input Data.csv',
    index_col=0, parse_dates=True)
    
    res = sm.tsa.x13_arima_analysis(io_gdp['Nominal GDP'])
    
    io_gdp['x13 Nominal GDP'] = res.seasadj
    
    io_gdp['IO Real GDP'] = io_gdp[
        'x13 Nominal GDP'] / (io_gdp['GDP Deflator']/100)
    
    #res = sm.tsa.x13_arima_analysis(io_gdp['IO Real GDP'])
    
    io_gdp['QoQ Real GDP Growth Rate'] = io_gdp['IO Real GDP'].pct_change(
        ).dropna()
    
    io_gdp['intermediate_growth'] = 1+io_gdp['QoQ Real GDP Growth Rate']
    
    io_gdp['Quarterly Final Real GDP'] = io_gdp['intermediate_growth'].cumprod()*io_gdp.iloc[0,2]  
    
    gdp_data = io_gdp.loc[initial_training_end_date:].copy()
    
    gdp_data.at[initial_training_end_date,'Quarterly Final Real GDP'] = gdp_data.loc[initial_training_end_date,'Quarterly OE Real GDP']
    gdp_data.at[initial_training_end_date,'intermediate_growth'] = 1
    gdp_data['Quarterly Final Real GDP'] = gdp_data['intermediate_growth'].cumprod()*gdp_data.iloc[0,2]
    gdp_data['transformed_real_gdp'] = 100*(np.log(
            gdp_data['Quarterly Final Real GDP']).diff())
    gdp_data['transformed_real_gdp'].dropna().to_csv(
        'Input/Model 1/IO Model Real GDP Log first diff.csv')
    
    gdp_data.to_csv('gdp data calculation.csv')
    io_gdp.to_csv('io gdp data calculation.csv')
#############################################################################

def create_upstream_forecasts_for_model_5_3M_INT_Rate(folder='with Hours Worked'):
    upstream_forecasts_from_model_1 = pd.read_csv(
    'Output/Model 1/'+folder+'/Adj Forecasts/Real GDP_adj_forecasts_model_1.csv', index_col=0,parse_dates=True)
    col_no = 2
    upstream_forecasts_from_model_1 = upstream_forecasts_from_model_1.iloc[:,col_no]
    original_data = upstream_forecasts_from_model_1.copy()
    
    log_trans=[True]
    first_diff=[True]
    model_1_input_raw_data = pd.read_csv(
        'Input/Model 1/with Hours Worked/DesiredModel1Data_6Oct with hours.csv',
        parse_dates=True,
        index_col=0)
    col_loc = 0
    data_bef_forecast = np.array(model_1_input_raw_data.iloc[model_1_input_raw_data.index.get_loc(original_data.index[0]) -1,col_loc])
    original_data = np.array(original_data)
    real_gdp_transformed = pd.DataFrame(transformation(original_data,
                       log_trans,
                       first_diff,
                       data_bef_forecast,
                       multiplier=100))
    
    three_M_interest_data = pd.read_csv('Input/Model 5/with Hours Worked/3M Interest Rate OE Forecasts Transformed.csv',
                index_col=0,parse_dates=True).reset_index().iloc[:,1]
    
    
    # upstream_forecasts_from_model_1 = pd.read_csv(
    # 'Output/Model 1/'+folder+'/Adj Forecasts/UnemploymentRate_adj_forecasts_model_1_covid.csv', index_col=0,parse_dates=True)
    # col_no = 2
    # upstream_forecasts_from_model_1 = upstream_forecasts_from_model_1.iloc[:,col_no]
    # original_data = upstream_forecasts_from_model_1.copy()
    
    # log_trans=[True]
    # first_diff=[True]
    # model_1_input_raw_data = pd.read_csv(
    #     'Input/Model 1/with GDP deflator/DesiredModel1Data_25Aug_with_GDP_deflat.csv',
    #     parse_dates=True,
    #     index_col=0)
    # col_loc = 4
    # data_bef_forecast = np.array(model_1_input_raw_data.iloc[model_1_input_raw_data.index.get_loc(original_data.index[0]) -1,col_loc])
    # original_data = np.array(original_data)
    # unemp_rate_transformed = pd.DataFrame(transformation(original_data,
    #                    log_trans,
    #                    first_diff,
    #                    data_bef_forecast,
    #                    multiplier=100))
    
    real_gdp_transformed.join(three_M_interest_data,rsuffix='3m_rate').to_csv(
        'Input/Model 5/with Unemployment Rate/Model 1 Transformed Output for Model 5.csv')


def create_upstream_forecasts_for_model_3(folder='with Hours Worked'):
    upstream_forecasts_from_model_1 = pd.read_csv(
    'Output/Model 1/'+folder+'/Adj Forecasts/Real GDP_adj_forecasts_model_1.csv', index_col=0,parse_dates=True)
    col_no = 2
    upstream_forecasts_from_model_1 = upstream_forecasts_from_model_1.iloc[:,col_no]
    original_data = upstream_forecasts_from_model_1.copy()
    
    log_trans=[True]
    first_diff=[True]
    model_1_input_raw_data = pd.read_csv(
        'Input/Model 1/'+folder+'/DesiredModel1Data_6Oct with hours.csv',
        parse_dates=True,
        index_col=0)
    data_bef_forecast = np.array(model_1_input_raw_data.iloc[model_1_input_raw_data.index.get_loc(original_data.index[0]) -1,0])
    original_data = np.array(original_data)
    GDP_transformed = pd.DataFrame(transformation(original_data,
                       log_trans,
                       first_diff,
                       data_bef_forecast))
    if folder == 'with Hours Worked':
        upstream_forecasts_from_model_1 = pd.read_csv(
        'Output/Model 1/'+folder+'/Adj Forecasts/Consumer price index_adj_forecasts_model_1.csv', index_col=0,parse_dates=True)
        col_no = 2
        upstream_forecasts_from_model_1 = upstream_forecasts_from_model_1.iloc[:,col_no]
        original_data = upstream_forecasts_from_model_1.copy()
        
        log_trans=[True]
        first_diff=[True]
        model_1_input_raw_data = pd.read_csv(
            'Input/Model 1/with Hours Worked/DesiredModel1Data_6Oct with hours.csv',
            parse_dates=True,
            index_col=0)
        data_bef_forecast = np.array(model_1_input_raw_data.iloc[model_1_input_raw_data.index.get_loc(original_data.index[0]) -1,5])
        original_data = np.array(original_data)
        cpi_transformed = pd.DataFrame(transformation(original_data,
                           log_trans,
                           first_diff,
                           data_bef_forecast))
        cpi_transformed.join(GDP_transformed,rsuffix='GDP').to_csv('Input/Model 3/Model_1_Output_for_Model_3.csv')
    else:
        GDP_transformed.to_csv('Input/Model 3/Model_1_Output_for_Model_3.csv')
        
        
def create_upstream_forecasts_for_model_6(folder='with Hours Worked'):
    upstream_forecasts_from_model_1 = pd.read_csv(
    'Output/Model 1/'+folder+'/Adj Forecasts/Real PCE_adj_forecasts_model_1.csv', index_col=0,parse_dates=True)
    col_no = 2
    upstream_forecasts_from_model_1 = upstream_forecasts_from_model_1.iloc[:,col_no]
    original_data = upstream_forecasts_from_model_1.copy()
    
    log_trans=[True]
    first_diff=[True]
    model_1_input_raw_data = pd.read_csv(
        'Input/Model 1/'+folder+'/DesiredModel1Data_6Oct with hours.csv',
        parse_dates=True,
        index_col=0)
    data_bef_forecast = np.array(model_1_input_raw_data.iloc[model_1_input_raw_data.index.get_loc(original_data.index[0]) -1,1])
    original_data = np.array(original_data)
    PCE_transformed = pd.DataFrame(transformation(original_data,
                       log_trans,
                       first_diff,
                       data_bef_forecast))
    if folder == 'with Hours Worked':
        upstream_forecasts_from_model_1 = pd.read_csv(
        'Output/Model 1/'+folder+'/Adj Forecasts/Consumer price index_adj_forecasts_model_1.csv', index_col=0,parse_dates=True)
        col_no = 2
        upstream_forecasts_from_model_1 = upstream_forecasts_from_model_1.iloc[:,col_no]
        original_data = upstream_forecasts_from_model_1.copy()
        
        log_trans=[True]
        first_diff=[True]
        model_1_input_raw_data = pd.read_csv(
            'Input/Model 1/with Hours Worked/DesiredModel1Data_6Oct with hours.csv',
            parse_dates=True,
            index_col=0)
        data_bef_forecast = np.array(model_1_input_raw_data.iloc[model_1_input_raw_data.index.get_loc(original_data.index[0]) -1,5])
        original_data = np.array(original_data)
        cpi_transformed = pd.DataFrame(transformation(original_data,
                           log_trans,
                           first_diff,
                           data_bef_forecast))
        PCE_transformed.join(cpi_transformed,rsuffix='CPI').to_csv('Input/Model 6/Model_1_Output_for_Model_6.csv')
    else:
        PCE_transformed.to_csv('Input/Model 6/Model_1_Output_for_Model_6.csv')

def create_upstream_forecasts_for_model_2(folder='with Hours Worked',initial_training_end_date='2022-04-01'):
    upstream_forecasts_from_model_1 = pd.read_csv(
    'Output/Model 1/'+folder+'/Adj Forecasts/Real PCE_adj_forecasts_model_1.csv', index_col=0,parse_dates=True)
    col_no = 2
    upstream_forecasts_from_model_1 = upstream_forecasts_from_model_1.iloc[:,col_no]
    original_data = upstream_forecasts_from_model_1.copy()
    
    log_trans=[True]
    first_diff=[True]
    model_1_input_raw_data = pd.read_csv(
        'Input/Model 1/with Hours Worked/DesiredModel1Data_6Oct with hours.csv',
        parse_dates=True,
        index_col=0)
    data_bef_forecast = np.array(model_1_input_raw_data.iloc[model_1_input_raw_data.index.get_loc(original_data.index[0]) -1,1])
    original_data = np.array(original_data)
    pce_transformed = pd.DataFrame(transformation(original_data,
                       log_trans,
                       first_diff,
                       data_bef_forecast))
    if folder == 'with Hours Worked':
        upstream_forecasts_from_model_1 = pd.read_csv(
        'Output/Model 1/'+folder+'/Adj Forecasts/Consumer price index_adj_forecasts_model_1.csv', index_col=0,parse_dates=True)
        col_no = 2
        upstream_forecasts_from_model_1 = upstream_forecasts_from_model_1.iloc[:,col_no]
        original_data = upstream_forecasts_from_model_1.copy()
        
        log_trans=[True]
        first_diff=[True]
        model_1_input_raw_data = pd.read_csv(
            'Input/Model 1/with Hours Worked/DesiredModel1Data_6Oct with hours.csv',
            parse_dates=True,
            index_col=0)
        data_bef_forecast = np.array(model_1_input_raw_data.iloc[model_1_input_raw_data.index.get_loc(original_data.index[0]) -1,5])
        original_data = np.array(original_data)
        cpi_transformed = pd.DataFrame(transformation(original_data,
                           log_trans,
                           first_diff,
                           data_bef_forecast))
        pce_transformed.join(cpi_transformed,rsuffix='cpi').to_csv('Input/Model 2/Model 1 with HW Transformed Output for Model 2.csv')
    else:
        pce_transformed.to_csv('Input/Model 2/Model 1 with HW Transformed Output for Model 2.csv')
        
def create_x13_upstream_forecasts_for_Premodel_1(
        initial_training_end_date='2021-10-01'):
    io_gdp = pd.read_csv(
    'Input/PreModel 1/IO Model Input Data.csv',
    index_col=0, parse_dates=True)
    
    res = sm.tsa.x13_arima_analysis(io_gdp['Nominal GDP'])
    
    io_gdp['x13 Nominal GDP'] = res.seasadj
    ##
    io_gdp.to_csv(
        'Input/PreModel 1/test.csv')
    ##
    io_gdp['IO Real GDP'] = io_gdp[
        'x13 Nominal GDP'] / (io_gdp['GDP Deflator']/100)
    
    #res = sm.tsa.x13_arima_analysis(io_gdp['IO Real GDP'])
    
    io_gdp['QoQ Real GDP Growth Rate'] = io_gdp['IO Real GDP'].pct_change(
        ).dropna()
    
    io_gdp['intermediate_growth'] = 1+io_gdp['QoQ Real GDP Growth Rate']
    
    io_gdp['Quarterly Final Real GDP'] = io_gdp['intermediate_growth'].cumprod()*io_gdp.iloc[0,2]  
    
    gdp_data = io_gdp.loc[initial_training_end_date:].copy()
    
    gdp_data.at[initial_training_end_date,'Quarterly Final Real GDP'] = gdp_data.loc[initial_training_end_date,'Quarterly OE Real GDP']
    gdp_data.at[initial_training_end_date,'intermediate_growth'] = 1
    gdp_data['Quarterly Final Real GDP'] = gdp_data['intermediate_growth'].cumprod()*gdp_data.iloc[0,2]
    gdp_data['transformed_real_gdp'] = 100*(np.log(
            gdp_data['Quarterly Final Real GDP']).diff())
    gdp_data['transformed_real_gdp'].dropna().to_csv(
        'Input/PreModel 1/IO Model Real GDP Log first diff.csv')
    
    gdp_data.to_csv('gdp data calculation.csv')
    io_gdp.to_csv('io gdp data calculation.csv')
    
    
def create_upstream_forecasts_for_model_1_test_(initial_training_end_date='2021-10-01'):
    
    gdp_transformed = pd.read_csv(r"Input/PreModel 1/IO Model Real GDP Log first diff.csv",usecols=[1])

    model_1_input_raw_data = pd.read_csv(
            'Input/ARIMAX/DesiredModel1Data_6Oct with hours.csv',
            parse_dates=True,
            index_col=0)
    gdp_data = model_1_input_raw_data.loc[initial_training_end_date:].copy()
    gdp_data['transformed_10YT'] = 100*(np.log(
            gdp_data['10YTreasuryNotesRate']).diff())
    g=gdp_data['transformed_10YT'].dropna().reset_index()
    g.drop(['Date'],axis=1, inplace=True)
    gdp_transformed.join(g).to_csv('Input/Model 1/with Hours Worked/PreModel 1 with 10yt Transformed Output for Model 1.csv')
    
    
def create_upstream_forecasts_for_model_2_test(folder='with Hours Worked',initial_training_end_date='2022-04-01'):
    upstream_forecasts_from_model_1 = pd.read_csv(
    'Output/Model 1/'+folder+'/Adj Forecasts/Real PCE_adj_forecasts_model_1.csv', index_col=0,parse_dates=True)
    col_no = 2
    upstream_forecasts_from_model_1 = upstream_forecasts_from_model_1.iloc[:,col_no]
    original_data = upstream_forecasts_from_model_1.copy()
    
    log_trans=[True]
    first_diff=[True]
    model_1_input_raw_data = pd.read_csv(
        'Input/Model 1/with Hours Worked/DesiredModel1Data_6Oct with hours.csv',
        parse_dates=True,
        index_col=0)
    data_bef_forecast = np.array(model_1_input_raw_data.iloc[model_1_input_raw_data.index.get_loc(original_data.index[0]) -1,1])
    original_data = np.array(original_data)
    pce_transformed = pd.DataFrame(transformation(original_data,
                       log_trans,
                       first_diff,
                       data_bef_forecast))

    upstream_forecasts_from_model_1 = pd.read_csv(
        'Output/Model 1/'+folder+'/Adj Forecasts/Consumer price index_adj_forecasts_model_1.csv', index_col=0,parse_dates=True)
    col_no = 2
    upstream_forecasts_from_model_1 = upstream_forecasts_from_model_1.iloc[:,col_no]
    original_data = upstream_forecasts_from_model_1.copy()
        
    log_trans=[True]
    first_diff=[True]
    model_1_input_raw_data = pd.read_csv(
            'Input/Model 1/with Hours Worked/DesiredModel1Data_6Oct with hours.csv',
            parse_dates=True,
            index_col=0)
    data_bef_forecast = np.array(model_1_input_raw_data.iloc[model_1_input_raw_data.index.get_loc(original_data.index[0]) -1,5])
    original_data = np.array(original_data)
    cpi_transformed = pd.DataFrame(transformation(original_data,
                           log_trans,
                           first_diff,
                           data_bef_forecast))
    final=pce_transformed.join(cpi_transformed,rsuffix='cpi')
    
    model_1_input_raw_data = pd.read_csv(
            'Input/ARIMAX/ON Data LatestData_Collection.csv',
            parse_dates=True,
            index_col=0)
    gdp_data = model_1_input_raw_data.loc[initial_training_end_date:].copy()
    gdp_data['30YFixedMortgageRate']
    gdp_data['transformed_30FMR'] = np.log(
            gdp_data['30YFixedMortgageRate']).diff()
    g=gdp_data['transformed_30FMR'].dropna().reset_index()
    g.drop(['QtrYear'],axis=1, inplace=True)
    final.join(g).to_csv('Input/Model 2/Model 1 with CPI Transformed Output for Model 2.csv')
    
    
    
    
def create_upstream_forecasts_for_model_5_test(folder='with Hours Worked',initial_training_end_date='2021-10-01'):
    upstream_forecasts_from_model_1 = pd.read_csv(
    'Output/Model 1/'+folder+'/Adj Forecasts/Real GDP_adj_forecasts_model_1.csv', index_col=0,parse_dates=True)
    col_no = 2
    upstream_forecasts_from_model_1 = upstream_forecasts_from_model_1.iloc[:,col_no]
    original_data = upstream_forecasts_from_model_1.copy()
    
    log_trans=[True]
    first_diff=[True]
    model_1_input_raw_data = pd.read_csv(
        'Input/Model 1/with Hours Worked/DesiredModel1Data_6Oct with hours.csv',
        parse_dates=True,
        index_col=0)
    col_loc = 0
    data_bef_forecast = np.array(model_1_input_raw_data.iloc[model_1_input_raw_data.index.get_loc(original_data.index[0]) -1,col_loc])
    original_data = np.array(original_data)
    real_gdp_transformed = pd.DataFrame(transformation(original_data,
                       log_trans,
                       first_diff,
                       data_bef_forecast,
                       multiplier=100))
    
    model_1_input_raw_data = pd.read_csv(
            'Input/ARIMAX/Model 5 Desired Model Input Data with Unemployment Rate.csv',
            parse_dates=True,
            index_col=0)
    gdp_data = model_1_input_raw_data.loc[initial_training_end_date:].copy()
    gdp_data['transformed_3MIT'] = 100*(np.log(
            gdp_data['3MInterestRate']).diff())
    g=gdp_data['transformed_3MIT'].dropna().reset_index()
    g.drop(['Date'],axis=1, inplace=True)
    
    real_gdp_transformed.join(g,rsuffix='3mit').to_csv('Input/Model 5/with Hours Worked/Model 1 Transformed Output for Model 5.csv')


def create_upstream_forecasts_for_model_6_test(folder='with Hours Worked'):
    upstream_forecasts_from_model_1 = pd.read_csv(
    'Output/Model 1/'+folder+'/Adj Forecasts/Real PCE_adj_forecasts_model_1.csv', index_col=0,parse_dates=True)
    col_no = 2
    upstream_forecasts_from_model_1 = upstream_forecasts_from_model_1.iloc[:,col_no]
    original_data = upstream_forecasts_from_model_1.copy()
    
    log_trans=[True]
    first_diff=[True]
    model_1_input_raw_data = pd.read_csv(
        'Input/Model 1/'+folder+'/DesiredModel1Data_6Oct with hours.csv',
        parse_dates=True,
        index_col=0)
    data_bef_forecast = np.array(model_1_input_raw_data.iloc[model_1_input_raw_data.index.get_loc(original_data.index[0]) -1,1])
    original_data = np.array(original_data)
    PCE_transformed = pd.DataFrame(transformation(original_data,
                       log_trans,
                       first_diff,
                       data_bef_forecast))
    if folder == 'with Hours Worked':
        upstream_forecasts_from_model_1 = pd.read_csv(
        'Output/Model 1/'+folder+'/Adj Forecasts/Consumer price index_adj_forecasts_model_1.csv', index_col=0,parse_dates=True)
        col_no = 2
        upstream_forecasts_from_model_1 = upstream_forecasts_from_model_1.iloc[:,col_no]
        original_data = upstream_forecasts_from_model_1.copy()
        
        log_trans=[True]
        first_diff=[True]
        model_1_input_raw_data = pd.read_csv(
            'Input/Model 1/with Hours Worked/DesiredModel1Data_6Oct with hours.csv',
            parse_dates=True,
            index_col=0)
        data_bef_forecast = np.array(model_1_input_raw_data.iloc[model_1_input_raw_data.index.get_loc(original_data.index[0]) -1,5])
        original_data = np.array(original_data)
        cpi_transformed = pd.DataFrame(transformation(original_data,
                           log_trans,
                           first_diff,
                           data_bef_forecast))
        PCE_transformed.join(cpi_transformed,rsuffix='CPI').to_csv('Input/Model 6/Model_1_Output_for_Model_6.csv')
    else:
        PCE_transformed.to_csv('Input/Model 6/Model_1_Output_for_Model_6.csv')