# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 10:45:27 2021

@author: Avijit.Nandy
"""

import numpy as np
import pandas as pd

from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.vector_ar.vecm import VECM, select_order
import matplotlib as plt

import aggregation_helper_functions as ahf
import statsmodels.api as sm
import scipy.stats as stats
from statsmodels.stats.stattools import jarque_bera


################### Model Retail Sales 3 below

def model_3_RS(last_iteration=14):
    df = pd.read_csv('Input/Model 3/with Retail Sales/Model3QData-RS.csv', index_col=0,parse_dates=True)
    exog = pd.read_csv('Input/Model 3/with Retail Sales/Exog Workplace_oil Model 3.csv', index_col=0,parse_dates=True)
    #df.index.freq = "Q"
    #exog.index.freq = "Q"
    
    Model3Data = df[['CPI','CCI','Real GDP','CapacityUtilization','Retail Sales','ConsumerCredit']]
    Model3Data.columns = ['lCPI','lCCI','lReal GDP','lCapacityUtilization','lRS','lConsumerCredit']
    
    
    df['lCPI'] = np.log(df['CPI'])
    df['lCCI'] = np.log(df['CCI'])
    df['lReal GDP'] = np.log(df['Real GDP'])
    df['lCapacityUtilization'] = np.log(df['CapacityUtilization'])
    df['lRS'] = np.log(df['Retail Sales'])
    df['lConsumerCredit'] = np.log(df['ConsumerCredit'])
    
    df2 = df.drop(['CPI','CCI','Real GDP','CapacityUtilization','Retail Sales','ConsumerCredit'], axis = 1)
    df2 = df2[['lCPI','lCCI','lReal GDP','lCapacityUtilization','lRS','lConsumerCredit']]
    
    
    
    mod1 = coint_johansen(df2,0,2)
    output = pd.DataFrame([mod1.lr2,mod1.lr1],index=['Max Eigen',"Trace"])
    print('\nCE w/constant intercept\n' ,output.T, '\n')
    print("Critical values(90%, 95% , 99%) of Max Eign\n" , mod1.cvm , '\n')
    print("Critical values(90%, 95% , 99%) of Trace\n" , mod1.cvt, '\n')
    
      
    lo = select_order(df2,maxlags = 7)
    #lo = select_order(df2,maxlags = 7)
    lo.selected_orders
    
    #aic criterion gives 2 lag
    #ar_diff is lag order, coint_rank in no of co integrated reationship
    Model = VECM(df2,exog = exog ,k_ar_diff = 2, coint_rank = 2, deterministic= 'co')
    Model_res = Model.fit()
    
    forecast_extract = pd.DataFrame()
    for i in range(16):
        forecast_extract = forecast_extract.append({'lCPI':Model_res.predict(steps = len(exog), exog_fc = pd.DataFrame(exog))[i][0],
                                                    'lCCI':Model_res.predict(steps = len(exog), exog_fc = pd.DataFrame(exog))[i][1],
                                                    'lReal GDP':Model_res.predict(steps =len(exog), exog_fc = pd.DataFrame(exog ))[i][2],
                                                    'lCapacityUtilization':Model_res.predict(steps = len(exog), exog_fc = pd.DataFrame(exog) )[i][3],
                                                    'lRS':Model_res.predict(steps = len(exog), exog_fc = pd.DataFrame(exog ))[i][4],
                                                    'lConsumerCredit':Model_res.predict(steps = len(exog), exog_fc = pd.DataFrame(exog))[i][5]}
                                                    ,ignore_index=True)
    
    forecasts_trans_df = forecast_extract.head(14).copy()
    forecasts_trans_df = forecasts_trans_df[['lCPI','lCCI','lReal GDP','lCapacityUtilization','lRS','lConsumerCredit']]
    out = forecasts_trans_df.copy()
    
    log_list = [1,1,1,1,1,1]
    for lg in range(len(log_list)):
        if log_list[lg]==1:
            out.iloc[:,lg] = np.exp(out.iloc[:,lg])
    out.to_csv('Output/Model 3/with Retail Sales/ActualforecastModel3.csv')
    initial_forecasts_for_plot = out.copy()
    results = Model_res
    idx = pd.date_range('1980-06-30', periods=len(df2)-1, freq='Q')
    model_1_data = pd.DataFrame(ahf.difftrans(df2))
    model_1_data.set_index(idx, inplace = True)
    model_1_input_raw_data = model_1_data.copy()
    
    
    initial_training_end_date = '2022-04-01'
    
    fcast = df2.tail(1)
    fcast = fcast.append(forecasts_trans_df.copy(),ignore_index=True)
    forecasts_new = ahf.difftrans(fcast)
    
    
    n_steps = len(forecasts_new)
    n_variables = len(forecasts_new.columns)
    
    
    ## Code for adjusting forecasts
    
    
    initial_forecasts = forecasts_new.copy()
    initial_forecasts = np.array(initial_forecasts)
    
    irfs = results.irf(n_steps)
    irfs = irfs.orth_irfs
    
    lag_order = 6
    #lag_order = 3
    #new_forecasts_df_mod2 = pd.read_csv('Input/Model 3/Model_2_Output_for_Model_3.csv', index_col=0)
    new_forecasts_df_mod2 = pd.read_csv('Input/Model 3/Model_1_Output_for_Model_3.csv', index_col=0)
    df = pd.read_csv('Input/Model 3/with Retail Sales/Model3QData-RS.csv', index_col=0,parse_dates=True)
    #upstream_forecasts = pd.DataFrame(new_forecasts_df_mod2.iloc[:,3])
    #upstream_forecasts = np.array(upstream_forecasts)  
    upstream_forecasts = np.array(new_forecasts_df_mod2)  
    
    col_loc_adj_col = [0,2]
    
    bef_forecast_values= pd.DataFrame(model_1_data.tail(lag_order-1)).transpose()
        
    bef_forecast_values = np.array(bef_forecast_values)  
    
    coeffs = None
    new_forecasts = ahf.adjust_forecasts(
                        initial_forecasts,
                        irfs,
                        coeffs,
                        lag_order,
                        upstream_forecasts,
                        col_loc_adj_col,
                        bef_forecast_values,
                        last_iteration
                        )
    new_forecasts_df_mod3 = pd.DataFrame(new_forecasts)
    
    new_forecasts_df_mod3.columns = ['lCPI','lCCI','lReal GDP','lCapacityUtilization','lRS','lConsumerCredit']
    out = pd.DataFrame()
    for tr in range(new_forecasts_df_mod3.shape[0]):
        
        if tr == 0:
            out = df2.tail(1) + new_forecasts_df_mod3.iloc[tr,:]
        else:
            out = out.append(out.tail(1)+new_forecasts_df_mod3.iloc[tr,:], ignore_index = True)
    
    log_list = [1,1,1,1,1,1]
    for lg in range(len(log_list)):
        if log_list[lg]==1:
            out.iloc[:,lg] = np.exp(out.iloc[:,lg])
    
    final_output_Model3 = out
    
    #output_to_model4 = Model3Data.tail(1).append(out).diff(1).iloc[:,3].dropna()
    #output_to_model4.to_csv('Input/Model 4/Output Model 3 RS for Model 4.csv')
    out.to_csv('Output/Model 3/with Retail Sales/Model3AdjOutput.csv')

    new_adj_final_forecasts = out.copy()
    
    oe_forecasts_df = pd.read_csv('Input/Model 3/with Retail Sales/OE forecast data for Model 3.csv',
                                  parse_dates=True,
                                  index_col=0)
    new_adj_final_forecasts.index = oe_forecasts_df.index
    initial_forecasts_for_plot.index = oe_forecasts_df.index
    for column in out.columns:
        temp_dict = dict()
        temp_dict['OE_forecast'] = oe_forecasts_df[column]
        temp_dict['initial_forecast'] = initial_forecasts_for_plot[column]
        temp_dict['new_adj_forecast'] = new_adj_final_forecasts[column]
        ax = pd.DataFrame(temp_dict,index=oe_forecasts_df.index).plot(title=column)
        #ax.set_ylim(0, np.array(pd.DataFrame(temp_dict,index=oe_forecasts_df.index)).max()*1.5)
        #ax.set_ylim(0)
        fig = ax.get_figure()
        fig.savefig('Output/Variations/Variant 1/Model 3 RS '+column+'.png')
        pd.DataFrame(temp_dict,index=oe_forecasts_df.index).to_csv(
            'Output/Model 3/with Retail Sales/Adj Forecasts/'+column+\
                '_adj_forecasts_model_3.csv')
        pd.DataFrame(temp_dict,index=oe_forecasts_df.index).to_csv(
            'Output/Variations/Variant 1/Adj Forecasts/Model 3 RS/'+column+\
                '_adj_forecasts_model_3.csv')

################### Model Vehicle Sales 3 below
def model_3_VS(last_iteration=14):
    df = pd.read_csv('Input/Model 3/with Vehicle Sales/Model3QData-VS.csv', index_col=0,parse_dates=True)
    exog = pd.read_csv('Input/Model 3/with Vehicle Sales/Exog Workplace_oil Model 3.csv', index_col=0,parse_dates=True)
    #df.index.freq = "Q"
    #exog.index.freq = "Q"
    
    Model3Data = df[['CPI','CCI','Real GDP','CapacityUtilization','VS','ConsumerCredit']]
    Model3Data.columns = ['lCPI','lCCI','lReal GDP','lCapacityUtilization','lRS','lConsumerCredit']
    
    
    df['lCPI'] = np.log(df['CPI'])
    df['lCCI'] = np.log(df['CCI'])
    df['lReal GDP'] = np.log(df['Real GDP'])
    df['lCapacityUtilization'] = np.log(df['CapacityUtilization'])
    df['lVS'] = np.log(df['VS'])
    df['lConsumerCredit'] = np.log(df['ConsumerCredit'])
    
    df2 = df.drop(['CPI','CCI','Real GDP','CapacityUtilization','VS','ConsumerCredit'], axis = 1)
    df2 = df2[['lCPI','lCCI','lReal GDP','lCapacityUtilization','lVS','lConsumerCredit']]
    
    
    
    mod1 = coint_johansen(df2,0,2)
    output = pd.DataFrame([mod1.lr2,mod1.lr1],index=['Max Eigen',"Trace"])
    print('\nCE w/constant intercept\n' ,output.T, '\n')
    print("Critical values(90%, 95% , 99%) of Max Eign\n" , mod1.cvm , '\n')
    print("Critical values(90%, 95% , 99%) of Trace\n" , mod1.cvt, '\n')
    
      
    lo = select_order(df2,maxlags = 7)
    #lo = select_order(df2,maxlags = 7)
    lo.selected_orders
    
    #aic criterion gives 2 lag
    #ar_diff is lag order, coint_rank in no of co integrated reationship
    Model = VECM(df2,exog = exog ,k_ar_diff = 1, coint_rank = 2, deterministic= 'co')
    Model_res = Model.fit()
    
     #Granger Causalty test
    ahf.Casualty_test(df2,Model_res).to_csv('Output/Model 3/with Vehicle Sales/CausaltyTestResults.csv')
    
    #H_0: data generated by normally-distributed process
    normtest = Model_res.test_normality()
    print(normtest)
    print(normtest.conclusion)
    #failing normality tests
    
   
      
    for i in range(Model_res.resid.shape[1]):
        fig = sm.qqplot(Model_res.resid[:,i],stats.norm, fit=True, line='45')
        # plt.show()
        print("Results for", i)
        print(jarque_bera(Model_res.resid[:,i]))
        print("----------------------")
    
    #Autocorreleation test
    ahf.AutoCorrtest(Model_res,10).to_csv('Output/Model 3/with Vehicle Sales/AutocorreleationTestResults.csv')
    
    forecast_extract = pd.DataFrame()
    for i in range(14):
        forecast_extract = forecast_extract.append({'lCPI':Model_res.predict(steps = len(exog), exog_fc = pd.DataFrame(exog))[i][0],
                                                    'lCCI':Model_res.predict(steps = len(exog), exog_fc = pd.DataFrame(exog))[i][1],
                                                    'lReal GDP':Model_res.predict(steps =len(exog), exog_fc = pd.DataFrame(exog ))[i][2],
                                                    'lCapacityUtilization':Model_res.predict(steps = len(exog), exog_fc = pd.DataFrame(exog) )[i][3],
                                                    'lVS':Model_res.predict(steps = len(exog), exog_fc = pd.DataFrame(exog ))[i][4],
                                                    'lConsumerCredit':Model_res.predict(steps = len(exog), exog_fc = pd.DataFrame(exog))[i][5]}
                                                    ,ignore_index=True)
    
    forecasts_trans_df = forecast_extract.head(16).copy()
    forecasts_trans_df = forecasts_trans_df[['lCPI','lCCI','lReal GDP','lCapacityUtilization','lVS','lConsumerCredit']]
    out = forecasts_trans_df.copy()
    
    log_list = [1,1,1,1,1,1]
    for lg in range(len(log_list)):
        if log_list[lg]==1:
            out.iloc[:,lg] = np.exp(out.iloc[:,lg])
    out.to_csv('Output/Model 3/with Vehicle Sales/ActualforecastModel3.csv')
    initial_forecasts_for_plot = out.copy()
    results = Model_res
    idx = pd.date_range('1980-04-01', periods=len(df2)-1, freq='Q')
    model_1_data = pd.DataFrame(ahf.difftrans(df2))
    model_1_data.set_index(idx, inplace = True)
    model_1_input_raw_data = model_1_data.copy()
    
    
    initial_training_end_date = '2022-04-01'
    
    fcast = df2.tail(1)
    fcast = fcast.append(forecasts_trans_df.copy(),ignore_index=True)
    forecasts_new = ahf.difftrans(fcast)
    
    
    n_steps = len(forecasts_new)
    n_variables = len(forecasts_new.columns)
    
    
    ## Code for adjusting forecasts
    
    
    initial_forecasts = forecasts_new.copy()
    initial_forecasts = np.array(initial_forecasts)
    
    irfs = results.irf(n_steps)
    irfs = irfs.orth_irfs
    
    lag_order = 6
    #lag_order = 4
    #new_forecasts_df_mod2 = pd.read_csv('Input/Model 3/Model_2_Output_for_Model_3.csv', index_col=0)
    new_forecasts_df_mod2 = pd.read_csv('Input/Model 3/Model_1_Output_for_Model_3.csv', index_col=0)
    df = pd.read_csv('Input/Model 3/with Vehicle Sales/Model3QData-VS.csv', index_col=0,parse_dates=True)
    #upstream_forecasts = pd.DataFrame(new_forecasts_df_mod2.iloc[:,3])
    #upstream_forecasts = np.array(upstream_forecasts)  
    upstream_forecasts = np.array(new_forecasts_df_mod2)
    
    col_loc_adj_col = [0,2]
    
    bef_forecast_values= pd.DataFrame(model_1_data.tail(lag_order-1)).transpose()
        
    bef_forecast_values = np.array(bef_forecast_values)  
    
    coeffs = None
    new_forecasts = ahf.adjust_forecasts(
                        initial_forecasts,
                        irfs,
                        coeffs,
                        lag_order,
                        upstream_forecasts,
                        col_loc_adj_col,
                        bef_forecast_values,
                        last_iteration
                        )
    new_forecasts_df_mod3 = pd.DataFrame(new_forecasts)
    
    new_forecasts_df_mod3.columns = ['lCPI','lCCI','lReal GDP','lCapacityUtilization','lVS','lConsumerCredit']
    out = pd.DataFrame()
    for tr in range(new_forecasts_df_mod3.shape[0]):
        
        if tr == 0:
            out = df2.tail(1) + new_forecasts_df_mod3.iloc[tr,:]
        else:
            out = out.append(out.tail(1)+new_forecasts_df_mod3.iloc[tr,:], ignore_index = True)
    
    log_list = [1,1,1,1,1,1]
    for lg in range(len(log_list)):
        if log_list[lg]==1:
            out.iloc[:,lg] = np.exp(out.iloc[:,lg])
    
    final_output_Model3 = out
    
    output_to_model4 = Model3Data.tail(1).append(out).diff(1).iloc[:,3].dropna()
    output_to_model4.to_csv('Input/Model 4/Output Model 3 VS for Model 4.csv')
    out.to_csv('Output/Model 3/with Vehicle Sales/Model3AdjOutput.csv')

    new_adj_final_forecasts = out.copy()
    
    oe_forecasts_df = pd.read_csv('Input/Model 3/with Vehicle Sales/OE forecast data for Model 3.csv',
                                  parse_dates=True,
                                  index_col=0)
    new_adj_final_forecasts.index = oe_forecasts_df.index
    initial_forecasts_for_plot.index = oe_forecasts_df.index
    for column in out.columns:
        temp_dict = dict()
        temp_dict['OE_forecast'] = oe_forecasts_df[column]
        temp_dict['initial_forecast'] = initial_forecasts_for_plot[column]
        temp_dict['new_adj_forecast'] = new_adj_final_forecasts[column]
        ax = pd.DataFrame(temp_dict,index=oe_forecasts_df.index).plot(title=column)
        #ax.set_ylim(0, np.array(pd.DataFrame(temp_dict,index=oe_forecasts_df.index)).max()*1.5)
        #ax.set_ylim(0)
        fig = ax.get_figure()
        fig.savefig('Output/Variations/Variant 1/Model 3 VS '+column+'.png')
        pd.DataFrame(temp_dict,index=oe_forecasts_df.index).to_csv(
            'Output/Model 3/with Vehicle Sales/Adj Forecasts/'+column+\
                '_adj_forecasts_model_3.csv')
        pd.DataFrame(temp_dict,index=oe_forecasts_df.index).to_csv(
            'Output/Variations/Variant 1/Adj Forecasts/Model 3 VS/'+column+\
                '_adj_forecasts_model_3.csv')
####################### Model 4 below
def model_4_brent(last_iteration=14):
    df = pd.read_csv('Input/Model 4/with Brent/ConsoData-Paper4.csv', index_col=0,parse_dates=True)
    exog = pd.read_csv('Input/Model 4/with Brent/Exog OilP.csv', index_col=0,parse_dates=True)
    df.index.freq = "Q"
    exog.index.freq = "Q"
    
    df.drop(['RealOilPriceWTI'], axis = 1, inplace = True)
    #df.drop(['RealOilPriceBrent'], axis = 1, inplace = True)
    
    #With constant "0", lagorder "2"
    mod1 = coint_johansen(df,0,1)
    output = pd.DataFrame([mod1.lr2,mod1.lr1],index=['Max Eigen',"Trace"])
    print('\nCE w/constant intercept\n' ,output.T, '\n')
    print("Critical values(90%, 95% , 99%) of Max Eign\n" , mod1.cvm , '\n')
    print("Critical values(90%, 95% , 99%) of Trace\n" , mod1.cvt, '\n')
    
      
    lo = select_order(df,maxlags = 7)
    #lo = select_order(df,maxlags = 4)
    lo.selected_orders
    
    #aic criterion gives 2 lag
    #ar_diff is lag order, coint_rank in no of co integrated reationship
    Model = VECM(df,exog = exog,k_ar_diff = 1, coint_rank = 1)
    Model_res = Model.fit()
    
    forecast_extract = pd.DataFrame()
    
    for i in range(14):
        forecast_extract = forecast_extract.append({'ldOilProduction':Model_res.predict(steps = len(exog), exog_fc = exog)[i][0],
                                                    'CapacityUtilisation':Model_res.predict(steps = len(exog), exog_fc = exog)[i][1],
                                                    'RealOilPriceBrent':Model_res.predict(steps = len(exog), exog_fc = exog)[i][2],
                                                    },ignore_index=True)
    forecasts_trans_df = forecast_extract.head(14).copy()
    
    forecast_extract.to_csv('Output/Model 4/with Brent/ActualforecastModel4.csv')
    initial_forecasts_for_plot = forecast_extract.copy()
    results = Model_res
    idx = pd.date_range('1987-09-30', periods=len(df)-1, freq='Q')
    model_1_data = pd.DataFrame(ahf.difftrans(df))
    model_1_data.set_index(idx, inplace = True)
    model_1_input_raw_data = model_1_data.copy()
    
    
    fcast = df.tail(1)
    forecasts_trans_df = forecasts_trans_df[['ldOilProduction','CapacityUtilisation','RealOilPriceBrent']]
    fcast = fcast.append(forecasts_trans_df.copy(),ignore_index=True)
    forecasts_new = ahf.difftrans(fcast)
    
    
    n_steps = len(forecasts_new)
    n_variables = len(forecasts_new.columns)
    
    initial_forecasts = forecasts_new.copy()
    initial_forecasts = np.array(initial_forecasts)
    
    irfs = results.irf(n_steps)
    irfs = irfs.orth_irfs
    
    lag_order = 7
    #lag_order=0
    upstream_forecasts = pd.read_csv('Input/Model 4/Output Model 3 VS for Model 4.csv',
                                     index_col=0)
    #upstream_forecasts = pd.DataFrame(output_to_model4)
    upstream_forecasts = np.array(upstream_forecasts)  
    
    col_loc_adj_col = [1]
    
    bef_forecast_values= pd.DataFrame(model_1_data.tail(lag_order-1)).transpose()
        
    bef_forecast_values = np.array(bef_forecast_values)  
    
    coeffs = None
    new_forecasts = ahf.adjust_forecasts(
                        initial_forecasts,
                        irfs,
                        coeffs,
                        lag_order,
                        upstream_forecasts,
                        col_loc_adj_col,
                        bef_forecast_values,
                        last_iteration
                        )
    new_forecasts_df_mod4 = pd.DataFrame(new_forecasts)
    
    new_forecasts_df_mod4.columns  = ['ldOilProduction','CapacityUtilisation','RealOilPriceBrent']
    out = pd.DataFrame()
    for tr in range(new_forecasts_df_mod4.shape[0]):
        if tr == 0:
            out = df.tail(1) + new_forecasts_df_mod4.iloc[tr,:]
        else:
            out = out.append(out.tail(1)+new_forecasts_df_mod4.iloc[tr,:], ignore_index = True)
    
    final_output_Model4 = out
            
    out.to_csv('Output/Model 4/with Brent/Model4AdjOutput.csv')
    new_adj_final_forecasts = out.copy()
    
    oe_forecasts_df = pd.read_csv('Input/Model 4/with Brent/OE forecast data for Model 4.csv',
                                  parse_dates=True,
                                  index_col=0)
    
    new_adj_final_forecasts.index = oe_forecasts_df.index
    initial_forecasts_for_plot.index = oe_forecasts_df.index
    
    cpi_input_df = pd.read_csv('Input/Model 4/CPI Input.csv',
                                  parse_dates=True,
                                  index_col=0)
    
    for column in out.columns:
        temp_dict = dict()
        if column == 'RealOilPriceBrent':
            temp_dict['OE_forecast'] = oe_forecasts_df[column] * cpi_input_df['CPI']/100
            temp_dict['initial_forecast'] = initial_forecasts_for_plot[column] * cpi_input_df['CPI']/100
            temp_dict['new_adj_forecast'] = new_adj_final_forecasts[column] * cpi_input_df['CPI']/100
            ax = pd.DataFrame(temp_dict,index=oe_forecasts_df.index).plot(title='Nominal Oil Price Brent')
            #ax.set_ylim(0, np.array(pd.DataFrame(temp_dict,index=oe_forecasts_df.index)).max()*1.5)
            #ax.set_ylim(0)
        else:
            temp_dict['OE_forecast'] = oe_forecasts_df[column]
            temp_dict['initial_forecast'] = initial_forecasts_for_plot[column]
            temp_dict['new_adj_forecast'] = new_adj_final_forecasts[column]
            ax = pd.DataFrame(temp_dict,index=oe_forecasts_df.index).plot(title=column)
            #ax.set_ylim(0, np.array(pd.DataFrame(temp_dict,index=oe_forecasts_df.index)).max()*1.5)
            #ax.set_ylim(0)
        fig = ax.get_figure()
        fig.savefig('Output/Variations/Variant 1/Model 4 '+column+'.png')
        pd.DataFrame(temp_dict,index=oe_forecasts_df.index).to_csv(
            'Output/Model 4/with Brent/Adj Forecasts/'+column+\
                '_adj_forecasts_model_4.csv')
        pd.DataFrame(temp_dict,index=oe_forecasts_df.index).to_csv(
            'Output/Variations/Variant 1/Adj Forecasts/Model 4 Brent/'+column+\
                '_adj_forecasts_model_4.csv')

####################### Model 4 WTI
def model_4_WTI(last_iteration=14):
    df = pd.read_csv('Input/Model 4/with WTI/ConsoData-Paper4.csv', index_col=0,parse_dates=True)
    exog = pd.read_csv('Input/Model 4/with WTI/Exog OilP.csv', index_col=0,parse_dates=True)
    df.index.freq = "Q"
    exog.index.freq = "Q"
    
    #df.drop(['RealOilPriceWTI'], axis = 1, inplace = True)
    df.drop(['RealOilPriceBrent'], axis = 1, inplace = True)
    
    #With constant "0", lagorder "2"
    mod1 = coint_johansen(df,0,1)
    output = pd.DataFrame([mod1.lr2,mod1.lr1],index=['Max Eigen',"Trace"])
    print('\nCE w/constant intercept\n' ,output.T, '\n')
    print("Critical values(90%, 95% , 99%) of Max Eign\n" , mod1.cvm , '\n')
    print("Critical values(90%, 95% , 99%) of Trace\n" , mod1.cvt, '\n')
    
      
    lo = select_order(df,maxlags = 7)
    lo.selected_orders
    
    #aic criterion gives 2 lag
    #ar_diff is lag order, coint_rank in no of co integrated reationship
    Model = VECM(df,exog = exog,k_ar_diff = 1, coint_rank = 1)
    Model_res = Model.fit()
    
    forecast_extract = pd.DataFrame()
    
    for i in range(14):
        forecast_extract = forecast_extract.append({'ldOilProduction':Model_res.predict(steps = len(exog), exog_fc = exog)[i][0],
                                                    'CapacityUtilisation':Model_res.predict(steps = len(exog), exog_fc = exog)[i][1],
                                                    'RealOilPriceWTI':Model_res.predict(steps = len(exog), exog_fc = exog)[i][2],
                                                    },ignore_index=True)
    forecasts_trans_df = forecast_extract.head(16).copy()
    
    forecast_extract.to_csv('Output/Model 4/with WTI/ActualforecastModel4.csv')
    initial_forecasts_for_plot = forecast_extract.copy()
    results = Model_res
    idx = pd.date_range('1987-09-30', periods=len(df)-1, freq='Q')
    model_1_data = pd.DataFrame(ahf.difftrans(df))
    model_1_data.set_index(idx, inplace = True)
    model_1_input_raw_data = model_1_data.copy()
    
    
    fcast = df.tail(1)
    forecasts_trans_df = forecasts_trans_df[['ldOilProduction','CapacityUtilisation','RealOilPriceWTI']]
    fcast = fcast.append(forecasts_trans_df.copy(),ignore_index=True)
    forecasts_new = ahf.difftrans(fcast)
    
    
    n_steps = len(forecasts_new)
    n_variables = len(forecasts_new.columns)
    
    initial_forecasts = forecasts_new.copy()
    initial_forecasts = np.array(initial_forecasts)
    
    irfs = results.irf(n_steps)
    irfs = irfs.orth_irfs
    
    lag_order = 7
    #lag_order =1
    upstream_forecasts = pd.read_csv('Input/Model 4/Output Model 3 VS for Model 4.csv',
                                     index_col=0)
    #upstream_forecasts = pd.DataFrame(output_to_model4)
    upstream_forecasts = np.array(upstream_forecasts)  
    
    col_loc_adj_col = [1]
    
    bef_forecast_values= pd.DataFrame(model_1_data.tail(lag_order-1)).transpose()
        
    bef_forecast_values = np.array(bef_forecast_values)  
    
    coeffs = None
    new_forecasts = ahf.adjust_forecasts(
                        initial_forecasts,
                        irfs,
                        coeffs,
                        lag_order,
                        upstream_forecasts,
                        col_loc_adj_col,
                        bef_forecast_values,
                        last_iteration
                        )
    new_forecasts_df_mod4 = pd.DataFrame(new_forecasts)
    
    new_forecasts_df_mod4.columns  = ['ldOilProduction','CapacityUtilisation','RealOilPriceWTI']
    out = pd.DataFrame()
    for tr in range(new_forecasts_df_mod4.shape[0]):
        if tr == 0:
            out = df.tail(1) + new_forecasts_df_mod4.iloc[tr,:]
        else:
            out = out.append(out.tail(1)+new_forecasts_df_mod4.iloc[tr,:], ignore_index = True)
    
    final_output_Model4 = out
            
    out.to_csv('Output/Model 4/with WTI/Model4AdjOutput.csv')
    new_adj_final_forecasts = out.copy()
    
    oe_forecasts_df = pd.read_csv('Input/Model 4/with WTI/OE forecast data for Model 4.csv',
                                  parse_dates=True,
                                  index_col=0)
    
    new_adj_final_forecasts.index = oe_forecasts_df.index
    initial_forecasts_for_plot.index = oe_forecasts_df.index
    
    cpi_input_df = pd.read_csv('Input/Model 4/CPI Input.csv',
                                  parse_dates=True,
                                  index_col=0)
    
    for column in out.columns:
        temp_dict = dict()
        if column == 'RealOilPriceWTI':
            temp_dict['OE_forecast'] = oe_forecasts_df[column] * cpi_input_df['CPI']/100
            temp_dict['initial_forecast'] = initial_forecasts_for_plot[column] * cpi_input_df['CPI']/100
            temp_dict['new_adj_forecast'] = new_adj_final_forecasts[column] * cpi_input_df['CPI']/100
            ax = pd.DataFrame(temp_dict,index=oe_forecasts_df.index).plot(title='Nominal Oil Price WTI')
            #ax.set_ylim(0, np.array(pd.DataFrame(temp_dict,index=oe_forecasts_df.index)).max()*1.5)
            #ax.set_ylim(0)
        else:
            temp_dict['OE_forecast'] = oe_forecasts_df[column]
            temp_dict['initial_forecast'] = initial_forecasts_for_plot[column]
            temp_dict['new_adj_forecast'] = new_adj_final_forecasts[column]
            ax = pd.DataFrame(temp_dict,index=oe_forecasts_df.index).plot(title=column)
            #ax.set_ylim(0, np.array(pd.DataFrame(temp_dict,index=oe_forecasts_df.index)).max()*1.5)
            #ax.set_ylim(0)
        fig = ax.get_figure()
        fig.savefig('Output/Variations/Variant 1/Model 4 WTI '+column+'.png')
        pd.DataFrame(temp_dict,index=oe_forecasts_df.index).to_csv(
            'Output/Model 4/with WTI/Adj Forecasts/'+column+\
                '_adj_forecasts_model_4.csv')
        pd.DataFrame(temp_dict,index=oe_forecasts_df.index).to_csv(
            'Output/Variations/Variant 1/Adj Forecasts/Model 4 WTI/'+column+\
                '_adj_forecasts_model_4.csv')
            
            
def model_6(last_iteration=14):
    df = pd.read_csv('Input/Model 6/Model6 Data.csv', index_col=0,parse_dates=True)
    exog = pd.read_csv('Input/Model 6/Exog Workplace_oil.csv', index_col=0,parse_dates=True)
    #df.index.freq = "Q"
    #exog.index.freq = "Q"
    
    Model6Data = df[['Real_PCE','CPI','IndustrialProduction' ]]
    Model6Data.columns = ['lReal_PCE','lCPI','lIndustrialProduction']
    
    
    df['lReal_PCE'] = np.log(df['Real_PCE'])
    df['lCPI'] = np.log(df['CPI'])
    df['lIndustrialProduction'] = np.log(df['IndustrialProduction'])
    
    
    df2 = df.drop(['Real_PCE','CPI','IndustrialProduction'], axis = 1)
    df2 = df2[['lReal_PCE','lCPI','lIndustrialProduction']]
    
    
    
    mod1 = coint_johansen(df2,0,5)
    output = pd.DataFrame([mod1.lr2,mod1.lr1],index=['Max Eigen',"Trace"])
    print('\nCE w/constant intercept\n' ,output.T, '\n')
    print("Critical values(90%, 95% , 99%) of Max Eign\n" , mod1.cvm , '\n')
    print("Critical values(90%, 95% , 99%) of Trace\n" , mod1.cvt, '\n')
    
      
    lo = select_order(df2,maxlags = 7)
    lo.selected_orders
    
    #aic criterion gives 2 lag
    #ar_diff is lag order, coint_rank in no of co integrated reationship
    Model = VECM(df2,exog = exog ,k_ar_diff = 1, coint_rank = 1, deterministic= 'co')
    Model_res = Model.fit()
    
    
     #aic criterion gives 2 lag
    #ar_diff is lag order, coint_rank in no of co integrated reationship
    # Model = VECM(df2,exog = exog ,k_ar_diff = 3, coint_rank = 1, deterministic= 'co')
    # Model_res = Model.fit()
    
     #Granger Causalty test
    ahf.Casualty_test(df2,Model_res).to_csv('Output/Model 6/CausaltyTestResults.csv')
    
    #H_0: data generated by normally-distributed process
    normtest = Model_res.test_normality()
    print(normtest)
    print(normtest.conclusion)
    #failing normality tests
    
   
      
    for i in range(Model_res.resid.shape[1]):
        fig = sm.qqplot(Model_res.resid[:,i],stats.norm, fit=True, line='45')
        # plt.show()
        print("Results for", i)
        print(jarque_bera(Model_res.resid[:,i]))
        print("----------------------")
    
    #Autocorreleation test
    ahf.AutoCorrtest(Model_res,10).to_csv('Output/Model 6/AutocorreleationTestResults.csv')
    
    
    forecast_extract = pd.DataFrame()
    for i in range(14):
        forecast_extract = forecast_extract.append({'lReal_PCE':Model_res.predict(steps = len(exog), exog_fc = pd.DataFrame(exog))[i][0],
                                                    'lCPI':Model_res.predict(steps = len(exog), exog_fc = pd.DataFrame(exog))[i][1],
                                                    'lIndustrialProduction':Model_res.predict(steps =len(exog), exog_fc = pd.DataFrame(exog ))[i][2],
                                                     }
                                                    ,ignore_index=True)
                                                    
                                                    
    
    forecasts_trans_df = forecast_extract.head(16).copy()
    forecasts_trans_df = forecasts_trans_df[['lReal_PCE' ,'lCPI','lIndustrialProduction']]
    out = forecasts_trans_df.copy()
    
    log_list = [1,1,1]
    for lg in range(len(log_list)):
        if log_list[lg]==1:
            out.iloc[:,lg] = np.exp(out.iloc[:,lg])
    out.to_csv('Output/Model 6/ActualforecastModel6.csv')
    initial_forecasts_for_plot = out.copy()
    results = Model_res
    idx = pd.date_range('1980-06-30', periods=len(df2)-1, freq='Q')
    model_1_data = pd.DataFrame(ahf.difftrans(df2))
    model_1_data.set_index(idx, inplace = True)
    model_1_input_raw_data = model_1_data.copy()
    
    
    initial_training_end_date = '2021-12-31'
    
    fcast = df2.tail(1)
    fcast = fcast.append(forecasts_trans_df.copy(),ignore_index=True)
    forecasts_new = ahf.difftrans(fcast)
    
    
    n_steps = len(forecasts_new)
    n_variables = len(forecasts_new.columns)
    
    
    ## Code for adjusting forecasts
    
    
    initial_forecasts = forecasts_new.copy()
    initial_forecasts = np.array(initial_forecasts)
    
    irfs = results.irf(n_steps)
    irfs = irfs.orth_irfs
    
    lag_order = 4
    #new_forecasts_df_mod2 = pd.read_csv('Input/Model 3/Model_2_Output_for_Model_3.csv', index_col=0)
    new_forecasts_df_mod2 = pd.read_csv('Input/Model 6/Model_1_Output_for_Model_6.csv', index_col=0)
    df = pd.read_csv('Input/Model 6/Model6 Data.csv', index_col=0,parse_dates=True)
    #upstream_forecasts = pd.DataFrame(new_forecasts_df_mod2.iloc[:,3])
    #upstream_forecasts = np.array(upstream_forecasts)  
    upstream_forecasts = np.array(new_forecasts_df_mod2)  
    
    col_loc_adj_col = [0,1]
    
    bef_forecast_values= pd.DataFrame(model_1_data.tail(lag_order-1)).transpose()
        
    bef_forecast_values = np.array(bef_forecast_values)  
    
    coeffs = None
    new_forecasts = ahf.adjust_forecasts(
                        initial_forecasts,
                        irfs,
                        coeffs,
                        lag_order,
                        upstream_forecasts,
                        col_loc_adj_col,
                        bef_forecast_values,
                        last_iteration
                        )
    new_forecasts_df_mod3 = pd.DataFrame(new_forecasts)
    
    new_forecasts_df_mod3.columns = ['lReal_PCE' ,'lCPI', 'lIndustrialProduction']
    out = pd.DataFrame()
    for tr in range(new_forecasts_df_mod3.shape[0]):
        
        if tr == 0:
            out = df2.tail(1) + new_forecasts_df_mod3.iloc[tr,:]
        else:
            out = out.append(out.tail(1)+new_forecasts_df_mod3.iloc[tr,:], ignore_index = True)
    
    log_list = [1,1,1]
    for lg in range(len(log_list)):
        if log_list[lg]==1:
            out.iloc[:,lg] = np.exp(out.iloc[:,lg])
    
    final_output_Model3 = out
    
    # output_to_model4 = Model6Data.tail(1).append(out).diff(1).iloc[:,3].dropna()
    # output_to_model4.to_csv('Input/Model 4/Output Model 3 RS for Model 4.csv')
    out.to_csv('Output/Model 6/Model6AdjOutput.csv')

    new_adj_final_forecasts = out.copy()
    
    out.columns = ['Real_PCE' ,'CPI', 'IndustrialProduction']
    initial_forecasts_for_plot.columns = ['Real_PCE' ,'CPI', 'IndustrialProduction']
    new_adj_final_forecasts.columns = ['Real_PCE' ,'CPI', 'IndustrialProduction']
    oe_forecasts_df = pd.read_csv('Input/Model 6/OE Forecasts for Model6.csv',
                                  parse_dates=True,
                                  index_col=0)
    new_adj_final_forecasts.index = oe_forecasts_df.index
    initial_forecasts_for_plot.index = oe_forecasts_df.index
    for column in out.columns:
        temp_dict = dict()
        temp_dict['OE_forecast'] = oe_forecasts_df[column]
        temp_dict['initial_forecast'] = initial_forecasts_for_plot[column]
        temp_dict['new_adj_forecast'] = new_adj_final_forecasts[column]
        ax = pd.DataFrame(temp_dict,index=oe_forecasts_df.index).plot(title=column)
        #ax.set_ylim(0, np.array(pd.DataFrame(temp_dict,index=oe_forecasts_df.index)).max()*1.5)
        #ax.set_ylim(0)
        fig = ax.get_figure()
        fig.savefig('Output/Variations/Variant 1/Model 6  '+column+'.png')
        pd.DataFrame(temp_dict,index=oe_forecasts_df.index).to_csv(
            'Output/Model 6/Adj Forecasts/'+column+\
                '_adj_forecasts_model_6.csv')
        pd.DataFrame(temp_dict,index=oe_forecasts_df.index).to_csv(
            'Output/Variations/Variant 1/Adj Forecasts/Model 6/'+column+\
                '_adj_forecasts_model_6.csv')
            
            


###############
def model_HW_2(last_iteration=14):

    df = pd.read_csv('Input/Model 2/ON Data LatestData_Collection.csv', index_col=0,parse_dates=True)
    exog = pd.read_csv('Input/Model 2/Exog Workplace_oil.csv', index_col=0,parse_dates=True)
    #df.index.freq = "Q"
    #exog.index.freq = "Q"
    
    
    
    df['lConsumption'] = np.log(df['Consumption'])
    df['lNominalMortgageDebt'] = np.log(df['NominalMortgageDebt'])
    df['lCREPriceIndex'] = np.log(df['CREPriceIndex'])
    df['lHPI'] = np.log(df['HPI'])
    df['lHousingStarts'] = np.log(df['HousingStarts'])
    df['l30YFixedMortgageRate'] = np.log(df['30YFixedMortgageRate'])
    df['lResidentialInvestment'] = np.log(df['ResidentialInvestment'])
   
    
    df2 = df.drop(['Consumption','NominalMortgageDebt','CREPriceIndex','HPI','HousingStarts','30YFixedMortgageRate','ResidentialInvestment'], axis = 1)
    df2 = df2[['lConsumption','lNominalMortgageDebt','lCREPriceIndex','lHPI','lHousingStarts','l30YFixedMortgageRate','lResidentialInvestment']]
    df2.columns
    #   
    #With constant "0", lagorder "2"
    mod1 = coint_johansen(df2,0,1)
    output = pd.DataFrame([mod1.lr2,mod1.lr1],index=['Max Eigen',"Trace"])
    print('\nCE w/constant intercept\n' ,output.T, '\n')
    print("Critical values(90%, 95% , 99%) of Max Eign\n" , mod1.cvm , '\n')
    print("Critical values(90%, 95% , 99%) of Trace\n" , mod1.cvt, '\n')
    
      
    lo = select_order(df2,maxlags = 7)
    lo.selected_orders
    
    #aic criterion gives 2 lag
    #ar_diff is lag order, coint_rank in no of co integrated reationship
    Model = VECM(df2,exog = exog,k_ar_diff = 1, coint_rank = 2, deterministic= 'co')
    Model_res = Model.fit()
    
    #Granger Causalty test
    ahf.Casualty_test(df2,Model_res).to_csv('Output/Model 2/with Hours Worked/CausaltyTestResults.csv')
    
    #H_0: data generated by normally-distributed process
    normtest = Model_res.test_normality()
    print(normtest)
    print(normtest.conclusion)
    #failing normality tests
    
    #Autocorreleation test
    ahf.AutoCorrtest(Model_res,10).to_csv('Output/Model 2/with Hours Worked/AutocorreleationTestResults.csv')
    
    #Forecast
    forecast_extract = pd.DataFrame()
    for i in range(14):
        forecast_extract = forecast_extract.append({'lConsumption':Model_res.predict(steps = len(exog), exog_fc = exog)[i][0],
                                                    'lNominalMortgageDebt':Model_res.predict(steps = len(exog), exog_fc = exog)[i][1],
                                                    'lCREPriceIndex':Model_res.predict(steps = len(exog), exog_fc = exog)[i][2],
                                                    'lHPI':Model_res.predict(steps = len(exog), exog_fc = exog)[i][3],
                                                    'lHousingStarts':Model_res.predict(steps = len(exog), exog_fc = exog)[i][4],
                                                    'l30YFixedMortgageRate':Model_res.predict(steps = len(exog), exog_fc = exog)[i][5],
                                                    'lResidentialInvestment':Model_res.predict(steps = len(exog), exog_fc = exog)[i][6]},ignore_index=True)
    
    
    
    forecasts_trans_df = forecast_extract.head(14).copy()
    forecasts_trans_df = forecasts_trans_df[['lConsumption','lNominalMortgageDebt','lCREPriceIndex','lHPI','lHousingStarts','l30YFixedMortgageRate','lResidentialInvestment']]
    out = forecasts_trans_df.copy()
    
    log_list = [1,1,1,1,1,1,1]
    for lg in range(len(log_list)):
        if log_list[lg]==1:
            out.iloc[:,lg] = np.exp(out.iloc[:,lg])
    #out['30YFixedMortgageRate']
    out.to_csv('Output/Model 2/with Hours Worked/ActualforecastModel2.csv')
    
    initial_forecasts_for_plot = out.copy()
    
    results = Model_res
    idx = pd.date_range('1983-06-30', periods=len(df2)-1, freq='Q')
    model_1_data = pd.DataFrame(ahf.difftrans(df2))
    model_1_data.set_index(idx, inplace = True)
    model_1_input_raw_data = model_1_data.copy()
    #model_1_input_raw_data.to_csv('Output/Model 2/with Hours Worked/---ActualforecastModel2.csv')
    fcast = df2.tail(1)
    fcast = fcast.append(forecasts_trans_df.copy(),ignore_index=True)
    forecasts_new = ahf.difftrans(fcast)
    forecasts_new.columns
    
    n_steps = len(forecasts_new)
    n_variables = len(forecasts_new.columns)
    
    n_steps = len(forecasts_new)
    n_variables = len(forecasts_new.columns)
    
    ## Code for adjusting forecasts
    
    initial_forecasts = forecasts_new.copy()
    initial_forecasts = np.array(initial_forecasts)
    
    irfs = results.irf(n_steps)
    irfs = irfs.orth_irfs
    
    lag_order = 1
    upstream_forecasts = pd.read_csv(
        'Input/Model 2/Model 1 with CPI Transformed Output for Model 2.csv',
        index_col=0
        )
    
   
    #up_fr['30YFixedMortgageRate']
   # upstream_forecasts = pd.read_csv(
        #'Model 1 upstream forecasts.csv',
        #index_col=0, parse_dates=True)
    upstream_forecasts = np.array(upstream_forecasts)  
    
    col_loc_adj_col = [0,2]
    
    bef_forecast_values= pd.DataFrame(model_1_data.tail(lag_order-1)).transpose()
        
    bef_forecast_values = np.array(bef_forecast_values)  
    
    coeffs = None
    
    new_forecasts = ahf.adjust_forecasts(
                        initial_forecasts,
                        irfs,
                        coeffs,
                        lag_order,
                        upstream_forecasts,
                        col_loc_adj_col,
                        bef_forecast_values,
                        last_iteration
                        )
    new_forecasts_df_mod2 = pd.DataFrame(new_forecasts)
     
    new_forecasts_df_mod2.columns = ['lConsumption','lNominalMortgageDebt','lCREPriceIndex','lHPI','lHousingStarts','l30YFixedMortgageRate','lResidentialInvestment']
    
    out = pd.DataFrame()
    for tr in range(new_forecasts_df_mod2.shape[0]):
        
        if tr == 0:
            out = df2.tail(1) + new_forecasts_df_mod2.iloc[tr,:]
        else:
            out = out.append(out.tail(1)+new_forecasts_df_mod2.iloc[tr,:], ignore_index = True)
    log_list = [1,1,1,1,1,1,1]
    for lg in range(len(log_list)):
        if log_list[lg]==1:
            out.iloc[:,lg] = np.exp(out.iloc[:,lg])
    #out['30YFixedMortgageRate']= up_fr['30YFixedMortgageRate']
    final_output_Model2 = out
    out.to_csv('Output/Model 2/with Hours Worked/Model2AdjOutput.csv')
    new_adj_final_forecasts = out.copy()
    
    new_forecasts_df_mod2.to_csv('Input/Model 3/Model_2_Output_for_Model_3.csv')
    
    oe_forecasts_df = pd.read_csv('Input/Model 2/OE forecast data for Model 2.csv',
                                  parse_dates=True,
                                  index_col=0)
    new_adj_final_forecasts.index = oe_forecasts_df.index
    initial_forecasts_for_plot.index = oe_forecasts_df.index
    for column in out.columns:
        temp_dict = dict()
        temp_dict['OE_forecast'] = oe_forecasts_df[column]
        temp_dict['initial_forecast'] = initial_forecasts_for_plot[column]
        temp_dict['new_adj_forecast'] = new_adj_final_forecasts[column]
        ax = pd.DataFrame(temp_dict,index=oe_forecasts_df.index).plot(title=column)
        #ax.set_ylim(0, np.array(pd.DataFrame(temp_dict,index=oe_forecasts_df.index)).max()*1.5)
        fig = ax.get_figure()
        fig.savefig('Output/Variations/Variant 1/Model 2 with Hours Worked '+column+'.png')
        pd.DataFrame(temp_dict,index=oe_forecasts_df.index).to_csv(
            'Output/Model 2/with Hours Worked/Adj Forecasts/'+column+\
                '_adj_forecasts_model_2.csv')
        pd.DataFrame(temp_dict,index=oe_forecasts_df.index).to_csv(
            'Output/Variations/Variant 1/Adj Forecasts/Model 2/'+column+\
                '_adj_forecasts_model_2.csv')
###############

