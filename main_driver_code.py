# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 23:17:13 2021

@author: gaurav.tibude
"""

import model_1_code as model_1
import PreModel_1_code as Premodel_1
import Model_5_code as model_5
import VECMFinalCode as VECM_models
import aggregation_helper_functions as ahf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")



global_iteration_parameter=14

#ahf.create_x13_upstream_forecasts_for_Premodel_1(
    #initial_training_end_date='2021-10-01')

#Premodel_1.Premodel_1(last_iteration=global_iteration_parameter,
                          #recomputation=False)

#ahf.create_upstream_forecasts_for_model_1(
   #initial_training_end_date='2021-04-01')

ahf.create_x13_upstream_forecasts_for_model_1(
    initial_training_end_date='2022-04-01')

model_1.model_1_with_HoursWorked(last_iteration=global_iteration_parameter,
                          recomputation=False)


#ahf.create_upstream_forecasts_for_model_2(folder='with Hours Worked',initial_training_end_date='2022-04-01')
##ahf.create_upstream_forecasts_for_model_2_test(folder='with Hours Worked',initial_training_end_date='2022-04-01')
# End of comment above on 30 Sep
#VECM_models.model_HW_2(last_iteration=global_iteration_parameter)

ahf.create_upstream_forecasts_for_model_3(folder='with Hours Worked')
                                         
VECM_models.model_3_VS(last_iteration=global_iteration_parameter)
VECM_models.model_3_RS(last_iteration=global_iteration_parameter)

VECM_models.model_4_brent(last_iteration=global_iteration_parameter)

VECM_models.model_4_WTI(last_iteration=global_iteration_parameter)

#model_5.model_5_with_HW(last_iteration=global_iteration_parameter,output_folder='Variant 1'
                                   #,recomputation=False)

ahf.create_upstream_forecasts_for_model_6(folder='with Hours Worked')
VECM_models.model_6(last_iteration=global_iteration_parameter)


