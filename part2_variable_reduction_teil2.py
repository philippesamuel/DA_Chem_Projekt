#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 17:03:47 2020

@author: elron
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas_profiling
import re 
import pandas_profiling


### rabitat approch, remove all correlated variables
for i in listCorrelated:
    df.drop(i, axis = 1, inplace = True)
VarbNames = list(df.columns)

df.to_pickle("process_data_after_remove_variabel_remain_117.pkl")

df = pd.read_pickle("process_data_after_remove_variabel_remain_117.pkl")

# to drop columns
#'p_effective_area_per_leaf_m2', 'p_effective_pressure_bar', 'p_product_full_name','p_product_type', 'pa_beschichtete_rollenlange_m','pp_actual_product'
# 'pp_actual_product_short_name','pp_plan_ausbeute_elemente', 'pp_plan_product', 'pp_product_short_name'
# 'qc_p_position','roll_position','winding_product_short_name','winding_product_type'

df.drop(['p_effective_area_per_leaf_m2', 'p_effective_pressure_bar', 'p_product_full_name','p_product_type', 'pa_beschichtete_rollenlange_m','pp_actual_product'], axis = 1, inplace = True)
df.drop(['pp_actual_product_short_name','pp_plan_ausbeute_elemente', 'pp_plan_product', 'pp_product_short_name'], axis = 1, inplace = True)
df.drop(['qc_p_position','roll_position','winding_product_short_name','winding_product_type'], axis = 1, inplace = True)
df.to_pickle("process_data_after_remove_variabel_remain_103.pkl")
varbList = list(df.columns)


# set qc_seriennummer as index after drop duplicates (825) and NAs (17) 
df = pd.read_pickle("process_data_after_remove_variabel_remain_103.pkl")
df = df.dropna(subset = ['qc_serien_nummer'])
df = df.drop_duplicates(subset = ['qc_serien_nummer'])
#pd.isna(df['p_product_group']).sum()

df = df.set_index('qc_serien_nummer')
df.to_pickle("process_data_after_remove_variabel_remain_102_qc_serien_nummer_index.pkl")



#ps_c_losung_wt_% 310 zu 31 ändern
#ps_dicke_bs 1343 zu 134
# ps_gap_micro_m check if only for brackwasser /seewasser
#qc_durchminimalersalzrueckhalt


df = pd.read_pickle("process_data_after_remove_variabel_remain_102_qc_serien_nummer_index.pkl")


# Transform pa_staub-sauger_1_vor_aminbad_0_aus_>0_an into boolean just 100 and zero as values with greater 
# pa_staub-sauger_2_nach_aminbad, pa_staub-sauger_3_zw5_vor_hw2, pa_staub-sauger_5
# ps_c_losung_wt_% transform 310 into 31 because w % over makes not much sense

staub = list(df.filter(regex = 'staub-sauger').columns)
for i in staub:
    df[i] = df[i].map({100: False, 0: True})

bools = list(df.select_dtypes(bool).columns)

df['ps_c_losung_wt_%'] = df['ps_c_losung_wt_%'].map({310 : 31, 33 : 33, 31 : 31})
df['ps_c_losung_wt_%'].unique()

df['ps_dicke_bs'] = df['ps_dicke_bs'].replace([1343],134)

## Remove not important categorical variables

# not important categorical variables for pca ?
# p_product, p_product_group, p_Product_type_group
# qc_durchminimalersalzrueckhalt
# qc_lasttest
#

dropList = list(df.filter(regex = 'p_product').columns)+list(df.filter(regex = 'Product').columns)
dropList.append('qc_durchminimalersalzrueckhalt')
dropList.append('qc_lasttest')

df1 = df.copy()

for j in dropList:
    df1.drop(j, axis = 1, inplace = True)
    

X = df1.copy()

# Transforming Categorical variables : # sc_d_ergebnis_anz_max_ok, sc_d_ergebnis_nio, sc_d_links_anz_min, sc_d_parameter_anz_werte, # maybe ps_c_losung_wt_%, ps_gap_micro_m,  I take them as numerical
# pa_bahngeschwindigkeit has only three values but i would count as numerical
# categorical variables to encode
# pp_actual_usage, winding_product_line
lb_make = LabelEncoder()
X['pp_actual_usage_code'] = lb_make.fit_transform(df['pp_actual_usage'])
X['winding_product_line_code'] = lb_make.fit_transform(df['winding_product_line'])

X[['winding_product_line_code','pp_actual_usage_code']].head(11)
# drop of categorical variables since I had them decoded
X.drop(['pp_actual_usage','winding_product_line'], axis = 1, inplace = True)
# drop pa_tmc_gehalt_% since it has more than 64 % NA Values and normalisation doesnt function with NA values
X.drop(['pa_tmc_gehalt_%'], axis = 1, inplace = True)
# change boolean into True = 1 and False = 0
boolsX = list(X.select_dtypes(bool).columns)
for i in boolsX:
    X[i] = X[i].map({True: 1, False: 0})


X.to_pickle("process_data_after_remove_variabel_remain_96.pkl")



















 
