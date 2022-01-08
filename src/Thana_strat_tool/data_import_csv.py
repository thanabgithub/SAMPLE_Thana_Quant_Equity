# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 12:51:31 2020

@author: ADMIN
"""



import pandas as pd
import talib as ta
import numpy as np
from datetime import datetime
import statsmodels.api as sm
from statsmodels import regression
import timeit



def df_pivot(price_df_: pd.DataFrame, target_var: str, index: str, start_date_, end_date_):
    output_df = price_df_[[index, 'Ticker', target_var]]
    output_df = output_df.pivot_table(values=target_var, index=index, columns="Ticker")
    # print(output_df)
    if '0SET' in output_df.columns:
        output_df.drop('0SET', axis = 1, inplace = True)
    if 'Unnamed: 0' in output_df.columns:
        output_df.drop('Unnamed: 0', axis = 1, inplace = True)

    output_df.index = pd.DatetimeIndex(output_df.index)
    output_df = output_df[start_date_:end_date_]
    return output_df.astype(np.float64)

def df_index_clean(input_df_: pd.DataFrame, start_date_, end_date_):
    output_df = input_df_.copy()
    output_df.drop('0SET', axis = 1, inplace = True)
    output_df.index = pd.DatetimeIndex(output_df.index)
    output_df = output_df[start_date_:end_date_]
    return output_df    



def clean_price(input_df: pd.DataFrame):
    input_df.rename(columns = {'Date/Time':'datetime'}, inplace = True)  
    last_dummy_col = 'Unnamed: ' + str(len(input_df.columns)-1)
    try:
        input_df.drop(last_dummy_col, axis = 1, inplace = True)
    except:
        print('no dummy')
        
def clean_indices(input_df: pd.DataFrame, start_date_, end_date_):
    preproc_ = input_df.copy()
    clean_price(preproc_)
    preproc_.set_index('datetime', drop = True,inplace = True)
    preproc_.drop('Ticker', axis = 1, inplace = True)
    
    preproc_.index = pd.DatetimeIndex(preproc_.index)
    preproc_ = preproc_[start_date_:end_date_]
    return preproc_.astype(np.float64)  
    
def get_price(input_df: pd.DataFrame, start_date_, end_date_):
    print('getting prices ...')
    preproc_ = input_df.copy()
    clean_price(preproc_)
    # fillna(0) otherwise it will be fucked up when we use talib
    print('start cal o')
    o_ = df_pivot(preproc_, 'open', 'datetime', start_date_, end_date_).round(2).astype(np.float64)
    print('start cal h')
    h_ = df_pivot(preproc_, 'high', 'datetime', start_date_, end_date_).round(2).astype(np.float64)
    print('start cal l')
    l_ = df_pivot(preproc_, 'low', 'datetime', start_date_, end_date_).round(2).astype(np.float64)
    print('start cal c')
    c_ = df_pivot(preproc_, 'close', 'datetime', start_date_, end_date_).round(2).astype(np.float64)
    print('start cal v')
    v_ = df_pivot(preproc_, 'volume', 'datetime', start_date_, end_date_).round(2).astype(np.float64)
    
    try:
        print('start cal value')
        value_ = df_pivot(preproc_, 'value', 'datetime', start_date_, end_date_).round(2).astype(np.float64)    
        print('start cal vwap')
        vwap_ = (value_/v_).astype(np.float64)  
    except:
        print('start cal vwap')
        vwap_ = (o_ + h_/2.0 + l_/2.0 + c_)/3.0
    return o_, h_, l_, c_, v_, vwap_


def get_price_value_MKC(input_df: pd.DataFrame, start_date_, end_date_):
    print('getting prices ...')
    preproc_ = input_df.copy()
    clean_price(preproc_)
    # fillna(0) otherwise it will be fucked up when we use talib
    print('start cal o')
    o_ = df_pivot(preproc_, 'open', 'datetime', start_date_, end_date_).round(2).astype(np.float64)
    print('start cal h')
    h_ = df_pivot(preproc_, 'high', 'datetime', start_date_, end_date_).round(2).astype(np.float64)
    print('start cal l')
    l_ = df_pivot(preproc_, 'low', 'datetime', start_date_, end_date_).round(2).astype(np.float64)
    print('start cal c')
    c_ = df_pivot(preproc_, 'close', 'datetime', start_date_, end_date_).round(2).astype(np.float64)
    print('start cal v')
    v_ = df_pivot(preproc_, 'volume', 'datetime', start_date_, end_date_).round(2).astype(np.float64)
    print('start cal value')    
    value_ = df_pivot(preproc_, 'value', 'datetime', start_date_, end_date_).round(2).astype(np.float64)    
    print('start cal market cap')    
    MKC_ = df_pivot(preproc_, 'MKC', 'datetime', start_date_, end_date_).round(2).astype(np.float64)    
    return o_, h_, l_, c_, v_, value_, MKC_

def get_benchmark_as_industry_col(input_df: pd.DataFrame, start_date_, end_date_, industry_col_):
    print('getting indices ...')
    indices_df = clean_indices(input_df, start_date_, end_date_).astype(np.float64)
    
    SET_ = pd.DataFrame(index = indices_df.index, columns = industry_col_.columns)
    SET_ = SET_.apply(lambda col: indices_df['SET'].values).ffill(limit = 20).fillna(0)
    
    SET50_ = pd.DataFrame(index = indices_df.index, columns = industry_col_.columns)
    SET50_ = SET50_.apply(lambda col: indices_df['SET50'].values).ffill(limit = 20).fillna(0)    
    
    SET100_ = pd.DataFrame(index = indices_df.index, columns = industry_col_.columns)
    SET100_ = SET100_.apply(lambda col: indices_df['SET100'].values).ffill(limit = 20).fillna(0)
    
    return SET_, SET50_, SET100_    
    
    
def get_benchmark_v_as_industry_col(input_df: pd.DataFrame, start_date_, end_date_, industry_col_):
    print('getting indices vol ...')
    indices_df = clean_indices(input_df, start_date_, end_date_).astype(np.float64)
    
    SET_v_ = pd.DataFrame(index = indices_df.index, columns = industry_col_.columns)
    SET_v_ = SET_v_.apply(lambda col: indices_df['SET_v'].values).ffill(limit = 20).fillna(0)
    
    SET50_v_ = pd.DataFrame(index = indices_df.index, columns = industry_col_.columns)
    SET50_v_ = SET50_v_.apply(lambda col: indices_df['SET50_v'].values).ffill(limit = 20).fillna(0)    
    
    SET100_v_ = pd.DataFrame(index = indices_df.index, columns = industry_col_.columns)
    SET100_v_ = SET100_v_.apply(lambda col: indices_df['SET100_v'].values).ffill(limit = 20).fillna(0)
    
    return SET_v_, SET50_v_, SET100_v_ 
    
def get_indices(input_df: pd.DataFrame, start_date_, end_date_, o_):
    print('getting indices ...')
    indices_df = clean_indices(input_df, start_date_, end_date_).astype(np.float64)
    
    SET_ = pd.DataFrame(index = indices_df.index, columns = o_.columns)
    SET_ = SET_.apply(lambda col: indices_df['SET'].values).ffill(limit = 20).fillna(0).astype(np.float64)
    
    SET50_ = pd.DataFrame(index = indices_df.index, columns = o_.columns)
    SET50_ = SET50_.apply(lambda col: indices_df['SET50'].values).ffill(limit = 20).fillna(0).astype(np.float64)    
    
    SET100_ = pd.DataFrame(index = indices_df.index, columns = o_.columns)
    SET100_ = SET100_.apply(lambda col: indices_df['SET100'].values).ffill(limit = 20).fillna(0).astype(np.float64)
    
    I_AGRO_ = pd.DataFrame(index = indices_df.index, columns = o_.columns)
    I_AGRO_ = I_AGRO_.apply(lambda col: indices_df['I_AGRO'].values).ffill(limit = 20).fillna(0).astype(np.float64)    

    I_CONSUMP_ = pd.DataFrame(index = indices_df.index, columns = o_.columns)
    I_CONSUMP_ = I_CONSUMP_.apply(lambda col: indices_df['I_CONSUMP'].values).ffill(limit = 20).fillna(0).astype(np.float64)
    
    I_FINCIAL_ = pd.DataFrame(index = indices_df.index, columns = o_.columns)
    I_FINCIAL_ = I_FINCIAL_.apply(lambda col: indices_df['I_FINCIAL'].values).ffill(limit = 20).fillna(0).astype(np.float64)    
    
    I_INDUS_ = pd.DataFrame(index = indices_df.index, columns = o_.columns)
    I_INDUS_ = I_INDUS_.apply(lambda col: indices_df['I_INDUS'].values).ffill(limit = 20).fillna(0).astype(np.float64)
    
    I_PROPCON_ = pd.DataFrame(index = indices_df.index, columns = o_.columns)
    I_PROPCON_ = I_PROPCON_.apply(lambda col: indices_df['I_PROPCON'].values).ffill(limit = 20).fillna(0).astype(np.float64)        
    
    I_RESOURC_ = pd.DataFrame(index = indices_df.index, columns = o_.columns)
    I_RESOURC_ = I_RESOURC_.apply(lambda col: indices_df['I_RESOURC'].values).ffill(limit = 20).fillna(0).astype(np.float64)    
    
    I_TECH_ = pd.DataFrame(index = indices_df.index, columns = o_.columns)
    I_TECH_ = I_TECH_.apply(lambda col: indices_df['I_TECH'].values).ffill(limit = 20).fillna(0).astype(np.float64)
    
    I_SERVICE_ = pd.DataFrame(index = indices_df.index, columns = o_.columns)
    I_SERVICE_ = I_SERVICE_.apply(lambda col: indices_df['I_SERVICE'].values).ffill(limit = 20).fillna(0).astype(np.float64)        
    
    return SET_, SET50_, SET100_, I_AGRO_, I_CONSUMP_, I_FINCIAL_, I_INDUS_, I_PROPCON_, I_RESOURC_, I_TECH_, I_SERVICE_

def get_indices_v(input_df: pd.DataFrame, start_date_, end_date_, o_):
    print('getting indices_v ...')
    indices_df = clean_indices(input_df, start_date_, end_date_).astype(np.float64)
    
    SET_v_ = pd.DataFrame(index = indices_df.index, columns = o_.columns)
    SET_v_ = SET_v_.apply(lambda col: indices_df['SET_v'].values).ffill(limit = 20).fillna(0)
    
    SET50_v_ = pd.DataFrame(index = indices_df.index, columns = o_.columns)
    SET50_v_ = SET50_v_.apply(lambda col: indices_df['SET50_v'].values).ffill(limit = 20).fillna(0)    
    
    SET100_v_ = pd.DataFrame(index = indices_df.index, columns = o_.columns)
    SET100_v_ = SET100_v_.apply(lambda col: indices_df['SET100_v'].values).ffill(limit = 20).fillna(0)
    
    I_AGRO_v_ = pd.DataFrame(index = indices_df.index, columns = o_.columns)
    I_AGRO_v_ = I_AGRO_v_.apply(lambda col: indices_df['I_AGRO_v'].values).ffill(limit = 20).fillna(0)    

    I_CONSUMP_v_ = pd.DataFrame(index = indices_df.index, columns = o_.columns)
    I_CONSUMP_v_ = I_CONSUMP_v_.apply(lambda col: indices_df['I_CONSUMP_v'].values).ffill(limit = 20).fillna(0)
    
    I_FINCIAL_v_ = pd.DataFrame(index = indices_df.index, columns = o_.columns)
    I_FINCIAL_v_ = I_FINCIAL_v_.apply(lambda col: indices_df['I_FINCIAL_v'].values).ffill(limit = 20).fillna(0)    
    
    I_INDUS_v_ = pd.DataFrame(index = indices_df.index, columns = o_.columns)
    I_INDUS_v_ = I_INDUS_v_.apply(lambda col: indices_df['I_INDUS_v'].values).ffill(limit = 20).fillna(0)
    
    I_PROPCON_v_ = pd.DataFrame(index = indices_df.index, columns = o_.columns)
    I_PROPCON_v_ = I_PROPCON_v_.apply(lambda col: indices_df['I_PROPCON_v'].values).ffill(limit = 20).fillna(0)        
    
    I_RESOURC_v_ = pd.DataFrame(index = indices_df.index, columns = o_.columns)
    I_RESOURC_v_ = I_RESOURC_v_.apply(lambda col: indices_df['I_RESOURC_v'].values).ffill(limit = 20).fillna(0)    
    
    I_TECH_v_ = pd.DataFrame(index = indices_df.index, columns = o_.columns)
    I_TECH_v_ = I_TECH_v_.apply(lambda col: indices_df['I_TECH_v'].values).ffill(limit = 20).fillna(0)
    
    I_SERVICE_v_ = pd.DataFrame(index = indices_df.index, columns = o_.columns)
    I_SERVICE_v_ = I_SERVICE_v_.apply(lambda col: indices_df['I_SERVICE_v'].values).ffill(limit = 20).fillna(0)        
    
    return SET_v_, SET50_v_, SET100_v_, I_AGRO_v_, I_CONSUMP_v_, I_FINCIAL_v_, I_INDUS_v_, I_PROPCON_v_, I_RESOURC_v_, I_TECH_v_, I_SERVICE_v_

def get_member(input_df: pd.DataFrame, start_date_, end_date_):
    print('getting member ...')
    preproc_ = input_df.copy()
    clean_price(preproc_)
    inSET_ = df_pivot(preproc_, 'SET', 'datetime', start_date_, end_date_).astype(np.float64).replace(0, np.nan)
    inSET50_ = df_pivot(preproc_, 'SET50', 'datetime', start_date_, end_date_).astype(np.float64).replace(0, np.nan)
    inSET100_ = df_pivot(preproc_, 'SET100', 'datetime', start_date_, end_date_).astype(np.float64).replace(0, np.nan)
    inI_AGRO_ = df_pivot(preproc_, 'I_AGRO', 'datetime', start_date_, end_date_).astype(np.float64).replace(0, np.nan)
    inI_CONSUMP_ = df_pivot(preproc_, 'I_CONSUMP', 'datetime', start_date_, end_date_).astype(np.float64).replace(0, np.nan)
    inI_FINCIAL_ = df_pivot(preproc_, 'I_FINCIAL', 'datetime', start_date_, end_date_).astype(np.float64).replace(0, np.nan)
    inI_INDUS_ = df_pivot(preproc_, 'I_INDUS', 'datetime', start_date_, end_date_).astype(np.float64).replace(0, np.nan)
    inI_PROPCON_ = df_pivot(preproc_, 'I_PROPCON', 'datetime', start_date_, end_date_).astype(np.float64).replace(0, np.nan)
    inI_RESOURC_ = df_pivot(preproc_, 'I_RESOURC', 'datetime', start_date_, end_date_).astype(np.float64).replace(0, np.nan)
    inI_TECH_ = df_pivot(preproc_, 'I_TECH', 'datetime', start_date_, end_date_).astype(np.float64).replace(0, np.nan)
    inI_SERVICE_ = df_pivot(preproc_, 'I_SERVICE', 'datetime', start_date_, end_date_).astype(np.float64).replace(0, np.nan)
      
    return inSET_, inSET50_, inSET100_, inI_AGRO_, inI_CONSUMP_, inI_FINCIAL_, inI_INDUS_, inI_PROPCON_, inI_RESOURC_, inI_TECH_, inI_SERVICE_

def get_member_bool(input_df: pd.DataFrame, start_date_, end_date_):
    print('getting member bool ...')
    preproc_ = input_df.copy()
    clean_price(preproc_)
    inSET_ = df_pivot(preproc_, 'SET', 'datetime', start_date_, end_date_).astype(np.float64).replace(1.0, True).replace(np.nan, False).replace(0, False)
    inSET50_ = df_pivot(preproc_, 'SET50', 'datetime', start_date_, end_date_).astype(np.float64).replace(1.0, True).replace(np.nan, False).replace(0, False)
    inSET100_ = df_pivot(preproc_, 'SET100', 'datetime', start_date_, end_date_).astype(np.float64).replace(1.0, True).replace(np.nan, False).replace(0, False)
    inI_AGRO_ = df_pivot(preproc_, 'I_AGRO', 'datetime', start_date_, end_date_).astype(np.float64).replace(1.0, True).replace(np.nan, False).replace(0, False)
    inI_CONSUMP_ = df_pivot(preproc_, 'I_CONSUMP', 'datetime', start_date_, end_date_).astype(np.float64).replace(1.0, True).replace(np.nan, False).replace(0, False)
    inI_FINCIAL_ = df_pivot(preproc_, 'I_FINCIAL', 'datetime', start_date_, end_date_).astype(np.float64).replace(1.0, True).replace(np.nan, False).replace(0, False)
    inI_INDUS_ = df_pivot(preproc_, 'I_INDUS', 'datetime', start_date_, end_date_).astype(np.float64).replace(1.0, True).replace(np.nan, False).replace(0, False)
    inI_PROPCON_ = df_pivot(preproc_, 'I_PROPCON', 'datetime', start_date_, end_date_).astype(np.float64).replace(1.0, True).replace(np.nan, False).replace(0, False)
    inI_RESOURC_ = df_pivot(preproc_, 'I_RESOURC', 'datetime', start_date_, end_date_).astype(np.float64).replace(1.0, True).replace(np.nan, False).replace(0, False)
    inI_TECH_ = df_pivot(preproc_, 'I_TECH', 'datetime', start_date_, end_date_).astype(np.float64).replace(1.0, True).replace(np.nan, False).replace(0, False)
    inI_SERVICE_ = df_pivot(preproc_, 'I_SERVICE', 'datetime', start_date_, end_date_).astype(np.float64).replace(1.0, True).replace(np.nan, False).replace(0, False)
      
    return inSET_, inSET50_, inSET100_, inI_AGRO_, inI_CONSUMP_, inI_FINCIAL_, inI_INDUS_, inI_PROPCON_, inI_RESOURC_, inI_TECH_, inI_SERVICE_


def get_industry_col(input_df, start_date_, end_date_):
    print('getting industry cols ...')
    indices_df = clean_indices(input_df, start_date_, end_date_).astype(np.float64)
    list_industry_columns = ['I_AGRO', 'I_CONSUMP', 'I_FINCIAL', 'I_INDUS', 'I_PROPCON', 'I_RESOURC', 'I_TECH', 'I_SERVICE']
    industry_df_ = pd.DataFrame(columns = list_industry_columns, index = indices_df.index)
    for industry in list_industry_columns:
        industry_df_[industry] = indices_df[industry]
    return industry_df_

def get_industry_v_col(input_df, start_date_, end_date_):
    print('getting industry volume cols ...')
    indices_df = clean_indices(input_df, start_date_, end_date_).astype(np.float64)
    list_industry_columns = ['I_AGRO', 'I_CONSUMP', 'I_FINCIAL', 'I_INDUS', 'I_PROPCON', 'I_RESOURC', 'I_TECH', 'I_SERVICE']
    industry_df_v_ = pd.DataFrame(columns = list_industry_columns, index = indices_df.index)
    for industry in list_industry_columns:
        industry_df_v_[industry] = indices_df[industry+'_v']
    return industry_df_v_

def get_benchmark_col(input_df, start_date_, end_date_):
    print('getting benchmark cols...')
    indices_df = clean_indices(input_df, start_date_, end_date_).astype(np.float64)
    list_benchmark_columns = ['SET', 'SET100', 'SET50']
    bechmark_df_ = pd.DataFrame(columns = list_benchmark_columns, index = indices_df.index)
    for benchmark in list_benchmark_columns:
        bechmark_df_[benchmark] = indices_df[benchmark]
    return bechmark_df_



def get_stock_alpha_beta(benchmark: str, period_: int, start_date_, end_date_):
    period_ = int(period_)
    stock_alpha_ = pd.read_csv('alpha_c_' + benchmark + '_' + str(period_) + '.csv')
    stock_alpha_.set_index('datetime', drop = True, inplace = True)
    stock_alpha_.index = pd.DatetimeIndex(stock_alpha_.index)
    stock_alpha_ = stock_alpha_[start_date_:end_date_].astype(np.float64).fillna(0)
    
    stock_beta_ = pd.read_csv('beta_c_' + benchmark + '_' + str(period_) + '.csv')
    stock_beta_.set_index('datetime', drop = True, inplace = True)
    stock_beta_.index = pd.DatetimeIndex(stock_beta_.index)
    stock_beta_ = stock_beta_[start_date_:end_date_].astype(np.float64).fillna(0)

    return stock_alpha_, stock_beta_

def get_stock_ROC_1_avg_all__avg_m__avg_p__var_m__var_p__es_m__es_p__max_m__max_p(period_: int, start_date_, end_date_)->(pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    print('getting get_stock_ROC_1_avg_all__avg_m__avg_p__var_m__var_p__es_m__es_p__max_m__max_p ...')
    period_ = int(period_)
    avg_all_ = pd.read_csv('avg_all_ROC_1_'+ str(period_) + '.csv')
    avg_all_.set_index('datetime', drop = True, inplace = True)
    avg_all_.index = pd.DatetimeIndex(avg_all_.index)
    avg_all_ = avg_all_[start_date_:end_date_].astype(np.float64).fillna(0)

    avg_m_ = pd.read_csv('avg_m_ROC_1_'+ str(period_) + '.csv')
    avg_m_.set_index('datetime', drop = True, inplace = True)
    avg_m_.index = pd.DatetimeIndex(avg_m_.index)
    avg_m_ = avg_m_[start_date_:end_date_].astype(np.float64).fillna(0)

    avg_p_ = pd.read_csv('avg_p_ROC_1_'+ str(period_) + '.csv')
    avg_p_.set_index('datetime', drop = True, inplace = True)
    avg_p_.index = pd.DatetimeIndex(avg_p_.index)
    avg_p_ = avg_p_[start_date_:end_date_].astype(np.float64).fillna(0)

    var_m_ = pd.read_csv('var_m_ROC_1_'+ str(period_) + '.csv')
    var_m_.set_index('datetime', drop = True, inplace = True)
    var_m_.index = pd.DatetimeIndex(var_m_.index)
    var_m_ = var_m_[start_date_:end_date_].astype(np.float64).fillna(0)

    var_p_ = pd.read_csv('var_p_ROC_1_'+ str(period_) + '.csv')
    var_p_.set_index('datetime', drop = True, inplace = True)
    var_p_.index = pd.DatetimeIndex(var_p_.index)
    var_p_ = var_p_[start_date_:end_date_].astype(np.float64).fillna(0)

    es_m_ = pd.read_csv('es_m_ROC_1_'+ str(period_) + '.csv')
    es_m_.set_index('datetime', drop = True, inplace = True)
    es_m_.index = pd.DatetimeIndex(es_m_.index)
    es_m_ = es_m_[start_date_:end_date_].astype(np.float64).fillna(0)

    es_p_ = pd.read_csv('es_p_ROC_1_'+ str(period_) + '.csv')
    es_p_.set_index('datetime', drop = True, inplace = True)
    es_p_.index = pd.DatetimeIndex(es_p_.index)
    es_p_ = es_p_[start_date_:end_date_].astype(np.float64).fillna(0)

    max_m_ = pd.read_csv('max_m_ROC_1_'+ str(period_) + '.csv')
    max_m_.set_index('datetime', drop = True, inplace = True)
    max_m_.index = pd.DatetimeIndex(max_m_.index)
    max_m_ = max_m_[start_date_:end_date_].astype(np.float64).fillna(0)

    max_p_ = pd.read_csv('max_p_ROC_1_'+ str(period_) + '.csv')
    max_p_.set_index('datetime', drop = True, inplace = True)
    max_p_.index = pd.DatetimeIndex(max_p_.index)
    max_p_ = max_p_[start_date_:end_date_].astype(np.float64).fillna(0)

    return (avg_all_, avg_m_, avg_p_, var_m_, var_p_, es_m_, es_p_, max_m_, max_p_)    


# avg_all_ROC_1, avg_m_ROC_1, avg_p_ROC_1, var_m_ROC_1, var_p_ROC_1, es_m_ROC_1, es_p_ROC_1, max_m_ROC_1, max_p_ROC_1 = get_stock_ROC_1_avg_all__avg_m__avg_p__var_m__var_p__es_m__es_p__max_m__max_p(240, start_date, end_date)


#     return (avg_all_, avg_m_, avg_p_, var_m_, var_p_, es_m_, es_p_, max_m_, max_p_)

# avg_all_df.to_csv('avg_all_240.csv')
# avg_m_df.to_csv('avg_m_240.csv')
# avg_p_df.to_csv('avg_p_240.csv')
# var_m_df.to_csv('var_m_240.csv')
# var_p_df.to_csv('var_p_240.csv')
# es_m_df.to_csv('es_m_240.csv')
# es_p_df.to_csv('es_p_240.csv')
# max_m_df.to_csv('max_m_240.csv')
# max_p_df.to_csv('max_p_240.csv')



def get_industries_alpha_beta_col(benchmark: str, period_: int, start_date_, end_date_):
    period_ = int(period_)
    industy_alpha_ = pd.read_csv('alpha_industry_' + benchmark + '_' + str(period_) + '.csv')
    industy_alpha_.set_index('datetime', drop = True, inplace = True)
    industy_alpha_.index = pd.DatetimeIndex(industy_alpha_.index)
    industy_alpha_ = industy_alpha_[start_date_:end_date_].astype(np.float64).fillna(0)
    
    industy_beta_ = pd.read_csv('beta_industry_' + benchmark + '_' + str(period_) + '.csv')
    industy_beta_.set_index('datetime', drop = True, inplace = True)
    industy_beta_.index = pd.DatetimeIndex(industy_beta_.index)
    industy_beta_ = industy_beta_[start_date_:end_date_].astype(np.float64).fillna(0)

    return industy_alpha_, industy_beta_

def series_to_df(series_: pd.Series, mother_df: pd.DataFrame):
    output_df = pd.DataFrame(index = mother_df.index, columns = mother_df.columns)
    output_df = output_df.apply(lambda col: series_.values)   
    return output_df

    
def get_industries_alpha_beta(benchmark: str, period_: int, start_date_, end_date_, o_):
    industy_alpha, industy_beta = get_industries_alpha_beta_col(benchmark, period_, start_date_, end_date_)

    I_AGRO_alpha_ = pd.DataFrame(index = industy_alpha.index, columns = o_.columns)
    I_AGRO_alpha_ = I_AGRO_alpha_.apply(lambda col: industy_alpha['I_AGRO'].values).ffill(limit = 20).fillna(0)    

    I_AGRO_beta_ = pd.DataFrame(index = industy_alpha.index, columns = o_.columns)
    I_AGRO_beta_ = I_AGRO_beta_.apply(lambda col: industy_beta['I_AGRO'].values).ffill(limit = 20).fillna(0) 

    I_CONSUMP_alpha_ = pd.DataFrame(index = industy_alpha.index, columns = o_.columns)
    I_CONSUMP_alpha_ = I_CONSUMP_alpha_.apply(lambda col: industy_alpha['I_CONSUMP'].values).ffill(limit = 20).fillna(0)    

    I_CONSUMP_beta_ = pd.DataFrame(index = industy_alpha.index, columns = o_.columns)
    I_CONSUMP_beta_ = I_CONSUMP_beta_.apply(lambda col: industy_beta['I_CONSUMP'].values).ffill(limit = 20).fillna(0) 

    I_FINCIAL_alpha_ = pd.DataFrame(index = industy_alpha.index, columns = o_.columns)
    I_FINCIAL_alpha_ = I_FINCIAL_alpha_.apply(lambda col: industy_alpha['I_FINCIAL'].values).ffill(limit = 20).fillna(0)    

    I_FINCIAL_beta_ = pd.DataFrame(index = industy_alpha.index, columns = o_.columns)
    I_FINCIAL_beta_ = I_FINCIAL_beta_.apply(lambda col: industy_beta['I_FINCIAL'].values).ffill(limit = 20).fillna(0) 

    I_INDUS_alpha_ = pd.DataFrame(index = industy_alpha.index, columns = o_.columns)
    I_INDUS_alpha_ = I_INDUS_alpha_.apply(lambda col: industy_alpha['I_INDUS'].values).ffill(limit = 20).fillna(0)    

    I_INDUS_beta_ = pd.DataFrame(index = industy_alpha.index, columns = o_.columns)
    I_INDUS_beta_ = I_INDUS_beta_.apply(lambda col: industy_beta['I_INDUS'].values).ffill(limit = 20).fillna(0) 

    I_PROPCON_alpha_ = pd.DataFrame(index = industy_alpha.index, columns = o_.columns)
    I_PROPCON_alpha_ = I_PROPCON_alpha_.apply(lambda col: industy_alpha['I_PROPCON'].values).ffill(limit = 20).fillna(0)    

    I_PROPCON_beta_ = pd.DataFrame(index = industy_alpha.index, columns = o_.columns)
    I_PROPCON_beta_ = I_PROPCON_beta_.apply(lambda col: industy_beta['I_PROPCON'].values).ffill(limit = 20).fillna(0) 
       
    I_RESOURC_alpha_ = pd.DataFrame(index = industy_alpha.index, columns = o_.columns)
    I_RESOURC_alpha_ = I_RESOURC_alpha_.apply(lambda col: industy_alpha['I_RESOURC'].values).ffill(limit = 20).fillna(0)    

    I_RESOURC_beta_ = pd.DataFrame(index = industy_alpha.index, columns = o_.columns)
    I_RESOURC_beta_ = I_RESOURC_beta_.apply(lambda col: industy_beta['I_RESOURC'].values).ffill(limit = 20).fillna(0) 
    
    I_TECH_alpha_ = pd.DataFrame(index = industy_alpha.index, columns = o_.columns)
    I_TECH_alpha_ = I_TECH_alpha_.apply(lambda col: industy_alpha['I_TECH'].values).ffill(limit = 20).fillna(0)    

    I_TECH_beta_ = pd.DataFrame(index = industy_alpha.index, columns = o_.columns)
    I_TECH_beta_ = I_TECH_beta_.apply(lambda col: industy_beta['I_TECH'].values).ffill(limit = 20).fillna(0) 
    
    I_SERVICE_alpha_ = pd.DataFrame(index = industy_alpha.index, columns = o_.columns)
    I_SERVICE_alpha_ = I_SERVICE_alpha_.apply(lambda col: industy_alpha['I_SERVICE'].values).ffill(limit = 20).fillna(0)    

    I_SERVICE_beta_ = pd.DataFrame(index = industy_alpha.index, columns = o_.columns)
    I_SERVICE_beta_ = I_SERVICE_beta_.apply(lambda col: industy_beta['I_SERVICE'].values).ffill(limit = 20).fillna(0) 
    


    return I_AGRO_alpha_, I_AGRO_beta_, I_CONSUMP_alpha_, I_CONSUMP_beta_, I_FINCIAL_alpha_, I_FINCIAL_beta_, I_INDUS_alpha_, I_INDUS_beta_\
        , I_PROPCON_alpha_, I_PROPCON_beta_, I_RESOURC_alpha_, I_RESOURC_beta_, I_TECH_alpha_, I_TECH_beta_, I_SERVICE_alpha_, I_SERVICE_beta_
    
    















