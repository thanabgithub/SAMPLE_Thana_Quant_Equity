from typing import List
import numpy as np
from datetime import datetime
from potter import backtester as bt
from potter.backtester import factor as fc
import json
from datetime import date
from typing import Union
from pandas import DataFrame
import pandas as pd
import requests
import math
import json
import time
import shutil
from pathlib import Path

from potter.research import psims_research as psims

import ThaiEQ_strat_tool.data_import_csv as data
from ThaiEQ_strat_tool.indicators import *
import ThaiEQ_strat_tool.backtest as thana_backtester

import datetime as dt

def check_universe_SET100(universe_SET100, test_type, line_token):
    # use the following because universe_SET100 has 104 stocks from the past until 2013
    universe_SET100_mini = universe_SET100['2014':]
    # count number of stocks in SET100 everyday 
    count_SET100_everyday = ((universe_SET100_mini.sum(axis = 1).round() - 100).abs() < 0.001)
    count_SET100_until_prev = count_SET100_everyday.iloc[:-1].sum()
    # confirm the correctness of the series:
    check = ( abs(count_SET100_until_prev - len(count_SET100_everyday.iloc[:-1])) < 0.001 ) 
    if check:
        if test_type == 'signal':
            send_line(line_token, 'perfect SET100 (' + str(len(count_SET100_everyday.iloc[:-1])) + 'days)')
        print('check_universe_SET100')
        return True
    else:
        print("Weird universe SET100")
        error_perc = 1 - count_SET100_until_prev/len(count_SET100_everyday.iloc[:-1])
        MSG = 'error!!! \nuniverse SET100 is \nFUCKED UP \nFUCKED UP \n' + str(error_perc*100) + '% of total days\nABORT'
        print(MSG)
        if test_type == 'signal':
            send_line(line_token, MSG)
        raise      


def check_num_of_sig(target_weight, n_stocks, test_type, line_token):
    existing_sig_today = (target_weight > 0.00001).astype(np.float64).sum(axis = 1)[-2]    
    if existing_sig_today < n_stocks:
        if test_type == 'signal':
            send_line(line_token, "acceptable target_weight's num: " + str(existing_sig_today) + ' holdings')
        print('check_num_of_sig')
        return True
    else:
        print("Weird target weight")
        MSG = "error!!! \ntarget_weight's num is \nFUCKED UP \nFUCKED UP \n" + str(existing_sig_today) + ' holdings\nABORT'
        print(MSG)
        if test_type == 'signal':
            send_line(line_token, MSG)
        raise        

def check_over_pos(target_weight, init_weight, test_type, line_token):
    max_sig = target_weight.max(axis = 1)[-2]    
    if max_sig < init_weight:
        if test_type == 'signal':
            send_line(line_token, "acceptable target_weight's max: " + str(max_sig*100) + '%')
        print('check_over_pos')
        return True
    else:
        print("Weird pos")
        MSG = "error!!! \ntarget_weight's max is \nFUCKED UP \nFUCKED UP \n" + str(max_sig*100) + '%'
        print(MSG)
        if test_type == 'signal':
            send_line(line_token, MSG)
        raise        
    
        
def check_total_pos(target_weight, test_type, line_token):
    total_sig = target_weight.sum(axis = 1)[-2]    
    if total_sig < 1.001:
        print('check_total_pos')
        if test_type == 'signal':
            send_line(line_token, "acceptable target_weight's sum: " + str(total_sig*100) + '%')
        return True
    else:
        print("Weird pos")
        MSG = "error!!! \ntarget_weight's sum is \nFUCKED UP \nFUCKED UP \n" + str(total_sig*100) + '%'
        print(MSG)
        if test_type == 'signal':
            send_line(line_token, MSG)
        raise         

def check_pos_dim(target_weight, universe_all, test_type, line_token):
    target_weight_shape = target_weight.shape
    universe_all_shape = universe_all.shape    
    dim_check = (target_weight_shape == universe_all_shape)        
    if dim_check:
        print('check_pos_dim')
        if test_type == 'signal':
            send_line(line_token, "Similar target_weight's dim and universe_all's dim" + str(target_weight_shape))
        return True
    else:
        print("Weird target_weight's dim")
        MSG = "error!!! \ntarget_weight's dim is \nFUCKED UP \nFUCKED UP \n" + str(target_weight_shape) + "\nuniverse_all's dim is \nFUCKED UP \nFUCKED UP \n" + str(universe_all_shape)
        print(MSG)
        send_line(line_token, MSG)
        raise     
 
def check_no_liquidity(target_weight, value, test_type, line_token):
    
    check_scope = data_exist_1_nan(target_weight.replace(0, np.nan)) 
    value_to_check_ss = (check_scope*MA(value, 3))
    value_to_check_s = (check_scope*MA(value, 5)) 
    value_to_check_m = (check_scope*MA(value, 20)) 
    barSince_IPO = BarsSince_first_trg(~(value.ffill().isnull()))*data_exist_1_nan(value) 
    min_liq = 1*1000000
    check_ss = (value_to_check_ss > min_liq)|(barSince_IPO < 3)
    check_s = (value_to_check_s > min_liq)|(barSince_IPO < 5)
    check_m = (value_to_check_m > min_liq)|(barSince_IPO < 20)   
    consolidate_check = check_ss&value_to_check_s&value_to_check_m
    consolidate_check_count = consolidate_check.sum(axis = 1)
    final_check = abs(consolidate_check_count.iloc[-2] - check_scope.sum(axis = 1).iloc[-2]) < 0.001
    if final_check:
        print('check_no_liquidity')        
        if test_type == 'signal':
            send_line(line_token, "acceptable liquidity")
        return True
    else:
        print("Weird target_weight's liq")
        MSG = "error!!! \ntarget_weight's liquidity is \nFUCKED UP \nFUCKED UP"
        print(MSG)
        if test_type == 'signal':
            send_line(line_token, MSG)
        raise     




        
def get_data_backtest(db_host, df_user, db_pass, db_port, gs_start_backtest, gs_end_backtest):
    from datetime import date
    from potter.research import psims_research as psims    
    psims.set_psims_connection("postgresql://postgres:1234@192.168.100.197:5432/psims")
    
    psim_last_date = psims.get_trading_datetimes(
        start_date=date(2006, 1, 1),
        end_date = gs_end_backtest,
    )
    gs_start_load = str(int(gs_start_backtest[0:4]) - 3) + gs_start_backtest[4:]
    try:
        i = -2
        backtester_MAI = bt.SETUniverseSignalBacktester(start_date = gs_start_load, end_date = psim_last_date[i], universe = 'MAI' , db_host = db_host, db_user = df_user, db_pass = db_pass, db_port = db_port)
        backtester_SETMAI = bt.SETUniverseSignalBacktester(start_date = gs_start_load, end_date = psim_last_date[i], universe = 'SETMAI' , db_host = db_host, db_user = df_user, db_pass = db_pass, db_port = db_port)
        backtester_SET = bt.SETUniverseSignalBacktester(start_date = gs_start_load, end_date = psim_last_date[i], universe = 'SET' , db_host = db_host, db_user = df_user, db_pass = db_pass, db_port = db_port)
        backtester_SET100 = bt.SETUniverseSignalBacktester(start_date = gs_start_load, end_date = psim_last_date[i], universe = "SET100" , db_host = db_host, db_user = df_user, db_pass = db_pass, db_port = db_port)
    except:
        try:
            i = -3
            backtester_MAI = bt.SETUniverseSignalBacktester(start_date = gs_start_load, end_date = psim_last_date[i], universe = 'MAI' , db_host = db_host, db_user = df_user, db_pass = db_pass, db_port = db_port)
            backtester_SETMAI = bt.SETUniverseSignalBacktester(start_date = gs_start_load, end_date = psim_last_date[i], universe = 'SETMAI' , db_host = db_host, db_user = df_user, db_pass = db_pass, db_port = db_port)
            backtester_SET = bt.SETUniverseSignalBacktester(start_date = gs_start_load, end_date = psim_last_date[i], universe = 'SET' , db_host = db_host, db_user = df_user, db_pass = db_pass, db_port = db_port)
            backtester_SET100 = bt.SETUniverseSignalBacktester(start_date = gs_start_load, end_date = psim_last_date[i], universe = "SET100" , db_host = db_host, db_user = df_user, db_pass = db_pass, db_port = db_port)
        except:
            i = -4
            backtester_MAI = bt.SETUniverseSignalBacktester(start_date = gs_start_load, end_date = psim_last_date[i], universe = 'MAI' , db_host = db_host, db_user = df_user, db_pass = db_pass, db_port = db_port)
            backtester_SETMAI = bt.SETUniverseSignalBacktester(start_date = gs_start_load, end_date = psim_last_date[i], universe = 'SETMAI' , db_host = db_host, db_user = df_user, db_pass = db_pass, db_port = db_port)
            backtester_SET = bt.SETUniverseSignalBacktester(start_date = gs_start_load, end_date = psim_last_date[i], universe = 'SET' , db_host = db_host, db_user = df_user, db_pass = db_pass, db_port = db_port)
            backtester_SET100 = bt.SETUniverseSignalBacktester(start_date = gs_start_load, end_date = psim_last_date[i], universe = "SET100" , db_host = db_host, db_user = df_user, db_pass = db_pass, db_port = db_port)
    return backtester_MAI, backtester_SETMAI, backtester_SET, backtester_SET100


def load_basic(backtester):
    backtester.load_1d([fc.D_OPEN, fc.D_HIGH, fc.D_LOW, fc.D_CLOSE, fc.D_VOLUME, fc.D_VALUE, fc.D_PE, fc.D_MKT_CAP, fc.D_SHARE_LISTED]) 
    o, h, l, c, vol, value, MKC = backtester.get_1d(fc.D_OPEN, False), backtester.get_1d(fc.D_HIGH, False), backtester.get_1d(fc.D_LOW, False), backtester.get_1d(fc.D_CLOSE, False), backtester.get_1d(fc.D_VOLUME, False), backtester.get_1d(fc.D_VALUE, False), backtester.get_1d(fc.D_MKT_CAP, False)
    vwap = value/ vol
    trading_day_index = c.index.tolist()
    datetime_str = str(datetime.now().year) + '-' + str(datetime.now().month) + '-' + str(datetime.now().day)
    trading_day_index.append(pd.to_datetime(datetime_str))
    
    o = o.reindex(trading_day_index)
    h = h.reindex(trading_day_index)
    l = l.reindex(trading_day_index)
    c = c.reindex(trading_day_index)
    vol = vol.reindex(trading_day_index)
    value = value.reindex(trading_day_index)
    vwap = vwap.reindex(trading_day_index)
    MKC = MKC.reindex(trading_day_index)    
    
    return o, h, l, c, vol, value, vwap, MKC 

def load_index(c):
  
    MAI_finnix = psims.get_data_index_1d(["MAI"], ["close"], c.index[0], c.index[-1])['close']
    MAI = pd.DataFrame(index = MAI_finnix.index, columns = c.columns)    
    MAI = MAI.apply(lambda col: MAI_finnix['mai'].values)
    MAI = MAI.reindex(c.index)
    
    SET_finnix = psims.get_data_index_1d(["SET"], ["close"], c.index[0], c.index[-1])['close']
    SET = pd.DataFrame(index = SET_finnix.index, columns = c.columns)
    SET = SET.apply(lambda col: SET_finnix['SET'].values)
    SET = SET.reindex(c.index)

    SET100_finnix = psims.get_data_index_1d(["SET100"], ["close"], c.index[0], c.index[-1])['close']    
    SET100 = pd.DataFrame(index = SET100_finnix.index, columns = c.columns)
    SET100 = SET100.apply(lambda col: SET100_finnix['SET100'].values)
    SET100 = SET100.reindex(c.index)
    
    return MAI, SET, SET100

def load_universe(c, backtester_SET100):

    universe_all_raw = data_exist_1_nan(c)
    universe_SET100_raw = backtester_SET100.universe_df.replace(True, 1).replace(False, np.nan)

    c_stock_list = c.columns
    c_stock_set = set(c.columns)

    universe_SET_raw = universe_all_raw[c_stock_list]
    
    SET100_stock_set = set(universe_SET100_raw.columns)    
    SET100_stock_set = c_stock_set.intersection(SET100_stock_set)
                                                                                  
    universe_blueprint = pd.DataFrame(index = c.index, columns = c_stock_list, data = np.nan)   
    universe_all = universe_SET_raw[list(c_stock_list)]
    universe_SET100 = init_df(universe_blueprint, np.nan)
    universe_SET100[list(SET100_stock_set)] = universe_SET100_raw[list(SET100_stock_set)] 


    return universe_all, universe_SET100

def load_custom_q(backtester, c, factor):
    backtester.load_1q([factor])    
    var_raw = backtester.get_1q(factor, False).reindex(c.index)
    var = pd.DataFrame(columns = c.columns, index = c.index, data = np.nan)
    var[var_raw.columns] = var_raw
    return var

def load_custom_d(backtester, c, factor):
    backtester.load_1d([factor])    
    var_raw = backtester.get_1d(factor, False).reindex(c.index)
    var = pd.DataFrame(columns = c.columns, index = c.index, data = np.nan)
    var[var_raw.columns] = var_raw
    return var

def load_custom_univ(MKC, MKC_perc_in, MKC_days, value, value_perc_in, value_days):
    universe_all = data_exist_1_nan(MKC)
    MKC_sm = MKC.copy()
    value_sm = WMA(value, 20)
    
    MKC_ranking = ranking_DSC_within_universe_pct(MKC_sm, universe_all)
    MKC_ranking_sm = MA(MKC_ranking, MKC_days)
    MKC_cap_in = (MKC_ranking.round(5) < MKC_perc_in)&(MKC_ranking_sm.round(5) < MKC_perc_in)
    MKC_cap_out = (MKC_ranking.round(5) > MKC_perc_in)&(MKC_ranking_sm.round(5) > MKC_perc_in)
    universe_MKC_cap_raw = one_zero_gen_frm_BUYSELL(MKC_cap_in, MKC_cap_out, universe_all)
    
    value_ranking = ranking_DSC_within_universe_pct(value, universe_all)
    value_ranking_sm = MA(value_ranking, value_days)
    value_cap_in = (value_ranking.round(5) < value_perc_in)&(value_ranking_sm.round(5) < value_perc_in)
    value_cap_out = (value_ranking.round(5) > value_perc_in)&(value_ranking_sm.round(5) > value_perc_in)
    
    universe_value_cap_raw = one_zero_gen_frm_BUYSELL(value_cap_in, value_cap_out, universe_all)
    
    return universe_MKC_cap_raw*universe_value_cap_raw  

def load_custom_univ(MKC, MKC_perc_in, MKC_days, value, value_perc_in, value_days):
    universe_all = data_exist_1_nan(MKC)
    MKC_sm = MKC.copy()
    value_sm = WMA(value, 20)
    
    MKC_ranking = ranking_DSC_within_universe_pct(MKC_sm, universe_all)
    MKC_ranking_sm = MA(MKC_ranking, MKC_days)
    MKC_cap_in = (MKC_ranking.round(5) < MKC_perc_in)&(MKC_ranking_sm.round(5) < MKC_perc_in)
    MKC_cap_out = (MKC_ranking.round(5) > MKC_perc_in)&(MKC_ranking_sm.round(5) > MKC_perc_in)
    universe_MKC_cap_raw = one_zero_gen_frm_BUYSELL(MKC_cap_in, MKC_cap_out, universe_all).replace(0, np.nan)
    
    value_ranking = ranking_DSC_within_universe_pct(value, universe_all)
    value_ranking_sm = MA(value_ranking, value_days)
    value_cap_in = (value_ranking.round(5) < value_perc_in)&(value_ranking_sm.round(5) < value_perc_in)
    value_cap_out = (value_ranking.round(5) > value_perc_in)&(value_ranking_sm.round(5) > value_perc_in)
    
    universe_value_cap_raw = one_zero_gen_frm_BUYSELL(value_cap_in, value_cap_out, universe_all).replace(0, np.nan)
    
    return universe_MKC_cap_raw*universe_value_cap_raw


def load_SP_DELISTED(c, gs_start_backtest, gs_end_backtest):

    from potter.research import psims_research as psims
    
    psims.set_psims_connection("postgresql://postgres:1234@192.168.100.197:5432/psims")    
    gs_start_load = str(int(gs_start_backtest[0:4]) - 5) + gs_start_backtest[4:]
    get_sp = psims.get_sp(
        symbol_list=c.columns,
        start_date=gs_start_load,
        end_date=gs_end_backtest,
    )
    
    get_sp = get_sp.drop('sign', axis = 1)
    get_sp['bool'] = True
    SP_ = pd.pivot_table(get_sp, columns = 'symbol', index = 'trading_datetime', values = 'bool')
    SP_ = SP_.reindex(c.index)
    SP = init_df(c, False)
    SP[SP_.columns] = SP_
    
    
    get_delisted = psims.get_delisted(
        symbol_list=c.columns,
        start_date=gs_start_load,
        end_date=gs_end_backtest,
    )
    
    get_delisted['bool'] = True
    DELISTED_ = pd.pivot_table(get_delisted, columns = 'symbol', index = 'delisted_datetime', values = 'bool')
    DELISTED_ = DELISTED_.reindex(c.index)
    DELISTED = init_df(c, False)
    DELISTED[DELISTED_.columns] = DELISTED_
    return SP, DELISTED

def load_DPX_XD(c, gs_start_backtest, gs_end_backtest):
    
    from potter.research import psims_research as psims
    
    psims.set_psims_connection("postgresql://postgres:1234@192.168.100.197:5432/psims")
    gs_start_load = str(int(gs_start_backtest[0:4]) - 5) + gs_start_backtest[4:]
    DPS_XD_ = psims.get_cash_dividend(
        symbol_list= c.columns,
        start_date=gs_start_load,
        end_date=gs_end_backtest,
    )
    
    DPS_XD_ = pd.pivot_table(DPS_XD_, columns = 'symbol', index = 'ex_date', values = 'dps')
    DPS_XD_ = DPS_XD_.reindex(c.index)
    DPS_XD = init_df(c, np.nan)
    DPS_XD[DPS_XD_.columns] = DPS_XD_
    
    return DPS_XD

def get_path():
    import os
    try:
        script = os.path.realpath(__file__) 
        path = os.path.dirname(script)
        # print('in try')
    except:
        path = os.getcwd()   
    return path

def fintech_backtest(target_weight, gs_name, backtester_focus, gs_start_backtest, gs_end_backtest, gs_slippage_ticker_buy, gs_slippage_ticker_sell, gs_commission, gs_initial_deposit, save = False):
    signal_df = target_weight.shift()
    ready_signal_df = signal_df
    
    # change to dict 
    signal_dict = ready_signal_df.to_dict("index")
    for k, v in signal_dict.items():
        signal_dict[k] = {sym: weight for sym, weight in v.items() if weight != 0.0}
        

    end_date_ = gs_end_backtest
    if type(end_date_) == str:
        end_date_ = pd.to_datetime(gs_end_backtest)
    if end_date_ > target_weight.index[-2]:
        end_date_ = target_weight.index[-2]
    s = datetime.now().timestamp()
    result = backtester_focus.backtest_daily_rebalance(
        signal_dict = signal_dict,
        start_date=gs_start_backtest,
        end_date= end_date_,
        buy_slip=gs_slippage_ticker_buy,
        sell_slip=gs_slippage_ticker_sell,
        pct_commission=gs_commission,
        initial_cash=gs_initial_deposit,
    )
    plt.style.use('seaborn-whitegrid') 
    import seaborn as sns
    sns.set_context("notebook")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    
    summary_df = result.get_summary_df()
    
    summary_df.port_value.plot(ax = ax1, label = 'cap_gain')
    summary_df.port_value_with_dividend.plot(ax = ax1, label = 'cap_gain + div')
    ax1.set_xlabel('year')
    ax1.legend()
    
    port_NAV = summary_df.port_value_with_dividend
    curve_cum = pd.DataFrame(index = port_NAV.index)
    curve_cum['curve'] = port_NAV.values
    curve_cum['high-water'] = port_NAV.cummax().values
    curve_cum.plot( logy=True, ax = ax2)
    ax2.set_xlabel('year')

    
    monthly_return = result.get_monthly_return_df().reset_index()
    monthly_return = monthly_return[monthly_return['nav_name'] == 'port_value']
    monthly_return = monthly_return[['year', 'YTD']]
    # monthly_return.set_index('year', inplace = True)
    monthly_return.YTD = monthly_return.YTD*100

    pal = sns.color_palette("RdYlGn_r", len(monthly_return) + 4)
    del pal[int(len(monthly_return)/2)]
    del pal[int(len(monthly_return)/2)]
    del pal[int(len(monthly_return)/2)]
    del pal[int(len(monthly_return)/2)]
    rank = monthly_return.YTD.argsort().argsort()
    g = sns.barplot(data = monthly_return, x = 'year', y = 'YTD', ax = ax3, palette=np.array(pal[::-1])[rank])
    g.set(xlabel = 'year', ylabel = 'YTD')
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation= 45) 

    
    
    fig.set_size_inches(15, 5)

    if save:

        import os
        now = datetime.now()
        folder_name_time_format = now.strftime("%Y%m%d%H%M%S")
        path = get_path()[:-18]
        report_directory_ = path + '\\report\\' + gs_name  + '_' + str(folder_name_time_format)
        print(report_directory_)
        if (not os.path.exists(report_directory_)):
            os.mkdir(report_directory_)
            import time
            time.sleep(1)

        result.to_excel(report_directory_+ "\\" +str(gs_name) + ".xlsx")
        
        
        thana_backtester.backup_code(gs_name, path, now)
    print(result.get_stat_df())        
    return result, fig  
        
def get_signal_dict(
    signal_df: pd.DataFrame, model_name: str, last_signal_date: Union[date, str]
) -> dict:
    signal_df["signal_datetime"] = signal_df["signal_datetime"].apply(
        lambda x: str(x.date())
    )
    signal_df = signal_df.rename(
        columns={i: to_camel_case(i) for i in signal_df.columns}
    )
    last_signal_date = pd.to_datetime(last_signal_date).strftime("%Y-%m-%d")

    signal_dict = {
        "modelName": model_name,
        "lastSignalDate": last_signal_date,
        "modelSignals": signal_df.to_dict("records"),
    }
    return signal_dict

def save_signal_json(signal_dict: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(signal_dict, f)

def to_camel_case(snake_str: str) -> str:
    components = snake_str.split("_")
    return components[0] + "".join(x.title() for x in components[1:])

def gen_presig(target_weight):


    import os
    try:
        script = os.path.realpath(__file__) 
        path = os.path.dirname(script)
        # print('in try')
    except:
        path = os.getcwd()   
        # print('in except')
    
    
    
    
    from typing import Union
    

    
    
    # signal_df_g = target_weight.shift().iloc[:-1, :] # signal_df.shift(1)
    
    signal_df_g = target_weight.shift() # signal_df.shift(1)
    
    
    signal_dict_df = []
    for index_i in range(signal_df_g.shape[0]):
        if index_i % 100 == 0:
            print(index_i)
        target_weight_i = signal_df_g.iloc[index_i, :].replace(0, np.nan).dropna()
        signal_dict_df_i = pd.DataFrame(columns = ['signal_datetime', 'symbol', 'action', 'amount_type', 'amount_value'], index = np.arange(target_weight_i.index.size))
        signal_dict_df_i['signal_datetime'] = target_weight_i.name
        signal_dict_df_i['symbol'] = target_weight_i.index
        signal_dict_df_i['action'] = 'HOLD'
        signal_dict_df_i['amount_type'] = 'PCT_PORT'
        signal_dict_df_i['amount_value'] = target_weight_i.values
        signal_dict_df.append(signal_dict_df_i)
    
    
    
    signal_dict_df = pd.concat(signal_dict_df,ignore_index=True)

    return signal_dict_df_i, signal_dict_df, signal_df_g

def upload_from_2010(signal_dict_df, signal_df_g, model_name, api_key_, api_secret_, finnix_url):




    path = get_path()[:-18]
    start_date = datetime(2010,1,1)
    end_date = signal_df_g.index[-1].date()
    
    signal_dict_df['signal_datetime'] = pd.to_datetime(signal_dict_df.signal_datetime)
    signal_dict_df = (signal_dict_df[signal_dict_df['signal_datetime'] >= start_date])
    
    signal_dict_export = get_signal_dict(
            signal_dict_df, model_name, last_signal_date = end_date
        )
    
    signal_dict_df = signal_dict_df.reset_index().drop('index', axis = 1)
    
    signal_dict_save_json = save_signal_json(signal_dict_export, model_name + ".json")
    
    # 6913571255669006
    # DieXIdFeg.M7jt3g
    
    
    # # Upload Json File
    # def upload_signals_json(
    #     file_path: str, finnix_url: str, api_key: str, api_secret: str
    # ) -> None:
    #     data = json.load(open(file_path))
    #     res = requests.post(
    #         f"{finnix_url}/api/model/signal/add?apiKey={api_key}&apiSecret={api_secret}",
    #         json=data,
    #     )
    #     if not res.ok:
    #         print(res.text)

    print(path + "/" + model_name + ".json")
    data = json.load(open(path + "/" + model_name + ".json"))
    print(data)
    res = requests.post(
            f"{finnix_url}/api/model/signal/add?apiKey={api_key_}&apiSecret={api_secret_}",
            json=data,)
    print(res.text)

    
    # upload_signals_json(
    #     file_path= path + "/" + model_name + ".json",
    #     finnix_url= finnix_url_,
    #     api_key= api_key_,
    #     api_secret= api_secret_
    # )
    
def upload_from_test(signal_dict_df, signal_df_g, model_name, api_key_, api_secret_, finnix_url):




    path = get_path()[:-18]
    start_date = datetime(2020,1,1)
    end_date = signal_df_g.index[-1].date()
    
    signal_dict_df['signal_datetime'] = pd.to_datetime(signal_dict_df.signal_datetime)
    signal_dict_df = (signal_dict_df[signal_dict_df['signal_datetime'] >= start_date])

    signal_dict_export = get_signal_dict(
            signal_dict_df, model_name, last_signal_date = end_date
        )
    
    signal_dict_df = signal_dict_df.reset_index().drop('index', axis = 1)
    
    signal_dict_save_json = save_signal_json(signal_dict_export, model_name + ".json")
    

    print(path + "/" + model_name + ".json")
    data = json.load(open(path + "/" + model_name + ".json"))
    print(data)
    res = requests.post(
            f"{finnix_url}/api/model/signal/add?apiKey={api_key_}&apiSecret={api_secret_}",
            json=data,)
    print(res.text)
    print(signal_dict_df.head())
    

    
    
    
def upload_today_sig(signal_dict_df_i, signal_df_g, line_token, model_name, api_key_, api_secret_, finnix_url):
    end_date = signal_df_g.index[-1].date()
    signal_dict_export = get_signal_dict(
            signal_dict_df_i  , model_name, last_signal_date=end_date
        )


    def widespace(sentence: str, word: str, start: int = None, end: int = None):
        if not start:
            start = 0
        if not end:
            end = len(sentence)
    
        spacedword = " ".join(word)
        indices = []
        cur = sentence.find(word, start)
    
        # Find all occurrences of the word
        while cur >= 0 and cur < end + len(word) - 1:
            # Add to list
            indices.append(cur) 
    
            # Next occurrence
            cur = cur + len(word)
            cur = sentence.find(word, cur, end + len(word) - 1)
    
        # Replace word with spaced-out word
        while len(indices) > 0:
            index = indices.pop()
            sa = sentence[:index]
            sb = sentence[index:index + len(word)]
            sc = sentence[index + len(word):]
            sb = sb.replace(word, spacedword)
            sentence = sa + sb + sc
    
        return sentence
    
    today_sig_clean = signal_dict_df_i[['symbol', 'amount_value']]
    # today_sig_clean['symbol'] = today_sig_clean['symbol'].apply(lambda row: "{0:<10}".format(row))
    today_sig_clean['amount_value'] = (today_sig_clean['amount_value']*100).round(3)
    today_sig_clean['amount_value'] = today_sig_clean['amount_value'].apply(lambda row: f' {row:2.1f}' + '%')
    # today_sig_clean['amount_value'] = today_sig_clean['amount_value'].apply(lambda row: "{0:>5}".format(row))
    sig_str = '\n'
    for row in range(today_sig_clean.shape[0]):
        for col in range(today_sig_clean.shape[1]):
            var = today_sig_clean.iloc[row, col]
            if col%2 ==0:
                var = "{0:.<12}".format(var) # widespace(var, var)
            else:
                var = "{0:.>10}".format(var)
            sig_str = sig_str + var
        sig_str = sig_str + '\n'
    time.sleep(1)    


    url = 'https://notify-api.line.me/api/notify'
    headers = {'content-type':'application/x-www-form-urlencoded','Authorization':'Bearer '+line_token}
    MSG = '\n' + model_name + '\n---' + str(sig_str)
    # r = requests.post(url, headers=headers, data = {'message': MSG})  
    
    import dataframe_image as dfi
    
    temp = signal_dict_df_i[['symbol', 'amount_value']]
    temp.columns =['symbol', '%']
    temp['%'] = (temp['%']*100)
    temp = temp.sort_values(by = '%', ascending=False)
    temp['%'] = temp['%'].apply(lambda row: f' {row:3.2f}')
    temp = temp.reset_index(drop=True)
    temp.index = temp.index + 1
            
    dfi.export(temp,"temp_table.png")

    prefile = open('temp_table.png','rb')
    file = {'imageFile':prefile}
    data = ({
    'message': MSG
    })
    LINE_HEADERS = {"Authorization":"Bearer "+line_token}
    session = requests.Session()
    url = 'https://notify-api.line.me/api/notify'
    r=session.post(url, headers=LINE_HEADERS, files=file, data = data)
    prefile.close()

    import os
    os.remove("temp_table.png")
    
    
    try:
        def upload_signals(signal_dict: dict, finnix_url: str, api_key: str, api_secret: str):
              res = requests.post(
                  f"{finnix_url}/api/model/signal/add?apiKey={api_key}&apiSecret={api_secret}",
                  json=signal_dict,
              )
              if not res.ok:
                    raise Exception(f"{res.json()['message']}")
          
        # -- upload signal
        upload_signals(
            signal_dict=signal_dict_export,
            finnix_url= finnix_url,
            api_key= api_key_,
            api_secret= api_secret_
        )
        print("bot-finnize","upload [SUCCESS]")
        MSG = signal_dict_df_i
        # r = requests.post(url, headers=headers, data = {'message': '\n success'})  
        send_line(line_token, '\n success')
    except Exception as e:
        print("bot-finnize",str(e)+"  [Fail]")
        # send_message("bot-finnize-uat","upload MO THANA Fail (weekend or error) [UAT FAIL]")
        # r = requests.post(url, headers=headers, data = {'message': '\n fail'}) 
        # r = requests.post(url, headers=headers, data = {'message': 'because \n' + str(e)})    
        send_line(line_token, '\nFAILED \n' + str(e))


    
    
def send_line(line_token, text):
    try:
        url = 'https://notify-api.line.me/api/notify'
        headers = {'content-type':'application/x-www-form-urlencoded','Authorization':'Bearer '+line_token}
        MSG = text
        r = requests.post(url, headers=headers, data = {'message': MSG})
    except:
        print('line noti failed')
        r = None
    return r   
        
def data_update_check(c, gs_end_backtest, model_name, line_token):
    from datetime import date
    from potter.research import psims_research as psims    
    psims.set_psims_connection("postgresql://postgres:1234@192.168.100.197:5432/psims")
    psim_last_date = psims.get_trading_datetimes(
        start_date=date(2006, 1, 1),
        end_date = gs_end_backtest,
    )
    
    status = c.index[-2] == psim_last_date[-2]   
    url = 'https://notify-api.line.me/api/notify'
    headers = {'content-type':'application/x-www-form-urlencoded','Authorization':'Bearer '+line_token}
    MSG = '\n' + model_name + '\n data update status:\t' + str(status)
    # r = requests.post(url, headers=headers, data = {'message': MSG})  
    if not status:
        send_line(line_token, MSG)
    return status


def check_today_sig_trading_day(today_sig, line_token) :
    interesting_day = today_sig.signal_datetime.iloc[-1]
    import datetime
    next_few_days = interesting_day + datetime.timedelta(days=20)
    
    
    from datetime import date
    from potter.research import psims_research as psims        
    
    
    psim_last_date = psims.get_trading_datetimes(
    start_date=date(2006, 1, 1),
    end_date = next_few_days,
    )
    status = (interesting_day in psim_last_date)
    if not status:
        MSG = "no sig to send because today is not a trading day"
        send_line(line_token, MSG)
    return status

