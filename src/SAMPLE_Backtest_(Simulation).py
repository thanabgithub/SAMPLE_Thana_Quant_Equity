#%% -- BACKTEST
from typing import List
import numpy as np
from datetime import datetime

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

from finnix_research import psims_research as psims

import ThaiEQ_strat_tool.data_import_csv as data
from ThaiEQ_strat_tool.indicators import *
import ThaiEQ_strat_tool.backtest as thana_backtester
from ThaiEQ_strat_tool.thana_fintech import *

import datetime as dt

import xarray as xr

import matplotlib as mpl


now = datetime.now()

filename = gs_name # must always include .py 


import os
try:
    script = os.path.realpath(__file__) 
    path = os.path.dirname(script)
    # print('in try')
except:
    path = os.getcwd()   
    # print('in except')





setting_backtest = {
        'initial_deposit': np.float64(round(gs_initial_deposit, 2)),
        'slip_pec': np.float64(round(gs_slippage_perc, 5)),
        'comm_fee_perc': np.float64(round(gs_commission, 5)),
        'management_fee': np.float64(round(0.00/100.00, 5)),
        'start_date': gs_start_backtest,#  '2019-01-01'
        'end_date': gs_end_backtest,
        'display': True,
        'benchmark': SET,
        'save_fig': True,
        'report_dir' : thana_backtester.backup_code(filename, path, datetime.now()),
        'mode': 'dynamic' # dynamic, static
    }


thana_backtester.setup_benchmark(setting_backtest)

time.sleep(0.5)


import math

"""
    unknown date update
        change cal from pandas to xarray
    20210816 update
        debug index 
        from
            print("Daily Model: " + str((port_NAV/port_NAV.shift()-1).round(4).iloc[-1].item()*100.0) + '%') 
        to
            print("Daily Model: " + str((port_NAV/port_NAV.shift()-1).round(4).iloc[-2].item()*100.0) + '%')
    20211007
        cap max share to trade
            port_exe_shares_buy[index, :]  = np.nan_to_num(np.minimum(port_exe_shares_buy[index, :], max_share_trade_i), nan = 0.0000).astype(np.int64)
    20211008
        add dividend
"""



"""
    Is it accurate?
    it results same with worldquant backtest which validated with amibroker for FX (open_0 == close_1)
    however it is different from amibroker with equity because amibroker use execution price to calculate position size not previous close price
    .
    Why negative cash is allowed? Why don't we buy stock with the remaining cash?
    We allow negative cash because of the fact of ATO execution
    Ofcourse, we would generate signal with the remaining cash. But when we convert the target value to we use previous day's close to generate target share.
    .
    Why don't we use open price to generate target share?
    We don't know ATO price before the market is open.
    .
    Therefore, negative cash is unavoidable for ATO execution
    .
    Why do we place ATO order?
    because we would like to minimize the difference between backtest result and actual trade
    

"""

count_negative_cash = 0
baseline_history_dict = {}
rebalance_buy_back = []
min_exposure = 1
num_sig = 0
setting_sig_cal = { 
        'cash_safety_perc': np.float64(round(1.00/100.00, 5)), # 0.00 is fine because we prepare for the slippage in the main strategy cal
        'allow_negative_cash': True,
        'buy_back': True,
        'trading_value_max_perc': np.float64(round(10000000000000000000.00/100.00, 5)), # 
        'trading_value_max_exe_perc': np.float64(round(10000000000000000.00/100.00, 5)) # 
    }


trading_value = value.ffill()
trading_value_s = MA(trading_value, 5)
trading_value_m = WMA(trading_value, 8)
trading_value_l = MA(trading_value, 21)
trading_value_min = trading_value_s.apply(lambda col: np.minimum(col, trading_value_m[col.name])).astype(np.float64)
# trading_value_min = trading_value_min.apply(lambda col: np.minimum(col, trading_value_l[col.name])).astype(np.float64)

max_share_to_hold = trading_value_min/c*setting_sig_cal['trading_value_max_perc']
max_share_to_hold = max_share_to_hold.replace(np.nan, 0)
max_share_to_hold = max_share_to_hold.replace(np.inf, 0)
max_share_to_hold = max_share_to_hold.replace(-np.inf, 0)

max_share_to_trade = trading_value_min/c*setting_sig_cal['trading_value_max_exe_perc']
max_share_to_trade = max_share_to_trade.replace(np.nan, 0)
max_share_to_trade = max_share_to_trade.replace(np.inf, 0)
max_share_to_trade = max_share_to_trade.replace(-np.inf, 0)


target_weight_ = target_weight[setting_backtest['start_date']:setting_backtest['end_date']].shift()

dataframe_blueprint = target_weight_.copy()




max_share_to_hold = (max_share_to_hold[setting_backtest['start_date']:setting_backtest['end_date']].shift()).to_numpy()

max_share_to_trade = (max_share_to_trade[setting_backtest['start_date']:setting_backtest['end_date']].shift()).to_numpy()


"""
this_week = pd.DataFrame(index = c.index, data = c.index.week)
last_week = week.shift(-1)
new_week = (this_week != last_week)
new_week = series_to_df(new_week.iloc[:, 0], c)

rt_1 = target_weight_ != target_weight_.shift()
rt_2 = new_week
rebalance_trigger = ((rt_1|rt_2).astype(np.float64).sum(axis = 1) > 0).to_numpy()
target_weight_ = target_weight_.to_numpy()
"""

rebalance_trigger = ((target_weight_ != target_weight_.shift()).astype(np.float64).sum(axis = 1) > 0).to_numpy()
target_weight_ = target_weight_.to_numpy()



c_range = (c.ffill()[setting_backtest['start_date']:setting_backtest['end_date']]).to_numpy()
exe_price = o.ffill()[setting_backtest['start_date']:setting_backtest['end_date']].to_numpy()

port_share = np.zeros(c_range.shape, dtype = np.int64)
port_cash =  np.empty((c_range.shape[0],1))
port_cash[:] =  setting_backtest['initial_deposit']

port_NAV =  np.empty((c_range.shape[0],1))
port_NAV[:] =  setting_backtest['initial_deposit']


port_exe_shares_buy = np.zeros(c_range.shape)
port_exe_shares_sell = np.zeros(c_range.shape)



'''
    slip_from_prev_close = ((o/c.shift()-1)*universe_SET100)['2013':].abs()
    slip_from_prev_close.mean(axis=1).max() equals to 0.10904517769813538
    slip_from_prev_close.max(axis=1).mean() equals to 0.044310811907052994
    slip_from_prev_close.max(axis=1).max() equals to 0.3333333730697632
    slip_from_prev_close.mean(axis=1).mean() equals to 0.006574578117579222
'''

'''
    Even if the open price slips from the prev close 4.43%, we still have sufficient money to buy without reduce the number of shares to buy.

'''


datetime_list = list(dataframe_blueprint.index)
stock_as_index = np.array(dataframe_blueprint.columns)


for index, date_i in enumerate(datetime_list, start = 0):

    ## port status before the market opens
        
    shares_sell_done = port_exe_shares_sell[index-1, :] 
    shares_buy_done = port_exe_shares_buy[index-1, :] 
    
    exe_price_i_1 = exe_price[index-1, :]
    cash_i_1_start = port_cash[index-2, :]
    
    sell_txn_value = np.nansum(shares_sell_done*exe_price_i_1*(1 - setting_backtest['slip_pec']))
    buy_txn_value =  np.nansum(shares_buy_done*exe_price_i_1*(1 + setting_backtest['slip_pec']))
    
    sell_txn_fee = sell_txn_value*setting_backtest['comm_fee_perc']
    buy_txn_fee = buy_txn_value*setting_backtest['comm_fee_perc']
    
    total_pnl_i_1 = sell_txn_value - buy_txn_value - sell_txn_fee - buy_txn_fee   
    cash_i_1_end = cash_i_1_start + total_pnl_i_1
    


    equity_i_1_end = port_share[index-1, :]*c_range[index-1, :]
    equity_sum_i_1_end = np.nansum(equity_i_1_end)
    NAV_i_1_end = equity_sum_i_1_end + cash_i_1_end
    
    port_cash[index-1] = cash_i_1_end  
    port_NAV[index-1, :] = NAV_i_1_end
        
    ## error check    
    if equity_sum_i_1_end < 0:
        print('negative equity_sum: ' + str(equity_sum_i_1_end))
        raise

    min_exposure = min(min_exposure, cash_i_1_end/NAV_i_1_end)


    if(index == (len(datetime_list) - 1)):     
        print(">>>  yesterday realized cash (% NAV): " + str((total_pnl_i_1*100)/NAV_i_1_end))

    ## initialize current day status by copying the data from the previous day
    port_share[index, :] = port_share[index-1, :]
    port_NAV[index, :] = NAV_i_1_end
    port_cash[index, :] = cash_i_1_end  



    if (index % 5 == 0):
        print('-------')
        print(date_i)
        print('get prev portfolio status')
        print('equity_sum_i_1_end: ' + str(equity_sum_i_1_end))
        print('cash_i_1_end: ' + str(cash_i_1_end)) 
        print('port_NAV: ' + str(NAV_i_1_end))
        holding = stock_as_index[port_share[index, :]!=0] 
        print('holding: ' + str(holding))    
    
    if (cash_i_1_end < 0):
        count_negative_cash = count_negative_cash + 1
        print('negative cash: ' + str(cash_i_1_end))

        if not setting_sig_cal['allow_negative_cash'] and min_exposure < -0.05:
            print("too much negative cash, exposure is " + str(min_exposure))
            raise
    
    ## gen trigger

    ## gen_target_weight_
    
    if(rebalance_trigger[index]):
        # print('rebalance')
        
    
        c_i_1 = c_range[index - 1, :]
    
        NAV_i_1 = (NAV_i_1_end*(1-setting_sig_cal['cash_safety_perc']))
        share_i_1 = port_share[index - 1, :]
        share_i_0 = share_i_1.copy()
        
        target_value = NAV_i_1*(target_weight_[index, :])
        share_i_0 = np.nan_to_num((target_value/c_i_1), nan = 0.0000)


        max_share_i = max_share_to_hold[index, :]
        share_i_0 =  np.minimum(share_i_0, max_share_i)
        share_i_0 = np.nan_to_num(share_i_0, nan = 0.0000)
        share_i_0 = (share_i_0/100).round(0)*100
        share_i_0 = share_i_0.astype(np.int64)
    
        

         
        buy_exe = (share_i_0  > share_i_1).astype(np.float64)
        if(setting_sig_cal['buy_back'] == False):
            buy_exe = ((share_i_0  > share_i_1)&( (share_i_1 - 1) < 1)).astype(np.float64)
        sell_exe = (share_i_0  < share_i_1).astype(np.float64)
        hold = (share_i_0  == share_i_1).astype(np.float64)        
        
 
        port_exe_shares_buy[index, :] = np.nan_to_num((buy_exe*(share_i_0  - share_i_1)), nan = 0.0000).astype(np.int64)
        port_exe_shares_sell[index, :] = np.nan_to_num((sell_exe*(share_i_1  - share_i_0)), nan = 0.0000).astype(np.int64)  
        
        max_share_trade_i = max_share_to_trade[index, :] 
        port_exe_shares_buy[index, :]  = np.nan_to_num(np.minimum(port_exe_shares_buy[index, :], max_share_trade_i), nan = 0.0000).astype(np.int64)
        if False:
            # want to sell otherwise we don't have sufficient cash
            port_exe_shares_sell[index, :]  = np.nan_to_num(np.minimum(port_exe_shares_sell[index, :], max_share_trade_i), nan = 0.0000).astype(np.int64)
        
        port_share[index, :] = share_i_1 + port_exe_shares_buy[index, :].astype(np.int64)  - port_exe_shares_sell[index, :].astype(np.int64)
        a = index
        if np.nansum(share_i_0 < 0) > 0:
            print('negative share end of day')
            raise   


    # if(date_i == datetime(2014, 2, 7)):  #  
    #     print('raise debug')

        
# port_NAV.plot()


port_NAV = pd.DataFrame(index = dataframe_blueprint.index, data = port_NAV)
port_cash = pd.DataFrame(index = dataframe_blueprint.index, data = port_cash)


port_share = pd.DataFrame(index = dataframe_blueprint.index, columns = dataframe_blueprint.columns, data = port_share)
port_exe_shares_buy = pd.DataFrame(index = dataframe_blueprint.index, columns = dataframe_blueprint.columns, data = port_exe_shares_buy)
port_exe_shares_sell = pd.DataFrame(index = dataframe_blueprint.index, columns = dataframe_blueprint.columns, data = port_exe_shares_sell)
c_range = pd.DataFrame(index = dataframe_blueprint.index, columns = dataframe_blueprint.columns, data = c_range)





port_equity = port_NAV.iloc[:,0] - port_cash.iloc[:,0]
port_exposure = port_equity/port_NAV.iloc[:,0]


DPS_XD = load_DPX_XD(c, gs_start_backtest, gs_end_backtest)
DPS_XD_mark = DPS_XD.shift()
dividend_gain = DPS_XD_mark*port_share
pay_dividend_local = backtester_focus.pay_dividend_df
ex_date = pd.pivot_table(pay_dividend_local, columns = 'symbol', index = 'ex_date', values = 'dps').reindex(c.index)
pay_date = pd.pivot_table(pay_dividend_local, columns = 'symbol', index = 'pay_date', values = 'dps').reindex(c.index)
div_comb = data_exist_1_nan(ex_date.shift()).replace(np.nan, 0) + data_exist_1_nan(pay_date).replace(np.nan, 0)
div_comb = (div_comb).replace(0, np.nan)
convert_mark_distribution = dividend_gain*div_comb
convert_mark_distribution = lookback_q(convert_mark_distribution, -1)
port_dividend = convert_mark_distribution.sum(axis = 1).cumsum().reindex(port_NAV.index)
port_dividend = series_to_df(port_dividend, port_NAV)

port_NAV_dividend = port_NAV + port_dividend





final_NAV = port_NAV.iloc[-1].item()
final_cash = port_cash.iloc[-1].item()
before_target_share = port_share.iloc[-2].replace(0, np.nan).dropna()
final_target_share = port_share.iloc[-1].replace(0, np.nan).dropna().astype(np.int32)
final_to_buy_share = port_exe_shares_buy.iloc[-1].replace(0, np.nan).dropna().astype(np.int32)
final_to_sell_share = port_exe_shares_sell.iloc[-1].replace(0, np.nan).dropna().astype(np.int32)
final_weight = (((port_share.iloc[-1]*c_range.iloc[-2]).replace(0, np.nan).dropna())/final_NAV).round(3)


SET_2021_return = (1 + ROC(SET, 1)['2021']/100).cumprod()
SET_2021_return = (SET_2021_return - 1)*100


port_NAV_df = series_to_df(port_NAV.iloc[:, 0].reindex(c.index), c)
model_2021_return = (1 + ROC(port_NAV_df, 1)['2021']/100).cumprod()
model_2021_return = (model_2021_return - 1)*100

print('\n\n\n\n')
print('--------- summary --------- \n')
print('Model: ' + gs_name)

print('Date: ' + str(port_share.index[-1]))
print("Daily SET: " + str(ROC(SET, 1).round(4).iloc[-2].PTT.item()) + '%')
print("Daily Model: " + str(round((port_NAV/port_NAV.shift()-1).round(4).iloc[-2].item(), 4)*100.0) + '%')
  
print("Ytd SET: " + str(SET_2021_return.round(2).iloc[-2].PTT.item()) + '%')  
print("Ytd Model: " + str(model_2021_return.round().iloc[-2].PTT.item()) + '%')  

print('---------------------------- \n')

print('start date: \n' + str(gs_start_backtest))
print('start NAV: \n' + str(f"{gs_initial_deposit:,}"))
print('morning NAV: \n' + str(f"{round(final_NAV, 2):,}"))
print('morning cash: \n' + str(f"{round(final_cash, 2):,}"))


print('=======================')
print('Target Weight:')
print('=======================')
print(str(final_weight) +'\n')
print('=======================')
print('Sell Order')
print('=======================')
final_to_sell_share_str = final_to_sell_share.apply(lambda x: f"{x:,}")
print(str(final_to_sell_share_str) +'\n')
print('=======================')
print('Buy Order')
print('=======================')
final_to_buy_share_str = final_to_buy_share.apply(lambda x: f"{x:,}")
print(str(final_to_buy_share_str) +'\n')
print('=======================')
print('Before Target Share')
print('=======================')
before_target_share_str = before_target_share.apply(lambda x: f"{x:,}")
print(str(before_target_share_str) +'\n')
print('=======================')
print('Final Target Share')
print('=======================')
final_target_share_str = final_target_share.apply(lambda x: f"{x:,}")
print(str(final_target_share_str) +'\n')



paid = data_exist_1_nan(convert_mark_distribution.replace(0, np.nan))
paid_date_series = paid.sum(axis = 1)
market_size = paid_date_series.replace(0, np.nan).dropna().values
paid_date_series = paid_date_series/paid_date_series

fig, ax = plt.subplots()

curve_cum = pd.DataFrame(index = port_NAV.index)
curve_cum['NAV'] = port_NAV.iloc[:, 0].values
curve_cum['NAV + div'] = port_NAV_dividend.iloc[:, 0].values
curve_cum['distributed'] = (paid_date_series.replace(0, np.nan).reindex(port_NAV.index))*port_NAV.iloc[:, 0].values

curve_cum['high-water'] = port_NAV.cummax().iloc[:, 0].values

div = curve_cum['NAV + div'] - curve_cum['NAV']

curve_cum.plot(ax = ax)
ax.plot(curve_cum['distributed'].index, curve_cum['distributed'].values, marker='.', color = 'green')
ax.set_yscale('log')


if True:    

    baseline_history_df = pd.DataFrame.from_dict(baseline_history_dict, orient = 'index')
    
    bar_return, turnover = thana_backtester.gen_bar_return(port_share, setting_backtest, c.ffill(), o.ffill())
    import ThaiEQ_strat_tool.backtest as thana_backtester
    annual_return, MaxDD_dep, CARMDD, annual_sharpe, yearly_MaxDD_dep, NAV_change = thana_backtester.run_backtest(bar_return, setting_backtest)
    
    
    print('annual_return: ' + str(annual_return))
    print('annual_sharpe: ' + str(annual_sharpe))
    print((port_NAV.iloc[-1] - (NAV_change + setting_backtest['initial_deposit']).iloc[-2])/NAV_change.iloc[-2]*100)




# %%    

    
from ThaiEQ_strat_tool.thana_fintech import *
summary_txn_by_timeline, summary_txn_by_order, summary_txn_dict = thana_backtester.get_summary_txn_pct(port_share, bar_return, o, setting_backtest)


