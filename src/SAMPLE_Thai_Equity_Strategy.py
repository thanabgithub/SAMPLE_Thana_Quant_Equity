"""SAMPLE_Thai_Equity_Strategy

This script demostrates frameworks/processes that Thana B. use to analyze data and generate trading signal. Basically, it loads price data as input and gives target portfolio as output.
The code is not in production format but in research format for ease to debug and access data via data explorer. The final deliverable/output is a variable named "target_weight".

It consists of 4 sessions.
1. load data and validate data
2. conduct data analytics that their results are commonly being used by many sub-models. 
3. generate signal for sub-models along with their specific data analytics.
4. consolidate signal and validate final signal

REMARK: for data analysis, it uses functions from my customized library package "Thana_strat_tool"


このスクリプトは Thana B. (タナ) がデータ分析を行い売買シグナルを算出するためのフレームワーク・プロセスを示すもの。基本的に、株価を入力変数としてロードして、ターゲットポートフォリオを出力変数として出力する。
簡単にデバッグしたりデータエクスプローラー経由でデータを確認したりできるように、このコードはプロダクション形式でなくて研究形式にされている。このコードでは、最終成果物あるいは最終出力は"target_weight"という変数。

このコードは4つのセクションで構成されている。
1. データの読み込み・検証
2. 結果が複数のサブモデルに共通に使われるデータ分析を予めに行う。
3. 特殊なデータ分析を行いながら、サブモデルの売買シグナルを算出する。
4. 売買シグナルを合成させて、その結果を検証
備考： データ分析に関しては、 "Thana_strat_tool"という独自のカスタムライブラリパッケージを用いて行う。

"""

__author__ = "Thana B"
__copyright__ = "Copyright 2022,  SAMPLE_Thai_Equity_Strategy (v33)"
__email__ = "thana.b@outlook.jp"
__status__ = "research_(non-production)"

#%% -- Loading Library
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

import Thana_strat_tool.data_import_csv as data
from Thana_strat_tool.indicators import *
import Thana_strat_tool.backtest as thana_backtester
from Thana_strat_tool.thana_fintech import *
import datetime as dt

now = datetime.now()


#%% general setting
gs_name = "SAMPLE_Thai_Equity_Strategy"


gs_initial_deposit = np.float64(round(10 * 1000000, 2))

gs_commission = np.float64(round(0.075 / 100.00, 5))

gs_slippage_perc = np.float64(round(0.00 / 100, 5))
gs_slippage_ticker = int(5)

line_token = "XX"


gs_commission = (gs_commission + 0.005 / 100.000 + 0.001 / 100.000 + 0.001 / 100.000) * (1.000 + 7.000 / 100.000)


# %%

test_type = "research"

if test_type == "research":
    gs_start_backtest = "2010-01-01"
    gs_end_backtest = now



# %% load psimp
print(gs_name)
print("load psimp")

if test_type == "signal":
    send_line(line_token, "\n" + gs_name + ":   START loading psimp")

db_host = "111.222.333.111"
df_user = "ABCDE"
db_pass = "aAbBcCdDeE"
db_port = "0000"
try:
    backtester_MAI, backtester_SETMAI, backtester_SET, backtester_SET100 = get_data_backtest(
        db_host, df_user, db_pass, db_port, gs_start_backtest, gs_end_backtest
    )
except:
    print("failed loading psimp")
    if test_type == "signal":
        send_line(line_token, "FAILED to access VPN\nABORT")
        raise


# %%% load all relevant data and universe
print("load all relevant data and universe")

###################################

backtester_focus = backtester_SET

## this determine the member of stocks in datafrarme ex SETMAI: both set and mai will be included, MAI: only MAI will be included
###################################


o, h, l, c, vol, value, vwap, MKC = load_basic(backtester_focus)  # backtester_SETMAI

if test_type == "signal":
    if not data_update_check(c, gs_end_backtest, gs_name, line_token):
        send_line(line_token, "ABORT ! data hasn't been updated yet")
        raise


MAI, SET, SET100 = load_index(c)

universe_all, universe_SET100 = load_universe(c, backtester_SET100)
universe_L = load_custom_univ(MKC, 0.20, 20, value, 0.40, 20)
universe_LM = load_custom_univ(MKC, 0.50, 20, value, 0.40, 20)


ROE_1 = load_custom_q(backtester_focus, c, "roe")
TR_1 = load_custom_q(backtester_focus, c, "total_revenue")
PE = load_custom_d(backtester_focus, c, "pe")


SP, DELISTED = load_SP_DELISTED(c, gs_start_backtest, gs_end_backtest)

BANNED = DELISTED  # SP|


DPS_XD = load_DPX_XD(c, gs_start_backtest, gs_end_backtest)

DPS_XD_bf_20 = DPS_XD.bfill(limit=20)

dividend_yield = DPS_XD_bf_20 / c

# %%

check_universe_SET100(universe_SET100, test_type, line_token)

# %%
target_sig_raw = {}

# %% favorite indicator cal
print("favorite indicator cal")

price = MA(c, 7)
price_sm = MA(price, 20)
price_sm_sm = MA(price_sm, 20)

price_uptrend = (price > price_sm) & (price_sm > price_sm_sm) & bool_more(price) & bool_more(price_sm) & bool_more(price_sm_sm)
price_downtrend = (price < price_sm) & (price_sm < price_sm_sm) & bool_less(price) & bool_less(price_sm) & bool_less(price_sm_sm)
price_uptrend = bool_extend_TRUE(price_uptrend)
price_not_downtrend = ~price_downtrend

price_downtrend = bool_extend_TRUE(price_downtrend)


TN_vf = TN_v(c, vol)
TN_vf_sm = MA(TN_vf, 20)
TN_vf_sm_sm = MA(TN_vf_sm, 20)
TN_vf_uptrend = (TN_vf > TN_vf_sm) & (TN_vf_sm > TN_vf_sm_sm) & bool_more(TN_vf) & bool_more(TN_vf_sm) & bool_more(TN_vf_sm_sm)
TN_vf_downtrend = (TN_vf < TN_vf_sm) & (TN_vf_sm < TN_vf_sm_sm) & bool_less(TN_vf) & bool_less(TN_vf_sm) & bool_less(TN_vf_sm_sm)
TN_vf_uptrend = bool_extend_TRUE(TN_vf_uptrend)
TN_vf_not_downtrend = ~TN_vf_downtrend
TN_vf_downtrend = bool_extend_TRUE(TN_vf_downtrend)

TN_ff = TN_f(c, vol)
TN_ff_sm = MA(TN_ff, 20)
TN_ff_sm_sm = MA(TN_ff_sm, 20)
TN_ff_uptrend = (TN_ff > TN_ff_sm) & (TN_ff_sm > TN_ff_sm_sm) & bool_more(TN_ff) & bool_more(TN_ff_sm) & bool_more(TN_ff_sm_sm)
TN_ff_downtrend = (TN_ff < TN_ff_sm) & (TN_ff_sm < TN_ff_sm_sm)
TN_ff_uptrend = bool_extend_TRUE(TN_ff_uptrend)
TN_ff_not_downtrend = ~TN_ff_downtrend
TN_ff_downtrend = bool_extend_TRUE(TN_ff_downtrend)


NT_vf = NT_v_PLUS(c, vol)
NT_vf_sm = MA(NT_vf, 20)
NT_vf_sm_sm = MA(NT_vf_sm, 20)


NT_vf_uptrend = (NT_vf > NT_vf_sm) & (NT_vf_sm > NT_vf_sm_sm) & bool_more(NT_vf) & bool_more(NT_vf_sm) & bool_more(NT_vf_sm_sm)
NT_vf_downtrend = (NT_vf < NT_vf_sm) & (NT_vf_sm < NT_vf_sm_sm) & bool_less(NT_vf) & bool_less(NT_vf_sm) & bool_less(NT_vf_sm_sm)
NT_vf_uptrend = bool_extend_TRUE(NT_vf_uptrend)
NT_vf_downtrend = bool_extend_TRUE(NT_vf_downtrend)


NT_vf_minus = NT_v_MINUS(c, vol)
NT_vf_minus_sm = MA(NT_vf_minus, 20)
NT_vf_minus_sm_sm = MA(NT_vf_minus_sm, 20)

NT_vf_minus_downtrend = (
    (NT_vf_minus > NT_vf_minus_sm)
    & (NT_vf_minus_sm > NT_vf_minus_sm_sm)
    & bool_more(NT_vf_minus)
    & bool_more(NT_vf_minus_sm)
    & bool_more(NT_vf_minus_sm_sm)
)
NT_vf_minus_uptrend = (
    (NT_vf_minus < NT_vf_minus_sm)
    & (NT_vf_minus_sm < NT_vf_minus_sm_sm)
    & bool_less(NT_vf_minus)
    & bool_less(NT_vf_minus_sm)
    & bool_less(NT_vf_minus_sm_sm)
)
NT_vf_minus_downtrend = bool_extend_TRUE(NT_vf_minus_downtrend)
NT_vf_minus_not_downtrend = ~NT_vf_minus_downtrend
NT_vf_minus_not_uptrend = ~NT_vf_minus_uptrend

NT_vf_minus_uptrend = bool_extend_TRUE(NT_vf_minus_uptrend)

daily_return = (c - c.shift()) / c.shift()
daily_return_sm = MA(daily_return, 7)
daily_return_sm_sm = MA(daily_return_sm, 25)
daily_return_growth = daily_return_sm_sm - MA(daily_return_sm_sm, 15)

daily_SET_return = (SET - SET.shift()) / SET.shift()
daily_abs_alpha = daily_return - daily_SET_return
daily_abs_alpha_sm = MA(daily_abs_alpha, 7)


stable_market_beating = stability(daily_abs_alpha_sm, 60)
stable_price_growth = stability(daily_return_growth, 60)

# %% submodel 02
print("submodel 02")
"""
    relatively passive model: outperform in general
    - find constantly outperforming stocks

"""

universe_selected = universe_SET100.copy()


daily_return = (c - c.shift()) / c.shift()

daily_SET_return = (SET - SET.shift()) / SET.shift()
daily_abs_alpha = daily_return - daily_SET_return
daily_abs_alpha_sm = MA(daily_abs_alpha, 7)


stable_market_beating = stability(daily_abs_alpha_sm, 60)
stable_market_beating_ranking = ranking_DSC_within_universe_pct(stable_market_beating, universe_selected)


score_target_ranking = stable_market_beating_ranking


ranking_buy = 0.05
ranking_sell = 0.05

buycond = {}
buycond["01"] = (score_target_ranking.round(5) <= ranking_buy) & (MA(score_target_ranking, 20).round(5) <= ranking_buy)


sellcond = {}
sellcond["01"] = (score_target_ranking.round(5) > ranking_sell) & (MA(score_target_ranking, 20).round(5) > ranking_sell)


BUY = AND_dict(buycond)
SELL = OR_dict(sellcond)

SELL_adj = SELL.copy()
SELL_adj[BANNED.shift(-1) == True] = True
SELL_adj[BANNED.shift(-2) == True] = True


BUY_adj = BUY.copy()
BUY_adj[BANNED.shift(-1) == True] = False
BUY_adj[BANNED.shift(-2) == True] = False

BUY[SELL == True] = False

stability_return = stability(ROC(c, 1), 60)
stability_return_z = z_score(stability_return, 20)
stability_return_uptrend = (stability_return_z.round(5) > 0.50).astype(np.float64)

stability_return_MS_raw = MA(MA(stability_return_uptrend, 60), 60)
stability_return_MS = perK(stability_return_MS_raw, stability_return_MS_raw, stability_return_MS_raw, 240, 20) / 100
# display_comparison_2_axis('PTT', c, 'c', stability_return_MS, 'mean stability_return_MS', '2012-01-01')
pos_cal = normalize_row_by_perK(stability_return_MS, universe_selected)
pos_cal_mean = mean_within_universe(pos_cal, universe_selected)
pos_cal[pos_cal.isnull() == True] = pos_cal_mean

target_sig_raw["submodel_02"] = one_zero_gen_frm_BUYSELL(BUY, SELL, universe_selected) * pos_cal
target_sig_raw["submodel_02"] = (target_sig_raw["submodel_02"]).replace(np.nan, 0)


# %% submodel 03
print("submodel 03")
"""
    relatively passive model: outperform in general

"""

universe_selected = universe_LM.copy()


daily_return = (c - c.shift()) / c.shift()

daily_SET_return = (SET - SET.shift()) / SET.shift()
daily_abs_alpha = daily_return - daily_SET_return
daily_abs_alpha_sm = MA(daily_abs_alpha, 7)


stable_market_beating = stability(daily_abs_alpha_sm, 60)
stable_market_beating_ranking = ranking_DSC_within_universe_pct(stable_market_beating, universe_selected)


score_target_ranking = stable_market_beating_ranking


ranking_buy = 0.10
ranking_sell = 0.10


stable_price_growth_ranking = ranking_DSC_within_universe_pct(stable_price_growth, universe_selected)


buycond = {}
buycond["01"] = (score_target_ranking.round(5) <= ranking_buy) & (MA(score_target_ranking, 60).round(5) <= ranking_buy)
buycond["02"] = NT_vf_uptrend & (TN_ff_uptrend & TN_vf_uptrend) & price_uptrend

sellcond = {}
sellcond["01"] = (score_target_ranking.round(5) > ranking_sell) & (MA(score_target_ranking, 10).round(5) > ranking_sell)
sellcond["02"] = TN_ff_downtrend | NT_vf_downtrend  # TN_vf_downtrend


BUY = AND_dict(buycond)
SELL = OR_dict(sellcond)

SELL_adj = SELL.copy()
SELL_adj[BANNED.shift(-1) == True] = True
SELL_adj[BANNED.shift(-2) == True] = True


BUY_adj = BUY.copy()
BUY_adj[BANNED.shift(-1) == True] = False
BUY_adj[BANNED.shift(-2) == True] = False

BUY[SELL == True] = False


daily_return = ROC(c, 1)

pv = (c - vwap) * vol
z_score_pv = z_score(pv, 60)
z_score_pv_max = z_score_pv.max(axis=1)
z_score_pv_max = series_to_df(z_score_pv_max, c)
z_score_pv = -1 * z_score_pv + z_score_pv_max


pos_cal = z_score_pv
pos_cal = normalize_row_by_perK(pos_cal, universe_selected)
pos_cal_mean = mean_within_universe(pos_cal, universe_selected)
pos_cal[pos_cal.isnull() == True] = pos_cal_mean


target_sig_raw["submodel_03"] = one_zero_gen_frm_BUYSELL(BUY, SELL, universe_selected) * pos_cal
target_sig_raw["submodel_03"] = (target_sig_raw["submodel_03"]).replace(np.nan, 0)


# %% SET market timing cal
print("SET market timing cal")

trade_universe = universe_all.copy()


SET_deta = ROC(SET, 5)
SET_deta_z = MA(z_score(SET_deta, 120), 5)
SET_delta_BUY = (SET_deta_z.round(5) > 1.0).astype(np.float64)
SET_delta_BUY = MA(SET_delta_BUY, 10)
SET_delta_BUY = MA(SET_delta_BUY, 10)
SET_delta_coef = 0.5 + 0.5 * SET_delta_BUY


SET_stability = stability(ROC(SET, 1), 60)
SET_stability_z = z_score(SET_stability, 20)
SET_stability_uptrend = (SET_stability_z.round(5) > 0.50).astype(np.float64)
SET_stability_uptrend_sm = MA(SET_stability_uptrend, 7)
SET_stability_uptrend_sm_sm = MA(MA(MA(SET_stability_uptrend, 3), 3), 3)

SET_stability_MS_raw = MA(MA(SET_stability_uptrend, 60), 60)
SET_stability_MS = perK(SET_stability_MS_raw, SET_stability_MS_raw, SET_stability_MS_raw, 240, 60) / 100


# %% rebalance setting and backtest
print("rebalance setting and backtest ")



universe_selected = universe_LM.copy()
universe_selected_num_series = universe_selected.sum(axis=1)
universe_selected_num = series_to_df(universe_selected_num_series, c)

ff_in = ((NT_vf_uptrend & TN_ff_uptrend & TN_vf_uptrend & price_uptrend).astype(np.float64) * universe_selected).sum(
    axis=1
) / universe_selected_num_series


ff_in_df = series_to_df(ff_in, c)
ff_in_stoch = perK(ff_in_df, ff_in_df, ff_in_df, 120, 20) / 100

stoch_K = perK(h, l, c, 60, 7)
average_stoch_K = mean_within_universe(stoch_K, universe_selected)
average_stoch_K_stock_K = perK(average_stoch_K, average_stoch_K, average_stoch_K, 120, 20) / 100

basic_coeff = {
    "submodel_02": 0.50, 
    "submodel_03": 0.25, 
}


coef_trd_01_param_02 = 0.50  # need this to reduce 2012 drawdown
SET_stability_coef_02 = 1 - coef_trd_01_param_02 + coef_trd_01_param_02 * SET_stability_uptrend_sm_sm

coef_trd_01_param_04 = 0.20
SET_stability_coef_04 = 1 - coef_trd_01_param_04 + coef_trd_01_param_04 * SET_stability_uptrend_sm_sm

coef_trd_01_param_05 = 0.50
SET_stability_coef_05 = 1 - coef_trd_01_param_05 + coef_trd_01_param_05 * SET_stability_uptrend_sm_sm


coef_MS_01_param = 0.10  #
SET_MS_coef = 1 - coef_MS_01_param + coef_MS_01_param * SET_stability_MS

# SET_stability_coef
dynamic_coeff = {"submodel_02": init_df(c, 1), "submodel_03": init_df(c, 1), "submodel_04": init_df(c, 1), "submodel_05": init_df(c, 1)}  #  # look

init_weight_sub = 1.00


target_sig_raw_list = list(target_sig_raw.keys())

execute_submodel = basic_coeff.keys()


target_sig_raw_ = target_sig_raw.copy()
for sig_name in execute_submodel:
    target_sig_raw_[sig_name] = target_sig_raw_[sig_name] / series_to_df(target_sig_raw_[sig_name].sum(axis=1), c)
    target_sig_raw_[sig_name] = cap_maximal_value(target_sig_raw_[sig_name], init_weight_sub)
    target_sig_raw_[sig_name] = target_sig_raw_[sig_name].replace(np.nan, 0)


target_sig = init_df(c, 0)
total_coeff = 0
for sig_name in execute_submodel:
    target_sig = target_sig + target_sig_raw_[sig_name] * basic_coeff[sig_name] * dynamic_coeff[sig_name]
    total_coeff = total_coeff + basic_coeff[sig_name]
target_sig = target_sig / total_coeff


target_weight = (target_sig * trade_universe).replace(np.nan, 0)


init_weight_before_buffer = 0.30
init_weight = init_weight_before_buffer * 0.99

over_cap_bool = (target_weight > init_weight).astype(np.float64)
total_exessive = (over_cap_bool * (target_weight - init_weight)).sum(axis=1)
under_cap_bool = ((target_weight <= init_weight) & (target_weight > 0.00000)).astype(np.float64)
under_cap_count = under_cap_bool.sum(axis=1)
add_distribute = series_to_df((total_exessive / under_cap_count).replace(np.nan, 0).replace(np.inf, 0), c) * (under_cap_bool.replace(np.nan, 0))

target_weight = cap_maximal_value(target_weight, init_weight) + add_distribute

target_weight = clean_weight_over_1(target_weight).replace(np.nan, 0)



# %%

check_universe_SET100(universe_SET100, test_type, line_token)
check_num_of_sig(target_weight, 30, test_type, line_token)
check_over_pos(target_weight, init_weight_before_buffer, test_type, line_token)
check_total_pos(target_weight, test_type, line_token)
check_pos_dim(target_weight, universe_all, test_type, line_token)
