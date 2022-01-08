# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 12:51:52 2020

@author: ADMIN
"""

import pandas as pd
import talib as ta
import numpy as np
from datetime import datetime
import statsmodels.api as sm
from statsmodels import regression
import timeit

import matplotlib.pyplot as plt


def setup_benchmark(setting_backtest_: dict):
    if type(setting_backtest_["start_date"]) != str:
        setting_backtest_["start_date"] = setting_backtest_["start_date"].strftime(
            "%Y-%m-%d"
        )
    if type(setting_backtest_["end_date"]) != str:
        setting_backtest_["end_date"] = setting_backtest_["end_date"].strftime(
            "%Y-%m-%d"
        )

    try:
        benchmark_series = setting_backtest_["benchmark"]["PTT"]
    except:
        benchmark_series = setting_backtest_["benchmark"].iloc[:, 0]
    try:
        year_range = [
            *range(
                int(setting_backtest_["start_date"].strftime("%Y-%m-%d")[0:4]),
                int(setting_backtest_["end_date"].strftime("%Y-%m-%d")[0:4]) + 1,
                1,
            )
        ]
    except:
        year_range = [
            *range(
                int(setting_backtest_["start_date"][0:4]),
                int(setting_backtest_["end_date"][0:4]) + 1,
                1,
            )
        ]

    benchmark_new_high = pd.Series(index=benchmark_series.index, dtype=np.float64)
    benchmark_new_high = benchmark_series[
        setting_backtest_["start_date"] : setting_backtest_["end_date"]
    ].cummax()
    benchmark_drawdown = pd.Series(index=benchmark_series.index, dtype=np.float64)
    benchmark_drawdown = benchmark_series - benchmark_new_high  # new_high - NAV
    benchmark_drawdown = benchmark_drawdown / benchmark_new_high
    benchmark_drawdown.dropna(inplace=True)

    benchmark_drawdown = benchmark_drawdown.reindex(
        pd.DatetimeIndex(benchmark_drawdown.index)
    )
    benchmark_series = benchmark_series.reindex(
        pd.DatetimeIndex(benchmark_series.index)
    )

    yearly_MaxDD_dep = {}

    for year_int in year_range:
        yearly_MaxDD_dep[year_int] = benchmark_drawdown[str(year_int)].min() * 100.00

    # yearly_benchmark_MaxDD_frm_NY = {}
    # for year_int in year_range:
    #     yearly_benchmark_MaxDD_frm_NY[year_int] = drawdown_percent_benchmark[str(year_int)].min()*100.

    yearly_bechmark_return = {}
    for year_int in year_range:
        yearly_bechmark_return[year_int] = (
            (benchmark_series[str(year_int)][-1] - benchmark_series[str(year_int)][0])
            / benchmark_series[str(year_int)][0]
            * 100.0
        )

    setting_backtest_["yearly_benchmark_return"] = yearly_bechmark_return
    setting_backtest_["yearly_benchmark_MaxDD"] = yearly_MaxDD_dep


def gen_bar_return(
    target_shares_: pd.DataFrame,
    setting_backtest_: dict,
    c_: pd.DataFrame,
    exe_: pd.DataFrame,
):

    target_shares_ = (round(target_shares_ / 100.00) * 100.00).astype(np.float32)
    # test_target_shares = weight.astype(np.float32)*100

    ## fixed lot ok
    ## commission ok
    ## execution price ( + slip) ok
    ## multi stock
    ## vwap
    slip = setting_backtest_["slip_pec"]
    exe_price_df = exe_.copy()
    exe_price_df[target_shares_.diff() > 0.0] = exe_.copy() * (1.0 + slip)
    exe_price_df[target_shares_.diff() < 0.0] = exe_.copy() * (1.0 - slip)

    # exe_price_df = exe_price_df['ADVANC']

    pnl_base_hld = (target_shares_.shift(+1) * c_.diff()).fillna(0)
    pnl_base_trd = (target_shares_.diff() * (c_ - exe_price_df)).fillna(0)

    commission_fee_by_perc = setting_backtest_["comm_fee_perc"]

    trade = (
        (target_shares_.diff() != 0).fillna(0) * target_shares_.diff() * exe_price_df
    )
    trade.iloc[0] = 0
    commission_fee = trade.abs() * commission_fee_by_perc

    bar_return_ = ((pnl_base_hld + pnl_base_trd) - commission_fee).replace(np.nan, 0)
    turnover_ = (
        (
            trade.abs()[setting_backtest_["start_date"] : setting_backtest_["end_date"]]
            .cumsum()
            .iloc[-1]
            / 2.00
        ).sum()
    ) / setting_backtest_["initial_deposit"]
    return bar_return_, turnover_  # , exe_, exe_price_df


def run_backtest(bar_return_: pd.DataFrame, setting_backtest_: dict):
    NAV_change = (
        bar_return_[setting_backtest_["start_date"] : setting_backtest_["end_date"]]
        .sum(axis=1)
        .cumsum()
    )

    # bar_return = hourly_return_input[start_date:end_date].ffill()
    # num_bar = bar_return.shape[0]
    # bar_avg_return = ta.SMA(bar_return.values, num_bar)
    # bar_sd_return = ta.STDDEV(bar_return.values, num_bar)

    # NAV_change = NAV_change
    NAV = setting_backtest_["initial_deposit"] + NAV_change

    # NAV.plot()

    new_high = pd.Series(index=NAV_change.index, dtype=np.float64)
    new_high = NAV[
        setting_backtest_["start_date"] : setting_backtest_["end_date"]
    ].cummax()
    drawdown = pd.Series(index=NAV_change.index, dtype=np.float64)
    drawdown = NAV - new_high  # new_high - NAV

    drawdown_percent_NAV = drawdown / new_high
    drawdown_percent_NAV.replace([np.inf, -np.inf], np.nan, inplace=True)
    drawdown_percent_NAV.fillna(method="bfill", inplace=True)

    drawdown_percent_dep = drawdown / setting_backtest_["initial_deposit"]
    drawdown_percent_dep.replace([np.inf, -np.inf], np.nan, inplace=True)
    drawdown_percent_dep.fillna(method="bfill", inplace=True)
    # drawdown_percent_dep.plot()

    ## display NAV

    if setting_backtest_["display"]:

        fig, ax = plt.subplots()
        # make a plot
        ax.plot(NAV.index, NAV.values, color="blue", label="model", linewidth=1.00)

        # set x-axis label
        # ax.set_xlabel("year",fontsize=14)
        # set y-axis label
        ax.set_ylabel("NAV", color="blue", fontsize=14)
        #     if(False):
        # #         ax.set_ylim([initial_deposit *0.8, max(NAV.iloc[-1]*1.05, initial_deposit *2.2)])
        #         ax.set_ylim([initial_deposit *0.8, max(NAV.iloc[-1].item()*1.05, initial_deposit *4.2)])
        #     else:
        #         # ax.set_ylim([NAV.min()*.999, min(NAV.iloc[-1]*1.01, initial_deposit *2.2)])
        #         ax.set_ylim([float(NAV.min()*.999), float(NAV.max()*1.05)])     # NAV.iloc[-1].item()
        try:
            benchmark_series = setting_backtest_["benchmark"]["PTT"][
                setting_backtest_["start_date"] : setting_backtest_["end_date"]
            ]
        except:
            benchmark_series = setting_backtest_["benchmark"].iloc[:, 0][
                setting_backtest_["start_date"] : setting_backtest_["end_date"]
            ]
        benchmark_series = benchmark_series.reindex(
            pd.DatetimeIndex(benchmark_series.index)
        )
        benchmark_series = (
            benchmark_series
            / (benchmark_series.iloc[0])
            * setting_backtest_["initial_deposit"]
        )
        ax.plot(
            benchmark_series.index,
            benchmark_series.values,
            color="black",
            label="benchmark",
            linewidth=0.50,
        )
        ax.tick_params("y", colors="blue")
        # twin object for two different y-axis on the sample plot
        ax2 = ax.twinx()
        # make a plot with different y-axis using second axis object
        if setting_backtest_["mode"] == "dynamic":
            ax2.plot(
                drawdown_percent_NAV.index,
                drawdown_percent_NAV.values,
                color="red",
                label="DD_NAV",
                alpha=0.3,
            )
        elif setting_backtest_["mode"] == "static":
            ax2.plot(
                drawdown_percent_dep.index,
                drawdown_percent_dep.values,
                color="red",
                label="DD_dep",
                alpha=0.3,
            )
        # ax2.plot(drawdown_percent_NAV.index, drawdown_percent_NAV.values,color="orange", label = 'DD_NAV')
        ax2.set_ylabel("drawdown_percent", color="red", fontsize=14)
        ax2.legend(loc="lower right")
        ax2.tick_params("y", colors="red")
        fig.set_size_inches([9, 6])
        # if(False):
        #     ax2.set_ylim([-1.00, 0.00])
        # else:
        #     ax2.set_ylim([max(drawdown_percent_dep.values.min()*.50, -0.25), 0.00])

        # ax3=ax.twinx()
        # ax3.plot(benchmark_series.index, benchmark_series.values, color="black", label = 'benchmark', linewidth=0.50)

        # ax3.legend(loc ="lower right")
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)
        #
        plt.show()
        if setting_backtest_["save_fig"]:
            plt.savefig(
                setting_backtest_["report_dir"] + "\\NAV.png",
                bbox_inches="tight",
                dpi=100,
            )

    ## display profit bar
    if type(setting_backtest_["start_date"]) == str:

        start_date = setting_backtest_["start_date"]
        start_year = start_date[0:4]

        start_date = datetime.strptime(setting_backtest_["start_date"], "%Y-%m-%d")

    else:
        start_date = setting_backtest_["start_date"]
        start_year = start_date.strftime("%Y-%m-%d")[0:4]
    if type(setting_backtest_["end_date"]) == str:

        end_date = setting_backtest_["end_date"]
        end_year = end_date[0:4]

        end_date = datetime.strptime(
            setting_backtest_["end_date"], "%Y-%m-%d"
        )  # datetime.datetime(2010,1,1)

    else:
        end_date = setting_backtest_["end_date"]
        end_year = end_date.strftime("%Y-%m-%d")[0:4]

    num_months = (end_date.year - start_date.year) * 12 + (
        end_date.month - start_date.month
    )

    bar_return_ = bar_return_.sum(axis=1)
    # start - close for opt

    year_range = [*range(int(start_year), int(end_year) + 1, 1)]

    yearly_return = {}

    year_series = pd.Series(index=bar_return_.index, data=bar_return_.index).dt.year
    last_day_of_year = year_series != year_series.shift(-1)  # &quarterly
    last_day_of_year = last_day_of_year.astype(np.float32)
    last_day_of_year = last_day_of_year.replace(0, np.nan)
    last_day_of_year = last_day_of_year * NAV
    prev_year_NAV = last_day_of_year.ffill()

    for year_int in year_range:
        if setting_backtest_["mode"] == "dynamic":
            yearly_return[year_int] = (
                bar_return_[str(year_int)].sum()
                / prev_year_NAV[str(year_int)].iloc[0].item()
                * 100.00
            )
        elif setting_backtest_["mode"] == "static":
            yearly_return[year_int] = (
                bar_return_[str(year_int)].sum()
                / setting_backtest_["initial_deposit"]
                * 100.00
            )

    yearly_return_np = np.array(list(yearly_return.values()))

    try:
        yearly_sd_np = ta.STDDEV(yearly_return_np, int(num_months / 12))
        yearly_sd_np = yearly_sd_np[-1]
    except:
        yearly_sd_np = 1

    NAV_daily_return = NAV / (NAV.shift()) - 1
    NAV_daily_return_np = NAV_daily_return.iloc[1:].astype(np.float64).values
    num_of_year = (end_date - start_date).days / 365.25
    bar_per_year = len(NAV_daily_return) / num_of_year
    yearly_sd_np = (ta.STDDEV(NAV_daily_return_np, len(NAV_daily_return_np))[-1]) * (
        (bar_per_year) ** (1.0 / 2.0)
    )

    yearly_MaxDD_dep = {}

    for year_int in year_range:
        if setting_backtest_["mode"] == "dynamic":
            yearly_MaxDD_dep[year_int] = (
                drawdown[str(year_int)].min()
                / new_high[str(year_int)].iloc[0].item()
                * 100.00
            )
        elif setting_backtest_["mode"] == "static":
            yearly_MaxDD_dep[year_int] = (
                drawdown[str(year_int)].min()
                / setting_backtest_["initial_deposit"]
                * 100.00
            )
    yearly_return = pd.DataFrame.from_dict(yearly_return, orient="index")
    yearly_MaxDD_dep = pd.DataFrame.from_dict(yearly_MaxDD_dep, orient="index")
    yearly_benchmark_return = pd.DataFrame.from_dict(
        setting_backtest_["yearly_benchmark_return"], orient="index"
    )
    yearly_bechmark_drawdown = pd.DataFrame.from_dict(
        setting_backtest_["yearly_benchmark_MaxDD"], orient="index"
    )

    yearly_profit_MaxDD_dep = pd.DataFrame()
    yearly_profit_MaxDD_dep["model_profit"] = yearly_return.iloc[:, 0]
    yearly_profit_MaxDD_dep["model_MaxDD_dep"] = yearly_MaxDD_dep.iloc[:, 0]
    yearly_profit_MaxDD_dep["_"] = 0.0
    yearly_profit_MaxDD_dep["benchmark_profit"] = yearly_benchmark_return.iloc[:, 0]
    yearly_profit_MaxDD_dep["benchmark_MaxDD"] = yearly_bechmark_drawdown.iloc[:, 0]
    color_dict = {
        "model_profit": "darkgreen",
        "model_MaxDD_dep": "red",
        "benchmark_profit": "cornflowerblue",
        "_": "grey",
        "benchmark_MaxDD": "orange",
    }
    # color_list = ['green', 'red', 'blue', 'orange']
    if setting_backtest_["display"]:
        # yearly_profit_MaxDD_dep.plot(kind="bar", legend=True, colors = color_list)
        yearly_profit_MaxDD_dep.plot(
            color=[
                color_dict.get(x, "#333333") for x in yearly_profit_MaxDD_dep.columns
            ],
            kind="bar",
            legend=True,
        )
        plt.xticks(rotation=30, horizontalalignment="center")
        plt.xlabel("year")
        plt.ylabel("MaxDD_dep   -    % Return")
        plt.show()

        if setting_backtest_["save_fig"]:
            plt.savefig(
                setting_backtest_["report_dir"] + "\\profit_bar.png",
                bbox_inches="tight",
                dpi=100,
            )

    # end - close for opt

    ## cal summary
    total_return = round(
        (NAV.iloc[-1].item() / setting_backtest_["initial_deposit"] - 1.00) * 100, 2
    )
    annual_return = round(total_return / num_months * 12.0, 2)

    if setting_backtest_["mode"] == "dynamic":
        annual_return = (
            NAV.iloc[-1].item() / setting_backtest_["initial_deposit"]
        ) ** (12 / (num_months)) - 1

    MaxDD_dep = round(drawdown_percent_dep.min() * 100, 2)
    MaxDD_NAV = round(drawdown_percent_NAV.min() * 100, 2)
    CARMDD = round(annual_return / MaxDD_dep * -1.0, 2)
    monthly_return = round(total_return / num_months, 2)
    annual_sharpe = round(annual_return / yearly_sd_np, 2)

    return annual_return, MaxDD_dep, CARMDD, annual_sharpe, yearly_MaxDD_dep, NAV_change


def get_summary_txn(
    target_shares_: pd.DataFrame, bar_return_: pd.DataFrame, setting_backtest_: dict
):
    start_date_ = setting_backtest_["start_date"]
    end_date_ = setting_backtest_["end_date"]

    summary_txn_by_timeline_ = target_shares_.copy().replace(0, np.nan)
    # summary_txn_by_timeline_.dropna(how = 'all', inplace = True)
    summary_txn_by_timeline_.dropna(axis=1, how="all", inplace=True)
    summary_txn_by_timeline_.replace(np.nan, 0, inplace=True)

    index_ = summary_txn_by_timeline_.index
    txn_dict = {}
    txn_num = 0

    summary_txn_stock = summary_txn_by_timeline_.columns.tolist()
    bar_return_ = bar_return_.copy()
    # print(summary_txn_by_timeline_.index)
    # print(bar_return_.index)

    for stock in summary_txn_stock:
        bar_return_mini = bar_return_.loc[:, stock]
        summary_txn_mini = summary_txn_by_timeline_.loc[:, stock]

        for row in range(1, len(summary_txn_mini), 1):

            if (summary_txn_mini.iloc[row] > 0.0) and (
                summary_txn_mini.iloc[row - 1] == 0.0
            ):
                txn_dict[txn_num] = {}
                txn_dict[txn_num]["symbol"] = summary_txn_mini.name
                txn_dict[txn_num]["entry_date"] = index_[row]

            if (summary_txn_mini.iloc[row] == 0.0) and (
                summary_txn_mini.iloc[row - 1] > 0.0
            ):
                txn_dict[txn_num]["exit_date"] = index_[row]
                txn_dict[txn_num]["profit"] = bar_return_mini[
                    txn_dict[txn_num]["entry_date"] : txn_dict[txn_num]["exit_date"]
                ].sum()

                txn_num = txn_num + 1
        if (txn_num in txn_dict) and (not ("exit_date" in txn_dict[txn_num])):
            # print('in the shit')
            txn_dict[txn_num]["exit_date"] = index_[len(summary_txn_mini) - 1]
            txn_dict[txn_num]["profit"] = bar_return_mini[
                txn_dict[txn_num]["entry_date"] : txn_dict[txn_num]["exit_date"]
            ].sum()
            txn_num = txn_num + 1

    summary_txn_by_order_ = pd.DataFrame.from_dict(txn_dict, orient="index")
    summary_txn_by_order_.sort_values(by=["entry_date"], inplace=True)
    summary_txn_by_order_.reset_index(inplace=True)
    summary_txn_by_order_.drop("index", inplace=True, axis=1)

    summary_txn_by_timeline_.to_csv(
        setting_backtest_["report_dir"] + "\\summary_txn_by_timeline.csv"
    )
    summary_txn_by_order_.to_csv(
        setting_backtest_["report_dir"] + "\\summary_txn_by_order.csv"
    )

    return summary_txn_by_timeline_, summary_txn_by_order_, txn_dict


def get_summary_txn_pct(
    target_shares_: pd.DataFrame,
    bar_return_: pd.DataFrame,
    o_: pd.DataFrame,
    setting_backtest_: dict,
):
    start_date_ = setting_backtest_["start_date"]
    end_date_ = setting_backtest_["end_date"]

    summary_txn_by_timeline_ = (
        target_shares_[start_date_:end_date_].copy().replace(0, np.nan)
    )
    summary_value_by_timeline_ = (
        target_shares_ * o_.ffill()[start_date_:end_date_]
    ).replace(0, np.nan)

    execution_price = o_.ffill()[start_date_:end_date_].replace(0, np.nan)

    # summary_txn_by_timeline_.dropna(how = 'all', inplace = True)
    summary_txn_by_timeline_.dropna(axis=1, how="all", inplace=True)
    summary_txn_by_timeline_.replace(np.nan, 0, inplace=True)

    summary_value_by_timeline_.dropna(axis=1, how="all", inplace=True)
    summary_value_by_timeline_.replace(np.nan, 0, inplace=True)

    index_ = summary_txn_by_timeline_.index
    txn_dict = {}
    txn_num = 0

    summary_txn_stock = summary_txn_by_timeline_.columns.tolist()
    bar_return_ = bar_return_[start_date_:end_date_].copy()
    # print(summary_txn_by_timeline_.index)
    # print(bar_return_.index)

    for stock in summary_txn_stock:
        bar_return_mini = bar_return_.loc[:, stock]
        summary_txn_mini = summary_txn_by_timeline_.loc[:, stock]

        for row in range(1, len(summary_txn_mini), 1):

            if (summary_txn_mini.iloc[row] > 1.0) and (
                summary_txn_mini.iloc[row - 1] < 1.0
            ):
                txn_dict[txn_num] = {}
                txn_dict[txn_num]["symbol"] = summary_txn_mini.name
                txn_dict[txn_num]["entry_date"] = index_[row]
                txn_dict[txn_num]["entry_price"] = execution_price[stock][row]
                txn_dict[txn_num]["buy_value"] = (
                    execution_price[stock][row] * summary_txn_mini.iloc[row]
                )
                # print(txn_dict[txn_num]['entry_value'])
            if txn_num in txn_dict.keys():
                if summary_txn_mini.iloc[row] - summary_txn_mini.iloc[row - 1] > 1:
                    # buy
                    txn_dict[txn_num]["buy_value"] = txn_dict[txn_num][
                        "buy_value"
                    ] + execution_price[stock][row] * (
                        summary_txn_mini.iloc[row] - summary_txn_mini.iloc[row - 1]
                    )

            if (summary_txn_mini.iloc[row] < 1.0) and (
                summary_txn_mini.iloc[row - 1] > 1.0
            ):

                txn_dict[txn_num]["exit_date"] = index_[row]
                txn_dict[txn_num]["exit_price"] = execution_price[stock][row]
                txn_dict[txn_num]["average_share"] = target_shares_[stock][
                    txn_dict[txn_num]["entry_date"] : txn_dict[txn_num]["exit_date"]
                ].mean()
                txn_dict[txn_num]["net_pnl"] = bar_return_mini[
                    txn_dict[txn_num]["entry_date"] : txn_dict[txn_num]["exit_date"]
                ].sum()

                # print(txn_dict[txn_num]['profit_pct'] )
                txn_num = txn_num + 1
        if (txn_num in txn_dict) and (not ("exit_date" in txn_dict[txn_num])):
            # print('in the shit')
            txn_dict[txn_num]["exit_date"] = index_[len(summary_txn_mini) - 1]
            txn_dict[txn_num]["exit_price"] = execution_price[stock][row]
            txn_dict[txn_num]["average_share"] = target_shares_[stock][
                txn_dict[txn_num]["entry_date"] : txn_dict[txn_num]["exit_date"]
            ].mean()
            txn_dict[txn_num]["net_pnl"] = bar_return_mini[
                txn_dict[txn_num]["entry_date"] : txn_dict[txn_num]["exit_date"]
            ].sum()
            # txn_dict[txn_num]['net_pnl'] = txn_dict[txn_num]['profit']
            # print(txn_dict[txn_num]['profit_pct'] )
            txn_num = txn_num + 1

    summary_txn_by_order_ = pd.DataFrame.from_dict(txn_dict, orient="index")
    summary_txn_by_order_.sort_values(by=["entry_date"], inplace=True)
    summary_txn_by_order_.reset_index(inplace=True)
    summary_txn_by_order_.drop("index", inplace=True, axis=1)

    summary_txn_by_timeline_.to_csv(
        setting_backtest_["report_dir"] + "\\summary_txn_by_timeline.csv"
    )
    summary_txn_by_order_.to_csv(
        setting_backtest_["report_dir"] + "\\summary_txn_by_order.csv"
    )

    return summary_txn_by_timeline_, summary_txn_by_order_, txn_dict


import shutil
import os


def backup_code(filename_: str, path_: str, time):

    folder_name_time_format = time.strftime("%Y%m%d%H%M%S")
    report_directory_ = (
        path_ + "\\report\\" + filename_ + "_" + str(folder_name_time_format)
    )
    if not os.path.exists(report_directory_):
        os.mkdir(report_directory_)

    original = path_ + "\\" + filename_ + ".py"
    target = report_directory_ + "\\code_backup.py"

    shutil.copyfile(original, target)
    return report_directory_
