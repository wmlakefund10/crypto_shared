import requests
import json
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import time
import pybit
import utility
import os
import sys
from multiprocessing import Pool

# old API
def get_bar_data(symbol, interval, startTime, exchange='bybit'):
    if exchange.lower() == 'bybit':
        url = "https://api.bybit.com/public/linear/kline"
    else:
        url = "https://api.bybit.com/public/linear/mark-price-kline"
    req_params = {"symbol": symbol, 'interval': interval, 'from': startTime}
    df = pd.DataFrame(json.loads(requests.get(url, params=req_params).text)['result'])
    if (len(df.index) == 0):
        return None
    if exchange == 'bybit':
        df.index = [dt.datetime.fromtimestamp(x) for x in df.open_time]
    else:
        df.index = [dt.datetime.fromtimestamp(x) for x in df.start_at]
    return df


def get_bar_data_v5(category, symbol, interval, startTime=None):
    url = "https://api.bybit.com/v5/market/kline"
    if startTime is None:
        req_params = {"category": category, "symbol": symbol, 'interval': interval}
    else:
        req_params = {"category": category, "symbol": symbol, 'interval': interval, 'start': startTime}
    df = pd.DataFrame(json.loads(requests.get(url, params=req_params).text)['result'])
    if (len(df.index) == 0):
        return None
    df = pd.concat([df['symbol'], pd.DataFrame(list(df['list']),
                                               columns=['startTime', 'openPrice', 'highPrice', 'lowPrice', 'closePrice',
                                                        'volume', 'turnover'])], axis=1)
    df.index = [dt.datetime.fromtimestamp(int(x) / 1000) for x in df.startTime]
    df = df.iloc[::-1]
    return df


def find_sdate(category, symbol, interval, dates=None):
    # this will give you a list containing all of the dates
    if dates is None:
        sdate = dt.date(2020, 1, 15)
        edate = dt.date.today()
        dates = [dt.datetime.strftime(sdate + dt.timedelta(days=x), '%Y-%m-%d') for x in
                 range((edate - sdate).days + 1)]

    sindex = 0
    tindex = len(dates) - 1
    found_sdate = False
    while sindex < tindex and (not found_sdate):
        mindex = int((sindex + tindex) / 2)
        tmp = get_bar_data_v5(category, symbol, interval, utility.dt_to_millsec(dates[mindex]))
        if tmp is None or (len(tmp) == 0):
            sindex = mindex + 1
        elif len(tmp) >= 200:
            tindex = mindex
        else:
            found_sdate = True
    if not found_sdate:
        return None
    else:
        return dates[mindex]


def query_histdata(symbol, interval0, sdate, exchange='bybit'):
    # interval param: 1 3 5 15 30 60 120 240 360 720 "D" "M" "W"
    if interval0 == 'D':
        interval = 1440
    elif interval0 == 'W':
        interval = 10080
    elif interval0 == 'M':
        interval = 44640
    else:
        interval = interval0
    starttime = dt.datetime.strptime(sdate, '%Y-%m-%d')
    df_list = []
    last_datetime = str(int(starttime.timestamp()))
    while True:
        print(dt.datetime.fromtimestamp((int(last_datetime))))
        new_df = get_bar_data(symbol, interval, last_datetime, exchange=exchange)
        if new_df is None:
            break
        df_list.append(new_df)
        if exchange == 'bybit':
            print(dt.datetime.fromtimestamp(new_df['open_time'].max()))
            last_datetime = new_df['open_time'].max() + 60 * interval
        else:
            last_datetime = new_df['start_at'].max() + 60 * interval
    df = pd.concat(df_list)
    return df


def query_histdata_v5(category, symbol, interval, sdate=None):
    # interval param: 1 3 5 15 30 60 120 240 360 720 "D" "M" "W"
    if interval in ['D', 'W', 'M']:
        interval0 = 86400
    else:
        interval0 = interval
    if sdate is None:
        if category == 'spot':
            sdate = find_sdate(category, symbol, interval)
        else:
            sdate = '2020-01-01'
    if sdate is None:
        return pd.DataFrame()
    df_list = []
    last_second = utility.dt_to_millsec(sdate)
    while True:
        print(dt.datetime.fromtimestamp((int(last_second / 1000))))
        new_df = get_bar_data_v5(category, symbol, interval, last_second)
        if new_df is None or int(new_df['startTime'].max()) < last_second:
            break
        df_list.append(new_df)
        last_second = int(new_df['startTime'].max()) + interval0 * 1000
    df = pd.concat(df_list)
    return df


def query_orderbook_futures(symbol):
    url = "https://api.bybit.com/v2/public/orderBook/L2"
    return pd.DataFrame(json.loads(requests.get(url, params={"symbol": symbol}).text)['result'])


# need to be finished
def query_public_trading_history(category, symbol):
    url = "https://api.bybit.com/v5/market/recent-trade"
    if category == 'futures':
        category = 'linear'
    df = pd.DataFrame(
        list(json.loads(requests.get(url, params={"category": category, "symbol": symbol}).text)['result']['list']))
    if df.shape[0] > 0:
        df.index = [dt.datetime.fromtimestamp(int(x) / 1000) for x in df.time]
        df = df.iloc[::-1]
    return df


# need to be finished
def query_open_Interest(symbol, intervalTime, startTime):
    # intervalTime: 5min,15min,30min,1h,4h,1d
    url = "https://api.bybit.com/v5/market/open-interest"
    req_params = {"symbol": symbol, 'interval': intervalTime, 'from': startTime}
    return pd.DataFrame(json.loads(requests.get(url, params={"symbol": symbol}).text)['result'])


def query_basicinfo_spot():
    url = "https://api.bybit.com/spot/v1/symbols"
    basicinfo = pd.DataFrame(json.loads(requests.get(url).text)['result'])[
        ['name', 'alias', 'baseCurrency', 'quoteCurrency', 'minPricePrecision']]
    return basicinfo


def query_basicinfo_futures():
    url = "https://api.bybit.com/v2/public/symbols"
    basicinfo = pd.DataFrame(json.loads(requests.get(url).text)['result'])[
        ['name', 'alias', 'status', 'base_currency', 'quote_currency', 'taker_fee', 'maker_fee']]
    return basicinfo


def download_crypto_data(basicinfo, saving_folder, sdate, interval, sindex=0, tindex=None, source='bybit',
                         returndict=False, assettype='futures'):
    datadict = {}
    if tindex is None:
        tindex = len(basicinfo)
    for i in range(sindex, tindex):
        if basicinfo['quote_currency'].iloc[i] != 'USDT':
            continue
        print(basicinfo['alias'].iloc[i])
        try:
            datadict[basicinfo['alias'].iloc[i]] = query_histdata(basicinfo['alias'].iloc[i], interval, sdate, source)
            datadict[basicinfo['alias'].iloc[i]].to_csv(saving_folder + basicinfo['alias'].iloc[i] + '.csv')
        except Exception as e:
            print(e)
    if returndict:
        return datadict


def download_crypto_data_v5(basicinfo, assettype, saving_folder, sdate, interval, sindex=0, tindex=None,
                            returndict=False):
    datadict = {}
    if assettype == 'futures':
        assettype = 'linear'
    if tindex is None:
        tindex = len(basicinfo)
    if 'quoteCurrency' in basicinfo.columns:
        quotecol = 'quoteCurrency'
    else:
        quotecol = 'quote_currency'
    for i in range(sindex, tindex):
        if basicinfo[quotecol].iloc[i] != 'USDT':
            continue
        print(basicinfo['alias'].iloc[i])
        try:
            if assettype == 'spot':
                datadict[basicinfo['alias'].iloc[i]] = query_histdata_v5(assettype, basicinfo['alias'].iloc[i],
                                                                         interval, sdate=None)
            else:
                datadict[basicinfo['alias'].iloc[i]] = query_histdata_v5(assettype, basicinfo['alias'].iloc[i],
                                                                         interval, sdate=sdate)
            datadict[basicinfo['alias'].iloc[i]].to_csv(saving_folder + basicinfo['alias'].iloc[i] + '.csv')
        except Exception as e:
            print(e)
    if returndict:
        return datadict


# save live orderbook futures data
class Orderbook_Futures(object):
    def __init__(self, data_dir):
        # data save folder
        raw_data_dir = os.path.join(data_dir, r'raw_orderbook_futures')
        if not os.path.exists(raw_data_dir):
            print('Creating folder {}...'.format(raw_data_dir))
            os.mkdir(raw_data_dir)
        else:
            print('Folder {} already existed'.format(raw_data_dir))
        self.raw_data_dir = raw_data_dir
        
        futures_info = query_basicinfo_futures()
        self.futures_universe = futures_info.name.unique()

    def helper(self, future_name):
        time_now = dt.datetime.fromtimestamp(time.time())
        raw_data_file_today = os.path.join(self.raw_data_dir,
                                           '{}.parquet'.format(int(time.time()*1000)))
        item0 = query_orderbook_futures(future_name).assign(collect_time=time_now)
        item0.to_parquet(raw_data_file_today, engine='fastparquet')

    def _live_pull(self):

        with Pool(4) as p:
            p.map(self.helper, list(self.futures_universe))


        
    def pulling(self, max_attempts=5):
        attempts = 0

        while attempts < max_attempts:
            try:
                self._live_pull()
            except:
                attempts += 1
                print("{} try failed.".format(attempts))




# save public trading history spot+futures data
class Public_Trading_History(object):
    def __init__(self, data_dir, categories=('spot',)):
        # check save data folders
        self.data_dir = data_dir
        for category in categories:
            os.makedirs(os.path.join(data_dir, 'raw_public_trading_history', category), exist_ok=True)

        # create names universe
        self.universe = dict()
        for category in categories:
            if category == 'spot':
                spot_info = query_basicinfo_spot()
                self.universe['spot'] = spot_info.name.unique()
            if category == 'futures':
                futures_info = query_basicinfo_futures()
                self.universe['futures'] = futures_info.name.unique()

        # create last records
        self.last_records = {category:dict() for category in categories}
        
    def _init_pull(self):
        # prepare: create last records, save them to parquet file, 
        # don't need to check whether they already existed,
        # assume they are the very first records for each name.
        for category, names in self.universe.items():
            for name in names:
                item0 = query_public_trading_history(category, name)
                if item0.shape[0] > 0:
                    self.last_records[category][name] = item0
                    path_today = os.path.join(self.data_dir, 'raw_public_trading_history', category, \
                                              '{}.parquet'.format(dt.datetime.today().strftime('%Y%m%d')))
                    if not os.path.isfile(path_today):
                        item0.to_parquet(path_today, engine='fastparquet')
                    else:
                        item0.to_parquet(path_today, engine='fastparquet', append=True)


    def _live_pull(self):
        while True:

            for category in self.last_records.keys():
                for name in self.last_records[category].keys():
                    current = query_public_trading_history(category, name)
                    path_today = os.path.join(self.data_dir, 'raw_public_trading_history', category,
                                              '{}.parquet'.format(dt.datetime.today().strftime('%Y%m%d')))
                    current_new = current[~current.execId.isin(self.last_records[category][name].execId)]
                    if not os.path.isfile(path_today):
                        current_new.to_parquet(path_today, engine='fastparquet')
                    else:
                        current_new.to_parquet(path_today, engine='fastparquet', append=True)
                    self.last_records[category][name] = current
    
    def pulling(self, max_attempts=5):
        attempts = 0
        while attempts < max_attempts:
            try:
                print('Init Pull started.')
                self._init_pull()
                print('Init Pull finished.')
                break
            except:
                attempts += 1
                print('Init Pull {} try failed.'.format(attempts))

        attempts = 0
        while attempts < max_attempts:
            try:
                print('Live pulling......')
                self._live_pull()
            except:
                attempts += 1
                print("{} try failed.".format(attempts))











