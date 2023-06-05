import smtplib
import mimetypes
from email.mime.multipart import MIMEMultipart
from email import encoders
from email.message import Message
from email.mime.audio import MIMEAudio
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email.mime.text import MIMEText
import datetime as dt
import time
import requests
import json
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import statsmodels.api as sm
from glob import glob
import numpy as np
import os
from IPython.display import display, HTML

def sendemail(emailto,
              subject,
              text="",
              html=None,
              emailfrom="wmlakefund.noreply@gmail.com",
              fileToSend=None,
              username="wmlakefund.noreply@gmail.com",
              password="wnxdtjhkdtromarz"):
    msg = MIMEMultipart("alternative")
    msg["From"] = emailfrom
    msg["To"] = emailto
    msg["Subject"] = subject
    msg.preamble = subject
    if not html is None:
        msg.attach(MIMEText(html, 'html'))
    else:
        msg.attach(MIMEText(text, 'plain'))

    if not fileToSend is None:
        ctype, encoding = mimetypes.guess_type(fileToSend)
        if ctype is None or encoding is not None:
            ctype = "application/octet-stream"
        maintype, subtype = ctype.split("/", 1)

        if maintype == "text":
            fp = open(fileToSend)
            # Note: we should handle calculating the charset
            attachment = MIMEText(fp.read(), _subtype=subtype)
            fp.close()
        elif maintype == "image":
            fp = open(fileToSend, "rb")
            attachment = MIMEImage(fp.read(), _subtype=subtype)
            fp.close()
        elif maintype == "audio":
            fp = open(fileToSend, "rb")
            attachment = MIMEAudio(fp.read(), _subtype=subtype)
            fp.close()
        else:
            fp = open(fileToSend, "rb")
            attachment = MIMEBase(maintype, subtype)
            attachment.set_payload(fp.read())
            fp.close()
            encoders.encode_base64(attachment)

        fileNameShort = fileToSend
        if '\\' in fileToSend:
            fileNameShort = fileToSend.split('\\')[-1]
        else:
            fileNameShort = fileToSend.split('/')[-1]

        attachment.add_header("Content-Disposition", "attachment", filename=fileNameShort)
        msg.attach(attachment)
    server = smtplib.SMTP("smtp.gmail.com:587")
    server.starttls()
    server.login(username, password)
    server.sendmail(emailfrom, emailto, msg.as_string())
    server.quit()


def sendemails(emailtolist, subject, fileToSend):
    for emailto in emailtolist:
        sendemail(emailto=emailto, subject=subject, fileToSend=fileToSend)


# convert date or datetime to seconds, as one argument for v5 API
def dt_to_millsec(date, dateformat='%Y-%m-%d'):
    if len(date) == 19 and dateformat == '%Y-%m-%d':
        dateformat = '%Y-%m-%d %H:%M:%S'
    t = dt.datetime.strptime(date, dateformat)
    return int(time.mktime(t.timetuple()) * 1000)

def gfun_df(nname, df, tgt_col, gb, f, cond=[], tgt_arr=[]):
    df[nname] = np.nan
    if 0 < len(tgt_arr):
        df[tgt_col] = tgt_arr
    if len(cond) > 0:
        df.loc[cond, nname] = df[cond].groupby(gb)[tgt_col].transform(f)
    else:
        df[nname] = df.groupby(gb)[tgt_col].transform(f)
    return None

def long2wide(df, factorname, date_var = "date", id_var = "symbol"):
    df_wide = df[[date_var, id_var, factorname]].groupby([date_var, id_var], as_index = False).sum().pivot(date_var, id_var).replace(0, np.nan)
    df_wide.columns = [col[1] for col in df_wide.columns]
    return df_wide

def wide2long(wide, value_name):
    long = wide.unstack().reset_index()
    long.columns = ['symbol', 'date', value_name]
    return long

def xrank(list_pandas, n):
    if all(list_pandas.isnull()):
        return list_pandas
    return pd.qcut(list_pandas, n, labels=False)

def rerank(list_pandas):
    s = list_pandas.notnull().sum()
    return (list_pandas.round(6).rank() - 0.5 * (s + 1))/(0.5 * (s - 1))

def generateweight(list_pandas):
    wt_pos = list_pandas * (list_pandas > 0)
    wt_neg = list_pandas * (list_pandas < 0)
    iszero = (np.abs(list_pandas) < 1e-10)
    wt = wt_pos / np.abs(wt_pos).sum() + wt_neg / np.abs(wt_neg).sum()
    return (wt * (1 - iszero)).replace([np.inf, -np.inf, np.nan], 0)

def dollar_balance(alldata, wtcol):
    alldata['tmp'] = alldata[wtcol]
    alldata.loc[alldata.inuniv, 'tmp'] = alldata.loc[alldata.inuniv, 'tmp'].fillna(0)
    alldata.loc[~alldata.inuniv, 'tmp'] = np.nan
    alldata['hdg_wt'] = alldata.groupby('date')['tmp'].transform(np.nanmean)
    alldata[wtcol+'_db'] = alldata['tmp'] - alldata['hdg_wt']
    return alldata

def getmaxdd(pnls):
    cumpnl = pnls.cumsum()
    maxdd = 0
    maxdd_sidx = -1
    maxdd_eidx = -1
    currhigh = 0
    currlow = 0
    currdd_sidx = -1
    currdd_eidx = -1
    currdd = 0
    for i in range(len(pnls)):
        if cumpnl.iloc[i] > currhigh:
            if currdd < maxdd:
                maxdd_sidx = currdd_sidx+1
                maxdd_eidx = i-1
                maxdd = currdd
            currdd = 0
            currhigh = cumpnl.iloc[i]
            currlow = cumpnl.iloc[i]
            currdd_sidx = i
        elif cumpnl.iloc[i] < currlow:
            currdd = cumpnl[i] - currhigh
            currlow = cumpnl[i]
            currdd_eidx = i
    if currdd < maxdd:
        maxdd_sidx = currdd_sidx
        maxdd_eidx = len(pnls)-1
        maxdd = currdd
    return maxdd, pnls.index[maxdd_sidx], pnls.index[maxdd_eidx]

import warnings
warnings.filterwarnings("ignore")
def pnldetails(alldata, wtcol, ret_col='fret_15m', sret_col=None, tcost=0, frequency='hourly'):
    if frequency == 'daily' or frequency == 'day':
        cnt = 356
    elif frequency == 'hourly' or frequency == 'hour':
        cnt = 24 * 356
    elif frequency == 'minutely' or frequency == 'minute_bar' or frequency == 'minute':
        cnt = 60 * 24 * 356
    alldata['tmp'] = alldata[wtcol]
    alldata[wtcol] = alldata[wtcol] * alldata.inuniv
    alldata['abs_wt'] = np.abs(alldata[wtcol])
    gfun_df('abs_wt_chg', alldata, wtcol, alldata.symbol, lambda x: np.abs(x.fillna(0).diff()))
    abs_wt_chg_interval = alldata.groupby('date')['abs_wt_chg'].sum().rename('abs_wt_chg')
    gmv_interval = alldata.groupby('date')['abs_wt'].sum().rename('gmv')
    gmv_interval = (gmv_interval + gmv_interval.shift(1)).replace(0,np.nan) / 2
    if not sret_col is None:
        alldata['spnl'] = alldata[wtcol] * alldata[sret_col]
        spnl_interval = alldata.groupby('date')['spnl'].sum().rename('spnl')
        spnl_interval_pct = (spnl_interval / gmv_interval.replace(0,np.nan)).rename('spnl(%)')
        spnl_interval_ac_pct = ((spnl_interval - abs_wt_chg_interval * tcost) / gmv_interval.replace(0,np.nan)).rename('spnl_ac(%)')
    alldata['pnl'] = alldata[wtcol] * alldata[ret_col]
    pnl_interval = alldata.groupby('date')['pnl'].sum().rename('pnl')
    pnl_interval_pct = (pnl_interval / gmv_interval.replace(0,np.nan)).rename('pnl(%)')
    pnl_interval_ac_pct = ((pnl_interval - abs_wt_chg_interval * tcost) / gmv_interval.replace(0,np.nan)).rename('pnl_ac(%)')

    turnover_interval = (abs_wt_chg_interval / gmv_interval).rename('turnover')
    if not sret_col is None:
        tab_interval = pd.concat([gmv_interval, turnover_interval, spnl_interval, spnl_interval_pct, spnl_interval_ac_pct, pnl_interval, pnl_interval_pct, pnl_interval_ac_pct, abs_wt_chg_interval], axis = 1)
    else:
        tab_interval = pd.concat([gmv_interval, turnover_interval, pnl_interval, pnl_interval_pct, pnl_interval_ac_pct, abs_wt_chg_interval], axis = 1)
    tab_interval.index = tab_interval.index.astype(str)
    tab_interval = tab_interval.reset_index()
    tab_interval = tab_interval[tab_interval['date'] > '2021']
    if not 'year' in tab_interval.columns:
        tab_interval['year'] = pd.to_datetime(tab_interval['date']).dt.year
    if not 'month' in tab_interval.columns:
        tab_interval['month'] = pd.to_datetime(tab_interval['date']).dt.month
    tab_interval['quarter'] = np.floor((tab_interval['month']-1)/3)+1
    tab_interval['quarter'] = tab_interval['quarter'].astype(int)
    tab_interval['YQ'] = tab_interval['year'].astype(str) + 'Q' + tab_interval['quarter'].astype(str)

    if not sret_col is None:
        tab_yq = pd.concat([tab_interval.groupby('YQ')[['gmv', 'turnover']].mean(), tab_interval.groupby('YQ')[['spnl(%)', 'pnl(%)', 'spnl_ac(%)', 'pnl_ac(%)']].sum() * 100], axis=1)
    else:
        tab_yq = pd.concat([tab_interval.groupby('YQ')[['gmv', 'turnover']].mean(), tab_interval.groupby('YQ')[['pnl(%)', 'pnl_ac(%)']].sum() * 100], axis=1)
    if not sret_col is None:
        tab_yq['sretptrade'] = tab_interval.groupby('YQ')['spnl'].sum() / tab_interval.groupby('YQ')['abs_wt_chg'].sum() * 1e4
        tab_yq['sSR'] = tab_interval.groupby('YQ')['spnl(%)'].mean() / tab_interval.groupby('YQ')['spnl(%)'].std() * np.sqrt(cnt)
        tab_yq['sSR_ac'] = tab_interval.groupby('YQ')['spnl_ac(%)'].mean() / tab_interval.groupby('YQ')['spnl_ac(%)'].std() * np.sqrt(cnt)
    tab_yq['retptrade'] = tab_interval.groupby('YQ')['pnl'].sum() / tab_interval.groupby('YQ')['abs_wt_chg'].sum() * 1e4
    tab_yq['SR'] = tab_interval.groupby('YQ')['pnl(%)'].mean() / tab_interval.groupby('YQ')['pnl(%)'].std() * np.sqrt(cnt)
    tab_yq['SR_ac'] = tab_interval.groupby('YQ')['pnl_ac(%)'].mean() / tab_interval.groupby('YQ')['pnl_ac(%)'].std() * np.sqrt(cnt)
    if not sret_col is None:
        tab_yq.loc['summary'] = [tab_interval['gmv'].mean(),
                                   tab_interval['turnover'].mean(),
                                   tab_interval['spnl(%)'].mean() * cnt * 100,
                                   tab_interval['pnl(%)'].mean() * cnt * 100,
                                   tab_interval['spnl_ac(%)'].mean() * cnt * 100,
                                   tab_interval['pnl_ac(%)'].mean() * cnt * 100,
                                   tab_interval['spnl'].sum() / tab_interval['abs_wt_chg'].sum() * 1e4,
                                   tab_interval['spnl(%)'].mean() / tab_interval['spnl(%)'].std() * np.sqrt(cnt),
                                   tab_interval['spnl_ac(%)'].mean() / tab_interval['spnl_ac(%)'].std() * np.sqrt(cnt),
                                   tab_interval['pnl'].sum() / tab_interval['abs_wt_chg'].sum() * 1e4,
                                   tab_interval['pnl(%)'].mean() / tab_interval['pnl(%)'].std() * np.sqrt(cnt),
                                   tab_interval['pnl_ac(%)'].mean() / tab_interval['pnl_ac(%)'].std() * np.sqrt(cnt)]
    else:
        tab_yq.loc['summary'] = [tab_interval['gmv'].mean(),
                                   tab_interval['turnover'].mean(),
                                   tab_interval['pnl(%)'].mean() * cnt * 100,
                                   tab_interval['pnl_ac(%)'].mean() * cnt * 100,
                                   tab_interval['pnl'].sum() / tab_interval['abs_wt_chg'].sum() * 1e4,
                                   tab_interval['pnl(%)'].mean() / tab_interval['pnl(%)'].std() * np.sqrt(cnt),
                                   tab_interval['pnl_ac(%)'].mean() / tab_interval['pnl_ac(%)'].std() * np.sqrt(cnt)]
    avg_pos_wt = (alldata[alldata[wtcol]>0].groupby('date')[wtcol].sum()).mean()
    avg_neg_wt = (alldata[alldata[wtcol]<0].groupby('date')[wtcol].sum()).mean()
    print(f'avg positive weight: {avg_pos_wt}')
    print(f'avg negative weight: {avg_neg_wt}')

    tab_yq = round(tab_yq.fillna(0), 2)
    display(HTML(tab_yq.to_html()))
    tab_interval = tab_interval.set_index('date')
    if not sret_col is None:
        tab_interval['fpnl(%)'] = tab_interval['pnl_ac(%)'] - tab_interval['spnl_ac(%)']
        tab_interval['pnl_ac(%)'].fillna(0).cumsum().plot(title = "PnL", color = "red", figsize = (15,6))
        tab_interval['spnl_ac(%)'].fillna(0).cumsum().plot(title = "PnL", color = "green", figsize = (15,6))
        tab_interval['fpnl(%)'].fillna(0).cumsum().plot(title = "PnL", color = "blue", figsize = (15,6))
        plt.gca().legend(("pnl_ac", "spnl_ac", "fpnl"))
    else:
        tab_interval['pnl_ac(%)'].fillna(0).cumsum().plot(title = "PnL", color = "red", figsize = (15,6))
    return

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
def cgma(df,
         gb,
         wtcol='',
         cn=[],
         rmod='fperf',
         ofun_dict={},
         ret_ds=[]):
    if len(ret_ds) == 0:
        if rmod == 'fperf':
            ret_ds = [1,2,3,5,12,24,48,24*7]
        elif rmod == 'fret':
            ret_ds = [1,5,10,15,20,30]
        else:
            raise ValueError('ret_ds is empty!')
    zip_flag = False
    if not type(gb) in (list, str):
        gname = gb.name
        if gname in df.columns:
            if df[gname].dtype == float:
                ofun_dict.update({
                    ('med_'+gname):lambda df: df[gname].median(),
                    ('min_'+gname):lambda df: df[gname].min(),
                    ('max_'+gname):lambda df: df[gname].max()
                })
                cn = cn + ['med_'+gname, 'min_'+gname, 'max_'+gname]
        gb = gb.rename('mygroup')
    if wtcol=='':
        df.loc[:,'ones'] = 1
        wtcol = 'ones'
    res_tab = df.groupby(gb).apply(multi_analysis, wtcol, cn, ofun_dict, rmod, ret_ds)
    if wtcol == 'ones':
        df = df.drop(columns=['ones'])
    index_cols = [i for i in list(res_tab.index.names) if i]
    res_tab = res_tab.reset_index()
    del res_tab['level_1']
    res_tab.set_index(index_cols)
    return res_tab

def multi_analysis(df,
                   wtcol='',
                   cn=[],
                   ofun_dict={},
                   rmod='fperf',
                   ret_ds=[1,3,5,10,15,20]):
    basic_fun_dict = {'cnt': lambda df: len(df), 'vol': lambda df: df['vol'].median(), 'adv': lambda df: df['adv'].median()}
    basic_fun_dict.update(ofun_dict)
    keys = []
    ress = []
    for c in cn:
        if c in basic_fun_dict:
            keys.append(c)
            ress.append((basic_fun_dict[c])(df))
        elif c in df.columns:
            keys.append('med_'+c)
            ress.append(df[c].median())
            keys.append('min_'+c)
            ress.append(df[c].min())
            keys.append('max_'+c)
            ress.append(df[c].max())
    if len(ret_ds) > 0:
        for nd in ret_ds:
            keys.append(nd)
            if nd == 0:
                ret = np.nansum(df['zperf'].clip(-0.5,0.5) * df[wtcol].abs()) / np.nansum(df[wtcol].abs())
            elif rmod == 'fret':
                ret = np.nansum(df[rmod+'_'+str(nd)+'m'].clip(-0.5,0.5) * df[wtcol].abs()) / np.nansum(df[wtcol].abs())
            else:
                ret = np.nansum(df[rmod+'_'+str(nd)].clip(-0.5,0.5) * df[wtcol].abs()) / np.nansum(df[wtcol].abs())
            ress.append(ret)
    return pd.DataFrame({keys[i]:[ress[i]] for i in range(len(keys))})

def getIC(alldata, wtcol, retcol='fperf_1', absret=False):
    gfun_df(f'alpha_{wtcol}', alldata, wtcol, 'date', lambda x: rerank(x), cond=alldata.inuniv)
    if absret:
        alldata[f'abs_{retcol}'] = np.abs(alldata[retcol])
        ic = alldata[[f'alpha_{wtcol}', f'abs_{retcol}']].corr().iloc[0,1]
        del alldata[f'abs_{retcol}']
    else:
        ic = alldata[[f'alpha_{wtcol}', retcol]].corr().iloc[0,1]
    del alldata[f'alpha_{wtcol}']
    return ic

def createRiskFactors(alldata):
    def seas30(x):
        res = x.shift(24)
        for i in range(2,31):
            res += x.shift(i*24)
        return res/30
    gfun_df(f'volume24', alldata, 'volume', 'symbol', lambda x: x.rolling(window=24).mean())
    alldata['illiq'] = np.abs(alldata['bperf_1']) / alldata['vol'] / alldata['volume']
    gfun_df(f'illiq24', alldata, 'illiq', 'symbol', lambda x: x.rolling(window=24).mean())
    gfun_df(f'zperf_seas30', alldata, 'zperf', 'symbol', lambda x: seas30(x))
    alldata['zperf_seas30_vol'] = alldata['zperf_seas30'] / alldata['vol']
    del alldata['illiq']
    del alldata['zperf_seas30']
    return alldata

def neutral(df):
    resret = None
    print("Start neutralization...")
    dates = df.index.unique()
    for i in range(len(dates)):
        print("Doing " + str(i+1) + "/" + str(len(dates)), end = "\r")
        try:
            dfi = df.loc[dates[i]].reset_index().drop(columns='date').set_index('symbol')
            model = sm.OLS(dfi[dfi.inuniv].iloc[:,-1], sm.add_constant(dfi[dfi.inuniv].iloc[:,1:-1]), missing = "drop")
            results = sm.add_constant(dfi.iloc[:,1:-1]).dot(model.fit().params)
            res = dfi.iloc[:,-1] - results
            res = res.rename('szperf').reset_index()
            res['date'] = dates[i]
            if resret is None:
                resret = res
            else:
                resret = pd.concat([resret, res])
        except Exception as e:
            print(e, "\n")
    print("Neutralization completed")
    return resret.reset_index().drop(columns=['index'])

def hedge_style(alldata, colname, tgtname=''):
    if tgtname == '':
        tgtname = colname + '_neu'
    riskfactors = alldata[['date', 'symbol', 'inuniv', 'vol', 'volume24', 'illiq24', 'zperf_seas30_vol', colname]]
    riskfactors = riskfactors.loc[riskfactors.date>'2022'].sort_values(by = ['date']).set_index('date')
    df_hdg = neutral(riskfactors)
    alldata = pd.merge(alldata, df_hdg.rename(columns={'szperf':tgtname}), on=['date', 'symbol'], how='left')
    return alldata

