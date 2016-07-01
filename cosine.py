#coding:utf8
import pandas as pd
import sys 
import os
import itertools
if __name__ == "__main__":
    indata = pd.read_csv('data/krank.csv',sep=",")
    del indata['date']
    del indata['symbol']
    del indata['mktcap']
    del indata['3mrs']
    del indata['6mrs']
    #del indata['6mr']
    del indata['idx']
    indata = indata.dropna()
    indata['3mr'] = indata['3mr'].rank(method="first",pct=1)
    indata['3mr'] = indata['3mr']*100
    indata['3mr'] = indata['3mr'].round(0)
    indata['6mr'] = indata['6mr'].rank(method="first",pct=1)
    indata['6mr'] = indata['6mr']*100
    indata['6mr'] = indata['6mr'].round(0)
    print indata 
    print indata.corr()
