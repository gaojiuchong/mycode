#coding:utf8
import urllib2
import pandas as pd
import sys
def http_get(b_date,e_date):
    url = "http://69.64.57.201:18088/roi/portfolio/spy:1/?start=" + b_date + "&end=" + e_date
    #url='http://69.64.57.201:18088/roi/portfolio/spy:1/?start=20140101&end=20140201'
    response = urllib2.urlopen(url)
    return response.read()
b_date = "20140101"
e_date = "20140201"
b_month = 1
e_month = 12
b_year = 2005
e_year = 2015
for year in range(b_year,e_year+1):
    for month in range(b_month,e_month+1):
        b_date = '%d%02d01' % (year, month)
        if (month+6) > 12:
            e_date = '%d%02d01' % (year+1, month-6)
        else:
            e_date = '%d%02d01' % (year, month+6)
        ret = http_get(b_date,e_date)
        pdata = pd.read_json(ret)
        roi = pdata["data"].get("roi")
        print b_date,e_date,roi
