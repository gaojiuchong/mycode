#coding:utf8
import collections as coll
import pandas as pd
import os
folder_path = "result/"
namelist = os.listdir(folder_path)
for filename in namelist:
    dict_ey = coll.defaultdict(float) 
    dict_ay = coll.defaultdict(float) 
    dict_ey_cap = coll.defaultdict(float) 
    dict_ay_cap = coll.defaultdict(float) 
    year_set = set()
    cap_set = set()
    print filename
    file_path = folder_path + filename 
    for line in file(file_path):
        list_temp = line.strip().split("\t")
        if len(list_temp) == 1:
            continue
        date = list_temp[0]
        year = date[:4]
        year_set.add(year)
        cap = list_temp[1]
        cap_set.add(cap)
        port = list_temp[2]
        rtn = float(list_temp[3])
        #每年全体量
        ey_key = (year,port)
        #全年全体量
        ay_key = (port)
        #每年各种体量（低，中，高）
        ey_cap_key = (year,cap,port)
        #全年各种体量（低，中，高）
        ay_cap_key = (cap,port)
        if dict_ey[ey_key] == 0.0:
            dict_ey[ey_key] = 1.0
        if dict_ay[ay_key] == 0.0:
            dict_ay[ay_key] = 1.0
        if dict_ey_cap[ey_cap_key] == 0.0:
            dict_ey_cap[ey_cap_key] = 1.0
        if dict_ay_cap[ay_cap_key] == 0.0:
            dict_ay_cap[ay_cap_key] = 1.0
        dict_ey[ey_key] *= (1+rtn/100.0)
        dict_ay[ay_key] *= (1+rtn/100.0)
        dict_ey_cap[ey_cap_key] *= (1+rtn/100.0)
        dict_ay_cap[ay_cap_key] *= (1+rtn/100.0)
    for target_y in year_set:
        """
        temp_dict = coll.defaultdict(list)
        for key in dict_ey:
            year = key[0]
            port = key[1]
            if target_y == year:
                temp_dict[key].append(dict_ey[key])
        dx=sorted(temp_dict.iteritems(),key=lambda d:d[1],reverse=True)  
        print("%s\t%.2f\t%s\t%.2f"%(dx[0][0],dx[0][1][0],dx[-1][0],dx[-1][1][0])) 
        """
        for target_cap in cap_set:
            temp_dict = coll.defaultdict(list)
            for key in dict_ey_cap:
                year = key[0]
                cap = key[1]
                port = key[2]
                if target_y == year and target_cap == cap:
                    temp_dict[key].append(dict_ey_cap[key])
            dx=sorted(temp_dict.iteritems(),key=lambda d:d[1],reverse=True)
            print("%s\t%.2f\t%s\t%.2f"%(dx[0][0],dx[0][1][0]-1,dx[-1][0],dx[-1][1][0]-1)) 
    #dx=sorted(dict_ay.iteritems(),key=lambda d:d[1],reverse=True)
    #print("%s\t%.2f\t%s\t%.2f"%(dx[0][0],dx[0][1],dx[-1][0],dx[-1][1])) 
    for target_cap in cap_set:
        temp_dict = coll.defaultdict(list)
        for key in dict_ay_cap:
            cap = key[0]
            port = key[1]
            if target_cap == cap:
                temp_dict[key].append(dict_ay_cap[key])
        dx=sorted(temp_dict.iteritems(),key=lambda d:d[1],reverse=True)
        print("%s\t%.2f\t%s\t%.2f"%(dx[0][0],dx[0][1][0]-1,dx[-1][0],dx[-1][1][0]-1)) 
