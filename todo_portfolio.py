#coding:utf8
import pandas as pd
import sys 
import os
import itertools
"""
x<=2000:small cap
x>2000,x<=10000:mid cap
x>=10000:large cap
"""
def top_25_score(tdate,indata,topk,outfile,rank_list):
    """
    tdate:所在日期YYYY-MM-DD
    indata:需要处理的数据
    topk:portfolio的规模
    outfile:结果写出的文件
    rank_list:本次portfolio所利用到的rank
    """
    indata = indata.dropna()
    scheme = indata.columns
    t1 = indata[scheme[:-2]].sum(axis=1)
    t2 = indata[scheme[-1]]
    t3 = indata[scheme[-2]]
    d1 = pd.DataFrame(t1,columns=['sum'])
    d2 = pd.DataFrame(t2,columns=['3mr'])
    d4 = pd.DataFrame(t3,columns=['mktcap'])
    bool_cap_dict = {} 
    bool_s = d4<=2000
    bool_cap_dict["samll"] = bool_s
    bool_m =((d4<=10000)&(d4>2000))
    bool_cap_dict["mid"] = bool_m
    bool_l = d4>=10000
    bool_cap_dict["large"] = bool_l
    for key in bool_cap_dict:
        bool_signal = bool_cap_dict[key]
        d5 = d4[bool_signal]
        d_comput = d1.add(d2,fill_value=0)
        d_comput = d_comput.add(d5,fill_value=0)
        d_comput = d_comput.dropna()
        #print key,len(d_comput)
        rdata = d_comput.sort_values(['sum'],ascending=False)
        p1 = rdata.head(topk)
        outfile.write("%s\t%s\t%s\t%.2f\n"%(tdate,key,"+".join(rank_list[:-2]),p1['3mr'].mean()))
        #outfile.write("%s\t%s\t%s\t%.2f\n"%(tdate,key,"+".join(rank_list[:-2]),p1['3mr'].sum()))
    d_comput = d1.add(d2,fill_value=0)
    d_comput = d_comput.add(d4,fill_value=0)
    rdata = d_comput.sort_values(['sum'],ascending=False)
    p1 = rdata.head(topk)
    outfile.write("%s\t%s\t%s\t%.2f\n"%(tdate,"all cap","+".join(rank_list[:-2]),p1['3mr'].mean()))
    #outfile.write("%s\t%s\t%s\t%.2f\n"%(tdate,"all cap","+".join(rank_list[:-2]),p1['3mr'].sum()))
    


if __name__ == "__main__":
    #data_part1 = pd.read_csv('data/krank.part1')
    #data_part2 = pd.read_csv('data/krank.part2')
    #data_part2_pre1 = data_part2.fillna(0)
    #indata = pd.merge(data_part1,data_part2_pre1)
    #topk_list = [30,40,60,80,100]
    topk_list = [40]
    indata = pd.read_csv('data/krank.csv',sep=",")
    temp_group = indata.groupby('date')
    dict_rank = {}
    dict_rank['A'] = 'krank'
    dict_rank['B'] = 'quality'
    dict_rank['C'] = 'value'
    dict_rank['D'] = 'momentum'
    dict_rank['E'] = 'growth'
    #if the target is 3mr:
    target_month = [1,4,7,10]
    #if the traget is 6mr:
    #target_month = [1,7]
    for topk in topk_list:
        print "topk:",topk
        ofile = "result/result." + str(topk)
        if os.path.exists(ofile):
            os.remove(ofile)
        outfile = open(ofile,'a')
        for key_date,group_data in temp_group:
            month = int(key_date[5:7])
            print key_date,month
            if month not in target_month:
                continue    
            for k in range(1,len(dict_rank)+1):
                for i in itertools.combinations('ABCDE', k):
                    need_list = []
                    for j in range(k):
                        need_list.append(dict_rank[i[j]])
                    #outfile.write("\n")
                    need_list.append('mktcap')
                    need_list.append('3mr')
                    krank = group_data[need_list]
                    top_25_score(key_date,krank,topk,outfile,need_list)
        outfile.close
