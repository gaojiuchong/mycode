#coding:utf8
import pandas as pd 
import logging
import sys
import math
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
import collections as coll
import time
log_file = "log/lstm_logger.log"
logging.DEBUG
class PredictResult: 
    #test_data_date = [] 
    #predict_label_test=[]
    #time_windows = 0
    #time_step = 0 
    relo2_acc = 0.0
    #rel02_acc: +-20% is right
    abs01_acc = 0.0
    #abs01_acc: +- 0.1 is right
    sign_acc = 0.0
    #sign_acc: +- predict
    loss_value = 0.0
    real_abs_rtn = []
    real_label = []
    pred_label = []
    pred_label_info = []
    symbol = []
    def __init__(self,real_label,predict_label,sigle_targe=True,real_abs_rtn=[],symbol=[]):
        self.real_abs_rtn = real_abs_rtn
        self.symbol = symbol
        self.real_label = []
        self.pred_label = []
        if sigle_targe:
            if len(real_label) != len(predict_label):
                logging.info('[MODEL] the numbers of label and prediction are different.')
                return None
            c = len(real_label)
            c1 = 0 
            c2 = 0
            c3 = 0
            los_value = 0
            for real,predict in zip(real_label,predict_label):
                real = real[0]
                predict = predict[0]
                los_value += (predict-real)*(predict-real)
                if real >= 0:
                    low1 = real*0.8
                    high1 = real*1.2
                else:
                    low1 = real*1.2
                    high1 = real*0.8
                low2 = real - 10
                high2 = real + 10
                if predict >= low1 and predict <= high1:
                    c1+=1
                if predict >= low2 and predict <= high2:
                    c2+=1
                if (real <= 0 and predict <= 0) or (real >= 0 and predict >= 0):
                    c3+=1
            self.loss_value = math.sqrt(los_value)/c
            self.relo2_acc= 1.0*c1/c
            self.abs01_acc= 1.0*c2/c
            self.sign_acc = 1.0*c3/c
        else:
            pre = 0
            cnt1 = 0
            cnt2_str = 0
            cnt2_weak = 0
            cnt2_all_str = 0
            cnt2_all_weak = 0
            #label_l = []
            self.pred_label_info = []
            for real,pro_label in zip(real_label, predict_label):
                real = real[0]
                self.real_label.append(real)
                temp_list = []
                maxtemp = -1.0
                for idx,value in enumerate(pro_label):
                    temp_list.append("%.2f"%value)
                    if value > maxtemp:
                        maxtemp = value
                        pre = idx
                self.pred_label_info.append(" ".join(temp_list))
                str_v = 0.5
                weak_v = 0.3
                self.pred_label.append(pre)
                if maxtemp >= str_v:
                    cnt2_all_str += 1
                if maxtemp <= weak_v:
                    cnt2_all_weak += 1
                if int(real) == int(pre):
                    cnt1 += 1
                    if maxtemp >= str_v:
                        cnt2_str += 1
                    if maxtemp <= weak_v:
                        cnt2_weak += 1
            self.relo2_acc = 1.0*cnt1/len(real_label)
            if cnt2_all_str != 0:
                print "strong prediction :", cnt2_str, cnt2_all_str, 1.0*cnt2_str/cnt2_all_str
            if cnt2_all_weak != 0:
                print "weak prediction :",cnt2_weak,cnt2_all_weak, 1.0*cnt2_weak/cnt2_all_weak
            """ 
            abs_rtn = real_abs_rtn
            outfile = open("info/detail","a") 
            for idx,rtn in enumerate(real_abs_rtn):
                score = label_l[idx]
            """

    def show(self):
        print self.relo2_acc
        print self.abs01_acc
        print self.sign_acc
        print self.loss_value

    def to_file(self,filename,time_series=[],time_windows=12,time_step=3,regression=True):
        """
        regression : True-regression result,False-classifier result
        """
        outfile = open(filename,"a")
        if regression == True:
            outfile.write("tw\tts\ttd\tacc1\tacc2\tacc3\tlossv\n")
            outfile.write("%d\t%d\t%s\t%.4f\t%.4f\t%.4f\t%.2f\n"%(time_windows,time_step,str(time_series[-1]),self.relo2_acc,self.abs01_acc,self.sign_acc,self.loss_value))
        else:
            indata = {'real_abs_rtn':self.real_abs_rtn,
                      'real_label':self.real_label,
                      'pred_label':self.pred_label,
                      'pred_label_info':self.pred_label_info}
            pdata = pd.DataFrame(indata,index=self.symbol)
            pdata['right'] = (pdata['real_label'] == pdata['pred_label'])
            pdata.index.name = "symbol"
            detail_filename = "detail/%d_%d_%s_classifier" %(time_windows,time_step,str(time_series[-1]))
            pdata.to_csv(detail_filename)
            del pdata['pred_label_info']
            group_data = pdata.groupby('pred_label')
            dict_acc = {}
            dict_rtn = {}
            #key:pred_label,value:acc
            for key_value,group_by_key in group_data:
                t = group_by_key['right'].size
                tr = group_by_key['right'].value_counts().get(True)
                if tr == None:
                    tr = 0
                dict_acc[key_value] = 1.0*tr/t
                dict_rtn[key_value] = group_by_key['real_abs_rtn'].mean()
            outfile.write("tw\tts\ttd\tacc1\t")
            for key in dict_acc:
                str_name = "acc_%s\trtn_mean_%s\t" %(str(key),str(key))
                outfile.write(str_name)
            outfile.write("\n")
            outfile.write("%d\t%d\t%s\t%.4f\t"%(time_windows,time_step,str(time_series[-1]),self.relo2_acc))
            for key in dict_acc:
                str_name = "%.4f\t%.4f\t" %(dict_acc[key],dict_rtn[key])
                outfile.write(str_name)
            outfile.write("\n")
        outfile.close() 

def read_file(filename,sep=","):
    return pd.read_csv(filename,sep)

def lstm_combine(time_windows=12, time_step=3, train_data=None, target="3mr", date="date",symbol="symbol", feature=None, test_data=None):
    """
    using the timewindows to build the model,predict the label of testdata
    input:train_data,train_label,test_data
    return: the list of predict label of testdata and the inner data prediction result
    feature : if feature == None,the feature is  all feature except target,date,symbol
    """
    if feature == None:
        feature = list(train_data.columns.drop([date,symbol,target],errors='ignore'))
    else:
        feature.remove(target)
        feature.remove(date)
        feature.remove(symbol)
    if time_windows < time_step:
        logging.info('[para error] time_windows < time_step')
        return 
    """
    if len_train_data == 0:
        logging.info('[MODEL] Testdata is null.')
    else:
        timestep_test = len(train_data[0])
        if timestep_test != timestep:
            logging.info('[MODEL] The timestep of testdata is %d,expect it is %d.'%(timestep_test,timestep))
        else:
    """
    alldate_l = train_data[date].unique()
    time_series_list = get_time_sequence(alldate_l,time_windows) 
    ddata = train_data[date]
    for time_series in time_series_list:
        bool_l = ddata.isin(time_series)
        this_train_data = train_data[bool_l]
        print time_series 
        #predict_result = lstm_regression(time_step,this_train_data,target,date,symbol,feature)
        predict_result = lstm_classifier(time_step,this_train_data,target,date,symbol,feature)
        #predict_result.show()
        tdate = time.strftime('%Y-%m-%d',time.localtime(time.time()))
        predict_result.to_file("lstm.classifier.result_" + tdate,time_series,time_windows,time_step,False)
    
def lstm_regression(time_step=3,train_data=None,target="3mr",date="date",symbol="symbol",feature=None):
    """
    input:
    return: PredictResult
    """
    date_l = train_data[date].unique()
    if time_step > len(date_l):
        return None
    model_train_data,model_train_label,model_test_data,model_test_label = split_regress_sample_in_time_series(time_step,train_data,feature,target,date,symbol)
    print("train sample number:%d, test samples number :%d "%(len(model_train_data),len(model_test_data)))
    if len(model_train_data) == 0:
        logging.info('[DATA ERROR]lstm')
    model = make_lstm_regession_model(model_train_data, model_train_label)
    predict= model.predict_proba(model_test_data)
    return PredictResult(model_test_label,predict,True)

def lstm_classifier(time_step=3,train_data=None,target="3mr",date="date",symbol="symbol",feature=None):
    """
    input:
    return: PredictResult
    """
    date_l = train_data[date].unique()
    if time_step > len(date_l):
        return None
    model_train_data,model_train_label,model_test_data,model_test_label,model_test_label_ab, model_test_symbol = split_class_sample_in_time_series(time_step,train_data,feature,target,date,symbol)
    print("train sample number:%d, test samples number :%d "%(len(model_train_data),len(model_test_data)))
    prt_d = coll.defaultdict(int)
    for label_l in model_train_label:
        for idx,label in enumerate(label_l):
            if int(label) == 1:
                prt_d[idx] += 1
    for prt in prt_d:
        print("label %d,sample numbel %d."%(prt,prt_d[prt]))
    print "real_label num :",len(model_test_label)
    if len(model_train_data) == 0:
        logging.info('[DATA ERROR]lstm')
    model = make_lstm_classifier_model(model_train_data, model_train_label)
    predict= model.predict_proba(model_test_data)
    print "pre_label num : ",len(predict)
    return PredictResult(model_test_label,predict,False, model_test_label_ab, model_test_symbol)

def make_lstm_regession_model(model_train_data,model_train_label):
    time_steps = len(model_train_data[0])
    data_dim = len(model_train_data[0][0])
    print "time_steps ,data_dim: ", time_steps,data_dim
    model = Sequential()
    model.add(LSTM(32, return_sequences=True,
            input_shape=(time_steps, data_dim)))
    model.add(LSTM(32, return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(32,init="glorot_uniform"))
    model.add(Dense(1,input_dim=32,init="glorot_uniform"))
    model.add(Activation("linear"))
    sgd = SGD(lr=0.05, decay=1e-6, momentum=0.11, nesterov=True)
    model.compile(loss='mean_absolute_error', optimizer=sgd)
    model.fit(model_train_data,model_train_label,nb_epoch = 500, batch_size = 4,verbose = 1,shuffle=True)
    return model
    
def make_lstm_classifier_model(model_train_data,model_train_label):
    time_steps = len(model_train_data[0])
    data_dim = len(model_train_data[0][0])
    class_number = len(model_train_label[0])
    print "time_steps ,data_dim: ", time_steps,data_dim
    model = Sequential()
    model.add(LSTM(32, return_sequences=True,
            input_shape=(time_steps, data_dim)))
    model.add(LSTM(32, return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(32,init="glorot_uniform"))
    model.add(Dense(class_number,activation='softmax',input_dim=32,init="glorot_uniform"))
    #model.add(Activation("softmax"))
    sgd = SGD(lr=0.05, decay=1e-6, momentum=0.11, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    model.fit(model_train_data,model_train_label,nb_epoch = 500, batch_size = 4,verbose = 0,shuffle=True)
    return model

def get_time_sequence(date_list,time_step):
    """
    use for lstm data
    input:
    data_list = [1,2,3,4,5]
    time_step = 3
    output:
        [
        [1,2,3],
        [2,3,4],
        [3,4,5]
        ]
    """
    if len(date_list) < time_step:
        logging.info('get_time_sequence[para error]')
        return 
    len_time = len(date_list)
    seq_num = len_time - time_step + 1
    return_list = []
    for i in range(seq_num):
        temp_list = []
        for j in range(time_step):
            temp_list.append(date_list[i+j])
        return_list.append(temp_list)
    return return_list

def split_regress_sample_in_time_series(time_step,train_data,feature,target="3mr",date="date",symbol="symbol"):
    """
    input: pandas DataFrame
    ouput: (1) list of list predict (2) the label of the predict data
    """
    input_data = train_data.sort_values([symbol,date])
    alldate_l = train_data[date].unique()
    time_series_list = get_time_sequence(alldate_l,time_step)
    model_train_x = []
    model_train_y = []
    model_test_x = []
    model_test_y = []
    group_data = input_data.groupby(symbol)
    for key_value,group_by_key in group_data:
        dframe_x = group_by_key.dropna()
        ddata = dframe_x[date]
        ldata = dframe_x[target]
        dframe_x = dframe_x[feature]
        if dframe_x.size == 0:
            continue
        for time_series in time_series_list[:-1]:
            bool_l = ddata.isin(time_series)
            bool_arr = bool_l.value_counts()
            true_number = bool_arr.get(True) 
            if true_number == None:
                true_number = 0
            if time_step == true_number:
                model_train_x.append(list(dframe_x[bool_l].values))
                target_value = ldata[bool_l].values
                #the last time'target is the final target
                target_value = target_value[time_step-1]
                model_train_y.append([target_value])
        time_series = time_series_list[-1]
        bool_l = ddata.isin(time_series)
        bool_arr = bool_l.value_counts()
        true_number = bool_arr.get(True)
        if time_step == true_number:
            model_test_x.append(list(dframe_x[bool_l].values))
            target_value = ldata[bool_l].values
            #the last time'target is the final target
            target_value = target_value[time_step-1]
            model_test_y.append([target_value])
    return model_train_x,model_train_y,model_test_x,model_test_y 

def split_class_sample_in_time_series(time_step,train_data,feature,target="3mr",date="date",symbol="symbol"):
    #class label split
    """
    input: pandas DataFrame
    ouput: (1) list of list predict (2) the label of the predict data
    """
    input_data = train_data.sort_values([symbol,date])
    alldate_l = train_data[date].unique()
    time_series_list = get_time_sequence(alldate_l,time_step)
    model_train_x = []
    model_train_y = []
    model_test_x = []
    model_test_y = []
    model_test_y_ab = []
    model_test_symbol = []
    #the target value changed to the level value among 0,1,2,3,4
    #0:the lowest 4:the highest
    series_l = []
    group_data = input_data.groupby(date)
    for key_value,group_by_key in group_data:
        dframe_x = group_by_key.dropna()
        #the target value is discretized
        ldata = pd.qcut(dframe_x[target],5,[0,1,2,3,4])
        series_l.append(pd.Series(ldata))
    input_data['ab target'] = input_data[target]
    input_data[target] = pd.Series(pd.concat(series_l))
    #the tareget value are redefined.
    group_data = input_data.groupby(symbol)
    labels = [
            [1,0,0,0,0],
            [0,1,0,0,0],
            [0,0,1,0,0],
            [0,0,0,1,0],
            [0,0,0,0,1]
            ]
    for key_value,group_by_key in group_data:
        dframe_x = group_by_key.dropna()
        ddata = dframe_x[date]
        ldata = dframe_x[target]
        ldata_ab = dframe_x['ab target']
        #ldata = pd.qcut(dframe_x[target],5,[0,1,2,3,4])
        dframe_x = dframe_x[feature]
        if dframe_x.size == 0:
            continue
        for time_series in time_series_list[:-1]:
            bool_l = ddata.isin(time_series)
            bool_arr = bool_l.value_counts()
            true_number = bool_arr.get(True) 
            if time_step == true_number:
                model_train_x.append(list(dframe_x[bool_l].values))
                target_value = list(ldata[bool_l].values)
                #the last time'target is the final target
                target_value = target_value[time_step-1]
                model_train_y.append(labels[target_value])
        time_series = time_series_list[-1]
        bool_l = ddata.isin(time_series)
        bool_arr = bool_l.value_counts()
        true_number = bool_arr.get(True)
        if time_step == true_number:
            model_test_x.append(list(dframe_x[bool_l].values))
            target_value = ldata[bool_l].values
            target_value_ab = ldata_ab[bool_l].values
            #the last time'target is the final target
            target_value = target_value[time_step-1]
            target_value_ab = target_value_ab[time_step-1]
            model_test_y.append([target_value])
            model_test_y_ab.append(target_value_ab)
            model_test_symbol.append(key_value)
    return model_train_x, model_train_y, model_test_x, model_test_y, model_test_y_ab, model_test_symbol

if __name__ == "__main__":
    #pdata = read_file("data/financial_features.csv.test")
    pdata = read_file("data/financial_features.csv.alldate")
    #target = 'f_percent_6month_relative_strength'
    target = 'target'
    date = 'date'
    symbol='symbol'
    #pdata['mktcap'] = pdata['mktcap']/100
    #pdata['3mr'] = pdata['3mr']/100
    lstm_combine(5, 3, pdata, target, date, symbol)
    """
    real = [[0],[0],[1],[1],[2],[2]]
    pre = [[0.2,0.3,0.4],[0.9,0.3,0.4],[0.9,0.3,0.4],[0.2,0.5,0.4],[0.2,0.5,0.4],[0.2,0.3,0.4]]
    rtn = [0.9,0.5,0.9,0.5,0.9,0.5]
    symbol = ['a','b','c','d','e','f']
    r = PredictResult(real,pre,False,rtn,symbol)
    r.to_file("test",[12,13,14],6,3,False)
    """
