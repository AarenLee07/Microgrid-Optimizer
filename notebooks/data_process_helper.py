import os
import sys
import numpy as np
import pandas as pd
import itertools

import os
import sys
import copy
from multiprocessing import Process, freeze_support
from threading import Thread
from time import sleep

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import copy
import re


def generate_exp_table(params_product,params_const,params_numerate,save_path,prori_key=None):
    
    comb=list(itertools.product(*params_numerate))
    combinations=list()
    for i in range(len(comb)):
        params={}
        params.update(params_product)
        params.update(params_const)
        for j in range(len(comb[i])):
            params.update(comb[i][j])
        
        print(params)
        combinations.extend(list(itertools.product(*params.values())))
        
    
    df = pd.DataFrame(combinations, columns=params.keys())
    if prori_key:
        df.sort_values(by=prori_key,ignore_index=True)
    df.index.name='id'
    df.to_excel(save_path, index=True)
    
    return df

def convert_table(ta,td,EV,delta_h,e_targ_rule='average'):
    hours=(td-ta).total_seconds()/3600
    assert hours%delta_h==0
    rounds=int(hours/delta_h)
    EV_convert=pd.DataFrame(columns=['t_start','t_end','ev_I_onsite','total_e_targ','ave_e_targ'])
    ta_i=ta
    td_i=ta+timedelta(hours=delta_h)
    EV['e_req']=None
    if e_targ_rule=='average':
        EV['p_ave_req']=EV.apply(lambda x: x['e_targ']/((x['td']-x['ta']).total_seconds()/3600),axis=1)
    for i in range(rounds):
        EV_tmp=EV[(EV['ta']<=td_i)&(EV['td']>ta_i)].copy()
        EV_convert.loc[i,'t_start']=ta_i
        EV_convert.loc[i,'t_end']=td_i
        if len(EV_tmp)==0:
            EV_convert.loc[i,'ev_I_onsite']=0
            EV_convert.loc[i,'total_e_targ']=0
            EV_convert.loc[i,'ave_e_targ']=0
        else:
            EV_tmp['e_req']=EV_tmp.apply(lambda x: x['p_ave_req']*delta_h,axis=1)#(td_i-x['ta']).total_seconds()/3600,
            EV_convert.loc[i,'ev_I_onsite']=len(EV_tmp)
            EV_convert.loc[i,'total_e_targ']=EV_tmp['e_req'].sum()
            EV_convert.loc[i,'ave_e_targ']=EV_tmp['e_req'].sum()/len(EV_tmp)
        ta_i+=timedelta(hours=delta_h)
        td_i+=timedelta(hours=delta_h)
    return EV_convert

def get_pred_ev(mpc,ta,td,freq_h=24):
    hours=(td-ta).total_seconds()/3600
    assert hours%freq_h==0
    rounds=int(hours/freq_h)
    pred_ev=[]
    t=ta
    for i in range(rounds):
        pred_ev.append(mpc.predictor.get_prediction(t,96,delta=0.25)['ev_sessions'])
        t=t+timedelta(hours=freq_h)
    pred_ev=pd.concat(pred_ev)
    
    return pred_ev