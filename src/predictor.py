import os
import sys

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

"""
============================= UPDATES =============================
(write down updates on the code - the last updated one at the first)
===================================================================

# [Yi, 2023/03/08]
- implement [Predictor_ev_Simple] same rules available as bld & pv
- TODO: allow to combine different predictor for different load type

# [Yi, 2023/03/04]
- implement simple rule-based predictor [Predictor_Simple]
    rule: "naive", "day", "week", "day_in_year", "week_in_year"
    rule_kws: "num", "exp_alpha"
    FIXME:
        - now rule & rule_kws are fixed - enable to accept specification
        - allow to combine different predictor for different load type, e.g., 

# [Yi, 2023/03/03]
- combine bld & pv predictor as [Predictor_load_XXX]

# [Yi, 2023/02/11]
- Now, only Predictor_GT works
"""

class Predictor():

    def __init__(self, data_pool, data_pool_xgb=None, shortcut = None, shift=None,shift_ratio=0,delta=0.25,
                 bld=None, pv=None, ev=None, price_buy=None, price_sell=None,
                 bld_kws=None, pv_kws=None, ev_kws=None, price_buy_kws=None, price_sell_kws=None):

        self.data_pool = data_pool
        self.data_pool_xgb = data_pool_xgb
        #self.data_pool_noise=None
        
        self.set_predictor(shortcut=shortcut,shift=shift,shift_ratio=shift_ratio,
                 bld=bld, pv=pv, ev=ev, price_buy=price_buy, price_sell=price_sell,
                 bld_kws=bld_kws, pv_kws=pv_kws, ev_kws=ev_kws, 
                 price_buy_kws=price_buy_kws, price_sell_kws=price_sell_kws)
        
 
    def get_prediction(self, t, K, delta):
        pred = dict()
        for var in self.predictors.keys():
            predictor = self.predictors[var]
            if predictor is not None:
                pred[var] = predictor.get_prediction(t, K, delta)
            else:
                pred[var] = None
        return pred


    def set_predictor(self, shortcut, bld=None, pv=None, ev=None, price_buy=None, price_sell=None, shift=None,shift_ratio=0,
                 bld_kws=None, pv_kws=None, ev_kws=None, price_buy_kws=None, price_sell_kws=None,**predictors):

        predictor_tmp = predictors

        
        short_cut_dic_1 = {
            #all GT as the upper bound
            "GT": {
                "bld": Predictor_load_GT, "bld_kws": None,
                "pv": Predictor_load_GT, "pv_kws": None,
                "ev": Predictor_ev_GT, "ev_kws": None,
                "price_buy": Predictor_tou_SDGE_DA, "price_buy_kws": None,
                "price_sell": None, "price_sell_kws": None
                },
            "Simple": {
                "bld": Predictor_load_Simple, "bld_kws": {"rule": "week", "num": 4, "exp_alpha": 0.1},#"bld": Predictor_load_GT, "bld_kws": None,
                #"pv":  Predictor_load_Simple, "pv_kws": {"rule": "day", "num": 3, "exp_alpha": 0.1},
                "ev": Predictor_ev_GT, "ev_kws": None,
                "pv": Predictor_load_GT, "pv_kws": None,
                #"ev": Predictor_ev_Simple, "ev_kws": {"rule": "week", "num": 1},
                "price_buy": Predictor_tou_SDGE_DA, "price_buy_kws": None,
                "price_sell": None, "price_sell_kws": None
                },
            # all naive as the lower bound
            "Naive": {
                "bld": Predictor_load_Simple, "bld_kws": {"rule": "naive"},#"bld": Predictor_load_GT, "bld_kws": None,
                "pv":  Predictor_load_Simple, "pv_kws": {"rule": "naive"},
                "ev": Predictor_ev_GT, "ev_kws": None, # Predictor_ev_Simple, "ev_kws": {"rule": "naive"},
                "price_buy": Predictor_tou_SDGE_DA, "price_buy_kws": None,
                "price_sell": None, "price_sell_kws": None
                },
            # UPDATE: 2023-07-21 LunLong
            # add the implementation of XGBoost prediction
            # "pv": Predictor_load_XGBoost, "pv_kws": {"xgb_prediction_path":'D:/Codes/GIthub_repo/Energy_grid/data/load_forecast/XGB/PV_sum_XGBoost_prediction.csv'},
            "Prediction":{ 
                "bld": Predictor_load_XGBoost, "bld_kws":{"data_pool_xgb":self.data_pool_xgb},#"bld": Predictor_load_GT, "bld_kws": None
                "pv": Predictor_load_GT, "pv_kws": None, #  Predictor_load_Simple, "pv_kws": {"rule": "day", "num": 3, "exp_alpha": 0.1},
                "ev": Predictor_ev_GT, "ev_kws": None,
                "price_buy": Predictor_tou_SDGE_DA, "price_buy_kws": None,
                "price_sell": None, "price_sell_kws": None
                },
            "Disturbance":{
                "bld": Predictor_load_Noise, "bld_kws":bld_kws,#"bld": Predictor_load_GT, "bld_kws": None
                "pv": Predictor_load_GT, "pv_kws": None, #  Predictor_load_Simple, "pv_kws": {"rule": "day", "num": 3, "exp_alpha": 0.1},
                "ev": Predictor_ev_GT, "ev_kws": None,
                "price_buy": Predictor_tou_SDGE_DA, "price_buy_kws": None,
                "price_sell": None, "price_sell_kws": None
                },      
        }

        short_cut_dic_2 = {
            #all GT as the upper bound
            "GT": {
                "bld": Predictor_load_GT, "bld_kws": None,
                "pv": Predictor_load_GT, "pv_kws": None,
                "ev": Predictor_ev_GT, "ev_kws": None,
                "price_buy": Predictor_tou_SDGE_DA, "price_buy_kws": None,
                "price_sell": None, "price_sell_kws": None
                },
            "Simple": {
                "bld": Predictor_load_Simple_shift, "bld_kws": {"rule": "week", "num": 4, "exp_alpha": 0.1, "shift_ratio":shift_ratio},#"bld": Predictor_load_GT, "bld_kws": None,
                "pv": Predictor_load_GT, "pv_kws": None,# Predictor_load_Simple, "pv_kws": {"rule": "day", "num": 3, "exp_alpha": 0.1},
                "ev": Predictor_ev_GT, "ev_kws": None,
                #"ev": Predictor_ev_Simple, "ev_kws": {"rule": "week", "num": 1},
                "price_buy": Predictor_tou_SDGE_DA, "price_buy_kws": None,
                "price_sell": None, "price_sell_kws": None
                },
            # all naive as the lower bound
            "Naive": {
                "bld": Predictor_load_Simple_shift, "bld_kws": {"rule": "naive","shift_ratio":shift_ratio},#"bld": Predictor_load_GT, "bld_kws": None,
                "pv": Predictor_load_GT, "pv_kws": None, # Predictor_load_Simple, "pv_kws": {"rule": "naive"},
                "ev": Predictor_ev_GT, "ev_kws": None, # Predictor_ev_Simple, "ev_kws": {"rule": "naive"},
                "price_buy": Predictor_tou_SDGE_DA, "price_buy_kws": None,
                "price_sell": None, "price_sell_kws": None
                },
            # UPDATE: 2023-07-21 LunLong
            # add the implementation of XGBoost prediction
            # "pv": Predictor_load_XGBoost, "pv_kws": {"xgb_prediction_path":'D:/Codes/GIthub_repo/Energy_grid/data/load_forecast/XGB/PV_sum_XGBoost_prediction.csv'},
            "Prediction":{ 
                "bld": Predictor_load_XGBoost_shift, "bld_kws":{"data_pool_xgb":self.data_pool_xgb, "shift_ratio":shift_ratio},#"bld": Predictor_load_GT, "bld_kws": None
                "pv": Predictor_load_GT, "pv_kws": None, #  Predictor_load_Simple, "pv_kws": {"rule": "day", "num": 3, "exp_alpha": 0.1},
                "ev": Predictor_ev_GT, "ev_kws": None,
                "price_buy": Predictor_tou_SDGE_DA, "price_buy_kws": None,
                "price_sell": None, "price_sell_kws": None
                },
            "Disturbance":{
                "bld": Predictor_load_Noise, "bld_kws":bld_kws,#"bld": Predictor_load_GT, "bld_kws": None
                "pv": Predictor_load_GT, "pv_kws": None, #  Predictor_load_Simple, "pv_kws": {"rule": "day", "num": 3, "exp_alpha": 0.1},
                "ev": Predictor_ev_GT, "ev_kws": None,
                "price_buy": Predictor_tou_SDGE_DA, "price_buy_kws": None,
                "price_sell": None, "price_sell_kws": None
                },      
        }
        if shift is None:
            short_cut_dic=short_cut_dic_1
        elif shift:
            short_cut_dic=short_cut_dic_2
        else:
            raise Exception("unrecognized shift")

        if shortcut in short_cut_dic.keys():
            predictor = short_cut_dic[shortcut]
            # TODO: enable to update shortcut settings with corresponding kwargs in predictor_tmp 
        else:
            predictor = predictor_tmp


        def set_predictor_help(key):
            p = predictor[key]
            kw = predictor[key+"_kws"]
            if p is None:
                return None
            if kw is None:
                kw = {}
            if key in ["bld", "pv"]:
                kw.update({"load_type": key})
            return p(self.data_pool, **kw)


        self.predictors = {
            "load_bld": set_predictor_help("bld"), 
            "load_pv": set_predictor_help("pv"),
            "ev_sessions": set_predictor_help("ev"),
            "energy_price_buy": set_predictor_help("price_buy"),
            "energy_price_sell": set_predictor_help("price_sell"),
        }


class Load_Predictor():
    
    def __init__(self, data_pool, delta=0.25):
        self.data_pool = data_pool
        #self.data_pool_noise = None
    def get_prediction(self, **kw):
        pass

# [Yi, 2023/03/03] combine bld & pv predictor
class Predictor_load_GT(Load_Predictor):
    def __init__(self, data_pool, load_type,delta=0.25):
        super().__init__(data_pool,delta=0.25)
        assert load_type in ["bld", "pv"]
        self.load_type = load_type
        #self.data_pool_noise=None
        
    def get_prediction(self, t, K, delta):
        # [Yi, 2022/12/22]: .loc with datetime is closed both sides
        pred = self.data_pool.data[f"load_{self.load_type}"].loc[t:t+timedelta(hours=(K-1)*delta)]
        #print("get load prediction gt method called")
        # FIXME: align pred index with delta?
        return pred
    
    
# [Lunlong, 2023/08/21] add predictor of random noise
class Predictor_load_Noise(Predictor_load_GT):

    def __init__(self, data_pool, load_type, rule, delta=0.25, **rule_kws):
        super().__init__(data_pool, load_type, delta=0.25)
        self.rule = rule    
        self.loc = rule_kws.get("loc", 0.0)   # suggest: 0
        self.scale = rule_kws.get("scale", 0.03)    # suggest: 0.03
        self.hourly_sign_uniform = rule_kws.get("hourly_sign_uniform",True)
        self.delta = delta   # suggest: 0.03
        # to cal the noise_table at once
        # in order to avoid the inconsistency when the duration of loaded data changed,
        #   cal the table longer than needed
        
        if self.scale>0.5:
            raise Warning("self.scale higher than 0.5, may lead to infasiblility")
        # Define the start and end timestamps
        ta = pd.Timestamp('2018-01-01 00:00:00')
        td = pd.Timestamp('2021-01-02 00:00:00')
        # fix the random seed
        np.random.seed(42)
        # Generate the timestamps with gaps
        timestamps = pd.date_range(start=ta, end=td, freq=f'{self.delta}H')
        # Generate random noises
        if self.rule in ["normal","normal_pos","normal_neg"]:
            noises = np.random.normal(loc=0, scale=0.1, size=len(timestamps))
        elif self.rule in ["uniform","uniform_pos","uniform_neg"]:
            noises = np.random.uniform(low=-1, high=1, size=len(timestamps))
        else:
            raise Warning("invalid disturbance rule")
        # scale the noise
        absolute_average = np.mean(np.abs(noises))
        adjustment_factor = self.scale / absolute_average
        noises=noises*adjustment_factor
        # cut down coef exceeding the range
        np.clip(noises,-1.0,1.0)
        # adjust the noise distribution according to the rule
        if self.rule in ["normal_pos","uniform_pos"]:
            noises=abs(noises)
        elif self.rule in ["normal_neg","uniform_neg"]:
            noises=-abs(noises)
        noise_series = pd.Series(data=noises, index=timestamps)
        
        # [Lunlong, 2023/09/18] add new constraint to the U series, keeping the sign of the noise the 
        #       same within the same hour
        #   This is implemented in an easy way:
        #       When hour is odd, sign is negative, while even and positive
        def sign_uniform_hour(x, **kwargs):
            time_index=x.index
            hour=time_index.hour
            if hour%2==1:
                uniformed_x=abs(x)*-1
            else:
                uniformed_x=abs(x)
            return uniformed_x
        
        if self.rule in ['normal','uniform'] and self.hourly_sign_uniform==True:
            #noise_series=noise_series.apply(sign_uniform_hour)
            noise_series_df=pd.DataFrame(noise_series)
            noise_series_df.index=pd.to_datetime(noise_series_df.index)
            noise_series_df["hour"]=timestamps.hour
            print(noise_series_df.info())
            #noise_series_df=noise_series_df.apply(sign_uniform_hour,axis=0)
            noise_series_df.loc[:,0]=noise_series_df.apply(lambda x: abs(x[0])*-1 if x['hour']%2==1 else abs(x[0]),axis=1)
            noise_series=noise_series_df.loc[:,0]
        
        tstart=min(self.data_pool.data[f"load_{self.load_type}"].index)
        tend=max(self.data_pool.data[f"load_{self.load_type}"].index)
        
        pred_ref=self.data_pool.data[f"load_{self.load_type}"]
        noise_ref=noise_series.loc[tstart:tend]
        assert len(noise_ref)==len(pred_ref)
        
        #self.data_pool=pred_ref.copy()
        self.data_pool.data[f"load_{self.load_type}_noise"]=\
            self.data_pool.data[f"load_{self.load_type}"]+self.data_pool.data[f"load_{self.load_type}"]*noise_ref

        
    def get_prediction(self, t, K, delta):
        
        if self.rule not in ["normal","normal_pos","normal_neg","uniform","uniform_pos","uniform_neg"]:
            raise Exception("Noise generating rule not implemented: ",self.rule)
        
        pred=self.data_pool.data[f"load_{self.load_type}_noise"].loc[t:t+timedelta(hours=(K-1)*delta)]
        assert len(pred) == K
        '''
        if self.scale>0.5:
            raise Warning("self.scale higher than 0.5, may lead to infasiblility")
        
        if self.rule in ["normal","normal_pos","normal_neg"]:
            # make most of the coef constrainted within [-1,1] by keep scale=1/3 
            coef=np.random.normal(loc=self.loc, scale=self.scale, size=len(pred_ref))
            absolute_average = np.mean(np.abs(coef))
            adjustment_factor = self.scale / absolute_average
            coef=coef*adjustment_factor
            # cut down coef exceeding the range
            np.clip(coef,-1,1)
            if self.rule=="normal":
                pred=pred_ref*coef+pred_ref
            elif self.rule=="normal_pos":
                coef=abs(coef)
                pred=pred_ref*coef+pred_ref
            elif self.rule=="normal_neg":
                coef=-abs(coef)
                pred=pred_ref*coef+pred_ref
        
        elif self.rule == "uniform":
            coef=np.random.uniform(low=self.scale*(-2), high=self.scale*2, size=len(pred_ref))
            pred=pred_ref*coef+pred_ref
            
        elif self.rule == "uniform_pos":
            coef=np.random.uniform(low=0, high=self.scale*2, size=len(pred_ref))
            pred=pred_ref*coef+pred_ref
            
        elif self.rule == "uniform_neg":
            coef=np.random.uniform(low=self.scale*(-2), high=0, size=len(pred_ref))
            pred=pred_ref*coef+pred_ref
        '''
        
        return pred



class Predictor_ev_GT(Load_Predictor):
    def get_prediction(self, t, K, delta):
        ev_sessions = self.data_pool.data.get("ev_sessions")
        if ev_sessions is None:
            return None       
        t0, t1 = t+timedelta(hours=delta), t+timedelta(hours=K*delta)
        sig = (ev_sessions["ta"] >= t0) & (ev_sessions["ta"] < t1)
        pred = ev_sessions.loc[sig].copy()
        return pred

class Predictor_tou_CAISO(Load_Predictor):
    def get_prediction(self, t, K, delta):
        price = np.zeros(K)
        for k in range(K):
            price[k] = self._get_price_t(t+timedelta(hours=k*delta))
        return price

    def _get_price_t(self, t):
        h = t.hour
        if h < 9 or (h>=14 and h<16) or h>=21:
            return 0.13
        if (h>=9 and h<14):
            return 0.11
        return 0.34
    
class Predictor_tou_SDGE_DA(Predictor_tou_CAISO):
    def _get_price_t(self, t):
        # FIXME: we use a fixed day-ahead price, i.e., everyday has the same curve
        # data source: SDGE hitorical data, average of year 2019,
        #   http://www.energyonline.com/Data/GenericData.aspx?DataId=20
        #   the original unit is [$/MWh], average rate is 0.04 $/kWh
        # we normalize the data so that it has the same mean with CAISO TOU, whose mean is 0.17 $/kWh
        #print("Method called: Predictor_tou_SDGE_DA._get_price_t")
        hour_price =np.array(
            [0.152, 0.143, 0.137, 0.137, 0.145, 0.172, 0.204, 0.185, 0.144, 0.123, 0.113, 0.109,
             0.110, 0.116, 0.127, 0.148, 0.181, 0.244, 0.279, 0.294, 0.249, 0.213, 0.181, 0.163])
        return hour_price[t.hour]
    


def find_dates(t, K, delta, rule, num, ev=False):
    
    # [Yi, 2023/03/08] add arg [ev]
    #   if ev == False, i.e., for load pred, return pd.date_range
    #   if ev == True, i.e., for ev pred, only return the start & end time


    if rule in ["day", "day_in_year"]:
        if rule == "day":
            tstart = t - timedelta(days = num)
        else:
            tstart = t - timedelta(days=365+round(num/2))
        tend = tstart + timedelta(days=num)

        if ev == False:
            ts = pd.date_range(start=tstart, end=tend, freq=f"{delta}H", inclusive="left")
        else:
            ts = [(tstart, tend)]
        
    
    elif rule in ["week", "week_in_year"]:
        if rule == "week":
            tstart = t - timedelta(days = 7*num)
        else:
            tstart = t - timedelta(days = (53+round(num/2))*7)
        
        ts = None if ev == False else []
        for i in range(num):
            t0 = tstart + timedelta(days = 7*i)
            t1 = t0 + timedelta(days=1)
            if ev == False:
                ts_i = pd.date_range(start=t0, end=t1, freq=f"{delta}H", inclusive="left")
                ts = ts_i if ts is None else ts.union(ts_i)
            else:
                ts.append((t0, t1))

    else:
        raise Exception(f"Rule [{rule}] has not been implemented")

    if ev == False:
        assert len(ts) == 24/delta*num
    
    return ts    



""" Support following rules
        [x] can get from rule_kw["num"], deault = 1
    - "naive": all as the last time step
    - "day": average of past [x] days
    - "week": average of the same weekday in past [x] weeks
    - "day_in_year": average of [x] days centered around the day 365 days ago
    - "week_in_year": average of the same weekday in [x] weeks centered around the day 365 days ago
""" 


class Predictor_load_Simple(Predictor_load_GT):
    def __init__(self, data_pool, load_type, rule, delta=0.25, **rule_kws):
        super().__init__(data_pool, load_type, delta=0.25)
        self.rule = rule    # suggest: bld: "week"; pv: "day"
        self.num = rule_kws.get("num", 1)   # suggest: bld: 1, pv: 3
        self.exp_alpha = rule_kws.get("exp_alpha", None)    # suggest: 0.1

    def get_prediction(self, t, K, delta):
        
        t_prev = t - timedelta(hours = delta)
        data = self.data_pool.data[f"load_{self.load_type}"]
        last = data.loc[t_prev]   # load at last step
        
        # [Lunlong 2023/08/23] fix a bug, when rule==naive, return a series rather than ndarray
        #                      same bug when not naive
        if self.rule == "naive":
            idx = pd.date_range(t, t+timedelta(hours=K*delta), freq=f"{delta}H", inclusive="left")
            #idx = idx - idx.floor(freq="D")
            pred = pd.Series(last, index=idx)
        else:
            ts = find_dates(t, K, delta, self.rule, self.num)
            ts = data.index.intersection(ts)
            if len(ts) < K:
                pred = np.array([last] * K)
            else:
                pred_ref = data.loc[ts]
                pred = pred_ref.groupby(pred_ref.index - pred_ref.index.floor(freq="D")).agg(np.nanmean)
                idx_ori = pd.date_range(t, t+timedelta(hours=K*delta), freq=f"{delta}H", inclusive="left")
                idx = idx_ori - idx_ori.floor(freq="D")
                pred = pred.loc[idx].values
                alpha = self.exp_alpha
                last_weights = np.exp(- alpha*(np.arange(K)+1)) if alpha is not None else 0
                pred_values = last_weights * last + (1-last_weights) * pred              
                pred = pd.Series(pred_values, index=idx_ori)
        
        assert len(pred) == K

        return pred


class Predictor_load_Simple_shift(Predictor_load_GT):
    def __init__(self, data_pool, load_type, rule, delta=0.25, **rule_kws):
        super().__init__(data_pool, load_type, delta=0.25)
        self.rule = rule    # suggest: bld: "week"; pv: "day"
        self.num = rule_kws.get("num", 1)   # suggest: bld: 1, pv: 3
        self.exp_alpha = rule_kws.get("exp_alpha", None)    # suggest: 0.1
        self.shift_ratio=rule_kws.get("shift_ratio",0)

    def get_prediction(self, t, K, delta):
        
        t_prev = t - timedelta(hours = delta)
        data = self.data_pool.data[f"load_{self.load_type}"]
        last = data.loc[t_prev]   # load at last step
        
        # [Lunlong 2023/08/23] fix a bug, when rule==naive, return a series rather than ndarray
        #                      same bug when not naive
        if self.rule == "naive":
            idx = pd.date_range(t, t+timedelta(hours=K*delta), freq=f"{delta}H", inclusive="left")
            #idx = idx - idx.floor(freq="D")
            pred = pd.Series(last, index=idx)
        else:
            ts = find_dates(t, K, delta, self.rule, self.num)
            ts = data.index.intersection(ts)
            if len(ts) < K:
                pred = np.array([last] * K)
            else:
                pred_ref = data.loc[ts]
                pred = pred_ref.groupby(pred_ref.index - pred_ref.index.floor(freq="D")).agg(np.nanmean)
                idx_ori = pd.date_range(t, t+timedelta(hours=K*delta), freq=f"{delta}H", inclusive="left")
                idx = idx_ori - idx_ori.floor(freq="D")
                pred = pred.loc[idx].values
                alpha = self.exp_alpha
                last_weights = np.exp(- alpha*(np.arange(K)+1)) if alpha is not None else 0
                pred_values = (last_weights * last + (1-last_weights) * pred)*(1+self.shift_ratio)     
                pred = pd.Series(pred_values, index=idx_ori)
        
        assert len(pred) == K

        return pred
    # UPDATE: 2023-07-21 LunLong
    # add the implementation of XGBoost prediction
    #class Predictor_load_XGBoost(Predictor_load_GT):

    # [Lunlong 2023/08/18] fix a bug here 
class Predictor_load_XGBoost(Load_Predictor):
    def __init__(self,data_pool,load_type, delta=0.25,**kw):
        assert load_type in ["bld", "pv"]
        self.load_type = load_type
        self.data_pool_xgb=kw["data_pool_xgb"]
        
    def get_prediction(self, t, K, delta):
        pred = self.data_pool_xgb.data[f"load_{self.load_type}"].loc[t:t+timedelta(hours=(K-1)*delta)]
        return pred
    
    # previous with terrible bug
    '''
    def __init__(self, data_pool, load_type, xgb_prediction_path):
        super().__init__(data_pool,load_type)
        assert load_type in ["bld", "pv"]
        #self.load_type = load_type
        self.xgb_prediction_path = xgb_prediction_path
        
    def get_prediction(self, t, K, delta):
        pred_ref=pd.read_csv(self.xgb_prediction_path)
        pred_ref['DateTime']=pd.to_datetime(pred_ref['DateTime'])
        pred_ref=pred_ref.set_index('DateTime')
        if 'RealPower' in pred_ref.columns:
            pred_ref=pred_ref.drop(columns=['RealPower'])
        
        # [Yi, 2022/12/22]: .loc with datetime is closed both sides
        pred = pred_ref.loc[t:t+timedelta(hours=(K-1)*delta)].values
    
        #print(pred)
        pred_flatten=pred.flatten()
        # FIXME: align pred index with delta?
        #print("XGB method called and predictions are as follows:")
        #print(type(pred_flatten))
        #print(pred_flatten)
        return pred_flatten
    '''
class Predictor_load_XGBoost_shift(Load_Predictor):
    def __init__(self,data_pool,load_type, delta=0.25, **kw):
        assert load_type in ["bld", "pv"]
        self.load_type = load_type
        self.data_pool_xgb=kw["data_pool_xgb"]
        self.shift_ratio=kw["shift_ratio"]
        
    def get_prediction(self, t, K, delta):
        pred = self.data_pool_xgb.data[f"load_{self.load_type}"].loc[t:t+timedelta(hours=(K-1)*delta)]
        pred = pred+pred*self.shift_ratio
        return pred
    
class Predictor_ev_Simple(Predictor_ev_GT):
    def __init__(self, data_pool, rule, delta=0.25, **rule_kws):
        super().__init__(data_pool, delta=0.25)
        self.rule = rule    # suggest: ev: "week"
        self.num = rule_kws.get("num", 1)   # suggest: ev: 1
    
    def get_prediction(self, t, K, delta):
        
        ev_sessions = self.data_pool.data.get("ev_sessions")

        if ev_sessions is None:
            return None       
        if self.rule == "naive":
            return None
        
        ts = find_dates(t, K, delta, self.rule, self.num, ev=True)
        to_concat = []
        for t0, t1 in ts:
            sig = (ev_sessions["ta"] >= t0) & (ev_sessions["ta"] < t1)
            to_concat.append(ev_sessions.loc[sig].copy())
        pred_before_sample = pd.concat(to_concat, axis=0)

        if self.num > 1:
            N = len(pred_before_sample)
            n_sample = round(N / self.num)
            rand_seed = (t - datetime(2000,1,1,0,0)).total_seconds() % 2023 # set an arbitrary random seed
            np.random.seed(int(rand_seed))
            idx_sample = np.random.choice(N, size=n_sample, replace=False)
            pred = pred_before_sample.iloc[idx_sample].copy()
        else:
            pred = pred_before_sample.copy()
        
        # change ta, td of pred EVs
        # FIXME: only works when K*delta <= 1 day
        # idea : t = t + ceil_to_day(t_now - t)

        day_diff = (t - pred["ta"]).dt.ceil(freq="D")
        pred["ta"] = pred["ta"] + day_diff
        pred["td"] = pred["td"] + day_diff
        # "td_actual" will not be used

        return pred



