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

    def __init__(self, data_pool, data_pool_xgb=None, shortcut = None, 
                 bld=None, pv=None, ev=None, price_buy=None, price_sell=None,
                 bld_kws=None, pv_kws=None, ev_kws=None, price_buy_kws=None, price_sell_kws=None):

        self.data_pool = data_pool
        self.data_pool_xgb = data_pool_xgb
        
        self.set_predictor(shortcut=shortcut,
                 bld=None, pv=None, ev=None, price_buy=None, price_sell=None,
                 bld_kws=None, pv_kws=None, ev_kws=None, price_buy_kws=None, price_sell_kws=None)
        
 
    def get_prediction(self, t, K, delta):
        pred = dict()
        for var in self.predictors.keys():
            predictor = self.predictors[var]
            if predictor is not None:
                pred[var] = predictor.get_prediction(t, K, delta)
            else:
                pred[var] = None
        return pred


    def set_predictor(self, shortcut, **predictors):

        predictor_tmp = predictors

        short_cut = {
            #all GT as the upper bound
            "GT": {
                "bld": Predictor_load_GT, "bld_kws": None,
                "pv": Predictor_load_GT, "pv_kws": None,
                "ev": Predictor_ev_GT, "ev_kws": None,
                "price_buy": Predictor_tou_SDGE_DA, "price_buy_kws": None,
                "price_sell": None, "price_sell_kws": None
                },
            "Simple": {
                "bld": Predictor_load_Simple, "bld_kws": {"rule": "week", "num": 4, "exp_alpha": 0.1},
                "pv": Predictor_load_GT, "pv_kws": None,# Predictor_load_Simple, "pv_kws": {"rule": "day", "num": 3, "exp_alpha": 0.1},
                "ev": Predictor_ev_GT, "ev_kws": None,
                #"ev": Predictor_ev_Simple, "ev_kws": {"rule": "week", "num": 1},
                "price_buy": Predictor_tou_SDGE_DA, "price_buy_kws": None,
                "price_sell": None, "price_sell_kws": None
                },
            # all naive as the lower bound
            "Naive": {
                "bld": Predictor_load_Simple, "bld_kws": {"rule": "naive"},
                "pv": Predictor_load_GT, "pv_kws": None, # Predictor_load_Simple, "pv_kws": {"rule": "naive"},
                "ev": Predictor_ev_GT, "ev_kws": None, # Predictor_ev_Simple, "ev_kws": {"rule": "naive"},
                "price_buy": Predictor_tou_SDGE_DA, "price_buy_kws": None,
                "price_sell": None, "price_sell_kws": None
                },
            # UPDATE: 2023-07-21 LunLong
            # add the implementation of XGBoost prediction
            # "pv": Predictor_load_XGBoost, "pv_kws": {"xgb_prediction_path":'D:/Codes/GIthub_repo/Energy_grid/data/load_forecast/XGB/PV_sum_XGBoost_prediction.csv'},
            "Prediction":{ 
                # the file path work only on local environment
                "bld": Predictor_load_XGBoost, "bld_kws":{"data_pool_xgb":self.data_pool_xgb},
                "pv": Predictor_load_GT, "pv_kws": None, #  Predictor_load_Simple, "pv_kws": {"rule": "day", "num": 3, "exp_alpha": 0.1},
                "ev": Predictor_ev_GT, "ev_kws": None,
                "price_buy": Predictor_tou_SDGE_DA, "price_buy_kws": None,
                "price_sell": None, "price_sell_kws": None
                },
                
        }


        if shortcut in short_cut.keys():
            predictor = short_cut[shortcut]
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
    def __init__(self, data_pool):
        self.data_pool = data_pool
    def get_prediction(self, **kw):
        pass

# [Yi, 2023/03/03] combine bld & pv predictor
class Predictor_load_GT(Load_Predictor):
    def __init__(self, data_pool, load_type):
        super().__init__(data_pool)
        assert load_type in ["bld", "pv"]
        self.load_type = load_type
        
    def get_prediction(self, t, K, delta):
        # [Yi, 2022/12/22]: .loc with datetime is closed both sides
        pred = self.data_pool.data[f"load_{self.load_type}"].loc[t:t+timedelta(hours=(K-1)*delta)]
        #print("get load prediction gt method called")
        # FIXME: align pred index with delta?
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
    def __init__(self, data_pool, load_type, rule, **rule_kws):
        super().__init__(data_pool, load_type)
        self.rule = rule    # suggest: bld: "week"; pv: "day"
        self.num = rule_kws.get("num", 1)   # suggest: bld: 1, pv: 3
        self.exp_alpha = rule_kws.get("exp_alpha", None)    # suggest: 0.1

    def get_prediction(self, t, K, delta):
        
        t_prev = t - timedelta(hours = delta)
        data = self.data_pool.data[f"load_{self.load_type}"]
        last = data.loc[t_prev]   # load at last step
        
        if self.rule == "naive":
            pred = np.array([last] * K)
        else:
            ts = find_dates(t, K, delta, self.rule, self.num)
            ts = data.index.intersection(ts)
            if len(ts) < K:
                pred = np.array([last] * K)
            else:
                pred_ref = data.loc[ts]
                pred = pred_ref.groupby(pred_ref.index - pred_ref.index.floor(freq="D")).agg(np.nanmean)
                idx = pd.date_range(t, t+timedelta(hours=K*delta), freq=f"{delta}H", inclusive="left")
                idx = idx - idx.floor(freq="D")
                pred = pred.loc[idx].values

                alpha = self.exp_alpha
                last_weights = np.exp(- alpha*(np.arange(K)+1)) if alpha is not None else 0
                pred = last_weights * last + (1-last_weights) * pred
        
        assert len(pred) == K
        
        #print("Simple method called and predictions are as follows:")
        #print(type(pred))
        #print(pred)
        return pred

    # UPDATE: 2023-07-21 LunLong
    # add the implementation of XGBoost prediction
    #class Predictor_load_XGBoost(Predictor_load_GT):

    # [Lunlong 2023/08/18] fix a bug here 
class Predictor_load_XGBoost(Load_Predictor):
    def __init__(self,data_pool,load_type,**kw):
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

class Predictor_ev_Simple(Predictor_ev_GT):
    def __init__(self, data_pool, rule, **rule_kws):
        super().__init__(data_pool)
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



