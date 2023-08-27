import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

from utils.utils import get_data_path
data_path = get_data_path()

"""
============================= UPDATES =============================
(write down updates on the code - the last updated one at the first)
===================================================================

[Yi, 2023/03/22]
- fix a bug in rescale EVs (ev_to_bld)
    original eperiment with ev_to_bld = 0.2, is actually around ev_to_bld = 0.25 
    (or, for exact reproduction, ev_to_bld = None)
"""


class DataLoader():

    """
    this class provides some general methods, 
    regardless of what data set you are to use
    to my understanding: when shifting to another dataset,
    you need to specify how to access [series of bld, pv, and df of EV]
        i.e., overload method [load_data_tmp]
    then, other process should be standard, including
    - choose by tstart, tend
    - align with time intervals
    - rescale PV and EV load with bld
    """

    def __init__(self, tstart=None, tend=None, delta=0.25,
                pv_to_bld=None, ev_to_bld=None, ev_rand_td=None, rand_seed=42, fillna=False, bld_load_mean=None,
                **kwargs_load_data):
        
        self.load_bld = None
        self.load_pv = None
        self.ev_sessions = None
        self.data_tmp = None

        # Step 1: load data (in temporary form)
        self.load_data_tmp(**kwargs_load_data)

        # Step 2: remove data out of time range
        #   for bld & pv: align with time inverals
        self.align_time_range(tstart=tstart, tend=tend, delta=delta, 
            ev_rand_td=ev_rand_td, rand_seed=rand_seed)

        print(bld_load_mean)
        # Step 3 (optional): rescale PV harvest and/or EV load based on bld load
        self.rescale_load(pv_to_bld=pv_to_bld, ev_to_bld=ev_to_bld, rand_seed=rand_seed, bld_load_mean=bld_load_mean)

        # [Yi, 2023/03/19]
        # check missing values - fill them if [fillna = True]
        #   fillna only when necessary (e.g., profile extraction -> False, MPC simulation -> True)
        for key in ["bld", "pv"]:
            s = self.load_bld if key == "bld" else self.load_pv
            num_na = s.isna().sum()
            if num_na > 0:
                print("!"*10, "MISSING VALUES", "!"*10, end=" || ")
                print(f"[{key}] has [{num_na}] missing values")
                if fillna:
                    s.fillna(method="ffill", inplace=True)
                    if s.isna().sum() > 0:
                        s.fillna(method="bfill", inplace=True)
                    assert s.isna().sum() == 0
                    print("="*10, "NA filled", "="*10)
        
        if self.ev_sessions is not None:
            ev_sessions = self.ev_sessions
            ta = ev_sessions["ta"]
            td = ev_sessions["td"]

            min_duration = 5 # min
            sig = (td - ta) <= timedelta(minutes = min_duration)
            print("!"*10, "EV SHORT DURATION", "!"*10, end=" || ")
            print(f"drop {sig.sum()} sessions")
            ev_sessions.drop(index=ev_sessions.loc[sig].index, inplace=True)

    def load_data_tmp(self, **kwargs_load_data):
        """ this method needs to be OVERRIDE for each specific dataset"""
        self.data_tmp = {   # tmp: short for "temporary"
            "load_bld": None, # a pd.Series
            "load_pv": None, # a pd.Series
            "ev_sessions": None, # a pd.DataFrame, 
            # indexed on EV id, with columns ["ta", "td", "e_targ", "Pmax",], 
            # sorted on "ta". optional keys: "e_init", "td_actual"  
        }
    
    def align_time_range(self, tstart, tend, delta, ev_rand_td=None, rand_seed=42):
        
        load_bld_tmp = self.data_tmp["load_bld"]
        load_pv_tmp = self.data_tmp["load_pv"]
        if "ev_sessions" in self.data_tmp:
            ev_sessions_tmp = self.data_tmp["ev_sessions"]
            warnings.warn("No ev data loaded, EIO if pred_model is Prediction")
        else:
            ev_sessions_tmp = None

        def datetime_round(t, mode="round"):
            t0 = datetime(2019,1,1,0,0)
            dt = (t-t0).total_seconds()
            round_fcn = {
                "round": np.round, "floor": np.floor, "ceil": np.ceil
            }
            return t0 + timedelta(seconds=
                round_fcn[mode](dt/(delta*3600)) * (delta*3600))

        if tstart is None:
            tstart = datetime_round(
                max(load_bld_tmp.index.min(), load_pv_tmp.index.min()), mode="floor")
        if tend is None:
            tend = datetime_round(
                min(load_bld_tmp.index.max(), load_pv_tmp.index.max()), mode="ceil")
        
        time_range = pd.date_range(
            # [Yi, 2023/02/26] closed=None -> inclusive="both"
            start=tstart, end=tend, freq="{}H".format(delta), inclusive="both") # None: both sides
        
        def load_align_timerange(s, mode="round"):
            by = s.index.round("{}H".format(delta))
            s_aligned = s.groupby(by=by).agg(np.nanmean)
            
            # make sure that index is exactly time_range
            #   even if there are missing values (will be NA)
            df_tmp = pd.DataFrame(index=time_range, columns=[0])
            df_tmp[0] = s_aligned
            return df_tmp.loc[time_range, 0]

        self.load_bld = load_align_timerange(load_bld_tmp)
        self.load_pv = load_align_timerange(load_pv_tmp)
        
        # FIXME:
        # assert if there are NA - if so, raise warnings
        ...


        if ev_sessions_tmp is None:
            return

        ev_cols = ["ta", "td", "td_actual", "e_init", "e_targ", "Pmax"]
        ev_cols_tmp = []
        for col in ev_cols:
            if col in ev_sessions_tmp.columns:
                ev_cols_tmp.append(col)
        ev_sig = (ev_sessions_tmp["ta"]>=tstart) & (ev_sessions_tmp["td"]<=tend)
        df_ev = ev_sessions_tmp.loc[ev_sig, ev_cols_tmp].copy()

        # ! check if the branch executed?
        if "td_actual" not in ev_cols_tmp:
            ev_rand_td = (0,0) if ev_rand_td is None else ev_rand_td
            np.random.seed(rand_seed)
            td_actual = df_ev["td"].values + pd.to_timedelta(
                np.vectorize(lambda dt: timedelta(hours=dt))(
                    delta*np.random.normal(
                        loc=ev_rand_td[0], scale=ev_rand_td[1], size=len(df_ev))))
            # Note: td_actual >= ta + 15 min
            min_dt = timedelta(hours=0.25)
            df_ev["td_actual"] = np.maximum(td_actual, df_ev["ta"]+min_dt)
        if "e_init" not in ev_cols_tmp:
            df_ev["e_init"] = 0
        
        self.ev_sessions = df_ev.sort_values(by="ta", ignore_index=True)
        return

    def rescale_load(self, pv_to_bld=None, ev_to_bld=None, rand_seed=42, bld_load_mean=None):
        
        tstart = self.load_bld.index.min()
        tend = self.load_bld.index.max()

        if bld_load_mean is None:
            bld_mean = self.load_bld.mean()
        else:
            bld_mean = bld_load_mean
        
        pv_mean = self.load_pv.mean()
        if pv_to_bld is not None:
            self.load_pv = self.load_pv * pv_to_bld * (bld_mean/pv_mean)
        
        if ev_to_bld is not None:
            ev_sessions = self.ev_sessions
            if ev_sessions is None:
                warnings.warn("No EV data loaded, EIO if pred_model is Prediction.")
                return

            ev_mean = (ev_sessions["e_targ"] - ev_sessions["e_init"]).sum()/\
                ((tend-tstart).total_seconds()/3600)
            
            ratio_I = bld_mean / ev_mean * ev_to_bld
            ratio_int = int(ratio_I)
            ratio_res = ratio_I - int(ratio_I)
            # [Yi, 2023/03/22] fix a terrible bug here
            sample_I = round(ratio_res * len(ev_sessions))
            np.random.seed(rand_seed)
            sampled_idx = np.random.choice(
                ev_sessions.index, size=sample_I, replace=False,)
            to_concat = [ev_sessions] * ratio_int + [ev_sessions.loc[sampled_idx]]
            ev_sessions_rescaled = pd.concat(to_concat, axis=0, ignore_index=True)
            self.ev_sessions = ev_sessions_rescaled.sort_values("ta", ignore_index=True)
            
            #self.ev_sessions.to_csv("ev_step3_dataloader.csv", index=True)
            
            return

    # update 2022/09/03: add a "get_data" method (consider data as a private attr)
    def get_data(self):
        copy_none = lambda x: x if x is None else x.copy()
        data = {
            "load_bld": copy_none(self.load_bld),
            "load_pv": copy_none(self.load_pv),
            "ev_sessions": copy_none(self.ev_sessions)
        }
        return data

def ev_data_loader(proj="UCSD", folder="UCSD_raw_data",
                    tstart=None, tend=None, station=None, year=None,
                    delta=None, Pmax=6.6, eta=0.98,pred_model=None):
    
    # ignore dataset "Boulder" & "slrpev"
    col_rename ={
        # "Boulder": {
        #     "Station_Name": "station",
        #     "Start_Date___Time": "ta",
        #     "End_Date___Time": "td",
        #     "Energy__kWh_": "e_targ",
        #     "Total_Duration__hh_mm_ss_": "T_tot",
        #     "Charging_Time__hh_mm_ss_": "T_char"
        # },
        "UCSD": {
            "Station Name": "station",
            "Start Date": "ta",
            "End Date": "td",
            "Energy (kWh)": "e_targ",
            "Total Duration (hh:mm:ss)": "T_tot",
            "Charging Time (hh:mm:ss)": "T_char",
        },
        # "slrpev": {
        #     "siteId": "station", # "siteId"
        #     "connectTime": "ta",
        #     "cumEnergy_Wh": "e_targ",
        #     "Duration": "T_tot",
        #     "vehicle_maxChgRate_W": "Pmax"
        # }
    }

    assert proj == "UCSD"
    fn_dir = {
        "UCSD": "EV_ChargePointEV"
    }

    fn = os.path.join(data_path, folder, fn_dir[proj]+".csv")
    if proj in ["UCSD"]:
        df = pd.read_csv(fn)
    else:
        df = pd.read_csv(fn, index_col=0)
    df.rename(columns=col_rename[proj], inplace=True)
    if proj in ["Boulder", "UCSD"]:
        df["ta"] = pd.to_datetime(df["ta"]) # remove '''infer_datetime_format=True'''
        df["td"] = pd.to_datetime(df["td"]) # remove '''infer_datetime_format=True'''
        df["T_tot"] = pd.to_timedelta(df["T_tot"])
        df["T_char"] = pd.to_timedelta(df["T_char"])
        df["Pmax"] = Pmax
    # if proj == "slrpev":
    #     df["ta"] = pd.to_datetime(df["ta"]) # remove '''infer_datetime_format=True'''
    #     df["T_tot"] = pd.to_timedelta(df["T_tot"])
    #     df["td"] = df["ta"] + df["T_tot"]
    #     df["T_char"] = pd.NaT
    #     df["e_targ"] = df["e_targ"] / 1000
    #     df["Pmax"] = df["Pmax"] / 1000
    
    # combine stations for UCSD data
    def station_combo_ucsd(x):
        for s in ["GILMAN", "HOPKINS", "OSLER", "ALL"]:
            if s in x:
                return s
        return pd.NA
    
    if proj == "UCSD":
        df["station"] = df["station"].apply(station_combo_ucsd)

    # filter station
    if station is None:
        sig_loc = [True] * len(df)
    elif isinstance(station, list):
        sig_loc = df["station"].isin(station)
    else:
        sig_loc = df["station"] == station
    
    # filter time
    tstart = df.ta.min() if tstart is None else tstart
    tend = df.td.max() if tend is None else tend
    if year is None:
        sig_year = [True] * len(df)
    elif isinstance(year, list):
        sig_year = df["ta"].dt.year.isin(year)
    else: # FIXME: None?
        sig_year = df["ta"].dt.year == year
    sig_time = (df.ta >= tstart) & (df.td <= tend) & sig_year

    df = df.loc[sig_loc & sig_time]

    # round ta, td to delta (no longer need to round)
    # if delta is not None:
        # t0 = df["ta"].min().replace(hour=0, minute=0, second=0)
        # ta = round((df["ta"] - t0).dt.total_seconds()/(delta*3600)).values.astype(int)
        # td = round((df["td"] - t0).dt.total_seconds()/(delta*3600)).values.astype(int)
        # timedelta_vec = np.vectorize(lambda x: timedelta(hours=x))
        # df["ta"] = t0 + timedelta_vec(delta*ta)
        # df["td"] = t0 + timedelta_vec(delta*td)
    
    # check feasibility
    
    
    # [Lunlong: 2023/07/28] 
    # check feasibility: drop records that duration is too short or ta=td
    df=df.dropna(axis=0,how='any') # drop rows containing any nans
    infea_idx=df[(df["td"]-df["ta"]).dt.total_seconds()/3600<=0.25]["td"].index
    df=df.drop(index=infea_idx)
    
    duration = (df["td"]-df["ta"]).dt.total_seconds()/3600 
    e_targ = df["e_targ"].values
    Pmax = df["Pmax"]
    e_init = e_targ * 0
    # FIXME: raise warnings if there are infeasible sessions
    df["e_targ"] = np.minimum(e_targ, Pmax*eta * duration)

    df.sort_values(by="ta", inplace=True, ignore_index=True)
    
    # ! for index debugging
    # df.to_csv("ev_step_1_evdataloader.csv", index=True)

    return df


class UCSD_dataloader(DataLoader):
    
    def load_data_tmp(self, 
        bld=None, pv=None, ev=None, combined_fn=None,
        folder="UCSD_raw_data", **ev_load_kw):
        

        """
        two methods to load data
        1. use raw files, for which you should give the file names of
            bld, PV, EV
        2. use a combined file (xlsx), with sheet names 
            "load" (with cols "bld" & "PV"), "EV"
        """
        
        self.data_tmp = {   # tmp: short for "temporary"
            "load_bld": None, # a pd.Series
            "load_pv": None, # a pd.Series
            "ev_sessions": None, # a pd.DataFrame, 
            # indexed on EV id, with columns ["ta", "td", "e_targ", "pmax",], 
            # sorted on "ta". optional keys: "e_init", "td_actual"  
        }
        
        if combined_fn is None:
            bld_fn = os.path.join(data_path, folder, "BLD_{}.csv".format(bld))
            pv_fn = os.path.join(data_path, folder, "PV_{}.csv".format(pv))
            load_bld = pd.read_csv(bld_fn, index_col=0)
            load_pv = pd.read_csv(pv_fn, index_col=0)
            load_bld.index = pd.to_datetime(load_bld.index) # remove '''infer_datetime_format=True'''
            load_pv.index = pd.to_datetime(load_pv.index) # remove '''infer_datetime_format=True'''
            self.data_tmp["load_bld"] = pd.Series(load_bld["RealPower"], copy=True)
            self.data_tmp["load_pv"] = pd.Series(load_pv["RealPower"], copy=True)

            if ev is not None:
                df_ev = ev_data_loader(proj="UCSD", folder=folder, station=ev, **ev_load_kw)
                self.data_tmp["ev_sessions"] = pd.DataFrame(df_ev, copy=True)

        else:
            fn = os.path.join(data_path, folder, combined_fn)
            dfs = pd.read_excel(fn, sheet_name=None, index_col=0)
            
            df_load = dfs["load"]
            df_load.index = pd.to_datetime(df_load.index) # remove '''infer_datetime_format=True'''
            self.data_tmp["load_bld"] = pd.Series(df_load["bld"], copy=True)
            self.data_tmp["load_pv"] = pd.Series(df_load["PV"], copy=True)
            
            if "EV" in dfs.keys():
                df_ev = dfs["EV"]
                for key in ["ta", "td", "td_actual"]:
                    if key in df_ev.columns:
                        df_ev[key] = pd.to_datetime(df_ev[key]) # remove '''infer_datetime_format=True'''
                self.data_tmp["ev_sessions"] = pd.DataFrame(df_ev, copy=True)
                

# [Lunlong 2023/08/18] 
# fix a terrible bug:
# add class for loading static XGB predicted load
# with preprocessing the same as USCD_dataloader           
class XGB_dataloader(DataLoader):
    
    def load_data_tmp(self, 
        bld=None, pv=None, ev=None, combined_fn=None,
        folder="load_forecast\XGB", **ev_load_kw):
        

        """
        two methods to load data
        1. use raw files, for which you should give the file names of
            bld, PV
        2. use a combined file (xlsx), with sheet names 
            "load" (with cols "bld" & "PV")
        """
        # only load of bld and pv available here
        self.data_tmp = {   # tmp: short for "temporary"
            "load_bld": None, # a pd.Series
            "load_pv": None, # a pd.Series
            # "ev_sessions": None, # a pd.DataFrame, 
            # indexed on EV id, with columns ["ta", "td", "e_targ", "pmax",], 
            # sorted on "ta". optional keys: "e_init", "td_actual"  
        }
        
        if combined_fn is None:
            bld_fn = os.path.join(data_path, folder, "BLD_{}.csv".format(bld))
            pv_fn = os.path.join(data_path, folder, "PV_{}.csv".format(pv))
            load_bld = pd.read_csv(bld_fn, index_col=0)
            load_pv = pd.read_csv(pv_fn, index_col=0)
            load_bld.index = pd.to_datetime(load_bld.index) # remove '''infer_datetime_format=True'''
            load_pv.index = pd.to_datetime(load_pv.index) # remove '''infer_datetime_format=True'''
            self.data_tmp["load_bld"] = pd.Series(load_bld["RealPower"], copy=True)
            self.data_tmp["load_pv"] = pd.Series(load_pv["RealPower"], copy=True)


        else:
            fn = os.path.join(data_path, folder, combined_fn)
            dfs = pd.read_excel(fn, sheet_name=None, index_col=0)
            
            df_load = dfs["load"]
            df_load.index = pd.to_datetime(df_load.index) # remove '''infer_datetime_format=True'''
            self.data_tmp["load_bld"] = pd.Series(df_load["bld"], copy=True)
            self.data_tmp["load_pv"] = pd.Series(df_load["PV"], copy=True)


import itertools

class Exp_table():
    def __init__(self,params_dic,save_path):
        # Get all possible combinations
        combinations = list(itertools.product(*params_dic.values()))
        # Convert to DataFrame
        df = pd.DataFrame(combinations, columns=params_dic.keys())
        df.to_excel(save_path, index=True)
        self.df=df
        
    def get_table(self):
        return self.df