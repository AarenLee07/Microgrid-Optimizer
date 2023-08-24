import warnings
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from datetime import datetime, timedelta
import os, sys

import utils.utils as utils
# from data_loader import UCSD_dataloader
# from data_pool import DataPool



"""
updates:
[Yi, 2023/03/22]
- fix the bug with no EVs (either no EV in raw data, or choose [incl_ev=False])

[Yi, 2023/03/12]
- implement two stats method: [within_profile_variance] & [cross_profile_variance] in class [RepProfiles]
- change the sample rule in sampling to be [with replacement], so that it is i.i.d.
- change the weights in importance sampling - do not normalize to sum one
- change the rule "l2" to be square_rooted, and add rule "l2_squared"
    some comments on propriate importance function:
    - on one hand, you want to sample rare profiles, so their biased weights (delta_i) need to be high
    - on the other hand, you want to have the sum of likelihood ratio (pi_i) to be closer to 1
    the arg [importance_func] gives you flexibility in trying different transformations, e.g,:
        if [idst_metric = "l2"] and [importance_func = lambda x: x ** 1.5]
        the biased weights will be: (||x||_2) ** 1.5
        Note: beofre applying importance_func, the weights have been normalized so that, e.g., max{||x||_2} = 1

[Yi, 2023/03/09] implement a new class [RepProfiles]
    basically, it contains representative profiles
    method [save] & [profiles_to_params] are implemented

[Yi, 2023/03/01] add sampling methods

[Yi, 2023/03/01] add key "agg_power" for clustered / sampled results, 
    replacing cache["cluster_centers"]

[Yi, 2023/03/01] rename method: cluster_postprocess -> sample_postprocess
    it is called by both clustering and sampling
"""


class RepProfiles():
    def __init__(self, rep_profiles, extract_config=None, save_config=None, **dataset_info):
        self.data = rep_profiles["data"]
        self.prob = rep_profiles["prob"]
        self.agg_power = rep_profiles["agg_power"]

        self.save_config = save_config
        self.extract_config = extract_config
        self.extract_config.update(dataset_info)

        self.t0 = extract_config.get("tstart", datetime(2019,1,1,0,0))

        self.load_df = None
        self.ev_df = None

        self.combine_profiles() # self.load_df, self.ev_df
        self.stats = self.rep_profile_analysis()

    
    def combine_profiles(self):
        
        data = self.data
        S = len(self.data)
        load_to_concat = []
        
        have_ev = ("ev_sessions" in data[0].keys()) and (data[0]["ev_sessions"] is not None)
        if have_ev:
            ev_to_concat = []
        
        for i in range(S):
            load_df = pd.DataFrame(columns = ["load_bld", "load_pv", "group"])
            load_df["load_bld"] = data[i]["load_bld"]
            load_df["load_pv"] = data[i]["load_pv"]
            load_df["group"] = i
            if self.extract_config["alg"] == "cluster":
                load_df.index = (pd.Series(load_df.index) + timedelta(days=i)).apply(lambda x: x+self.t0)
            load_to_concat.append(load_df)

            if have_ev:
                ev_sessions = data[i]["ev_sessions"]
                ev_sessions["group"] = i
                ev_to_concat.append(ev_sessions)

        self.load_df = pd.concat(load_to_concat, axis=0)
        if have_ev:
            self.ev_df = pd.concat(ev_to_concat, axis=0)


    def rep_profile_analysis(self):
        """  TODO [@ Linfeng]: provide a standard analysis of extracted profiles  """
        
        stats = {}
        
        # within-profile-variance
        for D in [1, 2, 4, 8, 16]:
            stats.update(
                self.within_profile_variance(self.agg_power, self.prob, D))

        # cross-profile-variance    store both "weighted" & "unweighted"
        stats.update(
            self.cross_profile_variance(self.agg_power, self.prob))
        stats.update(
            self.cross_profile_variance(self.agg_power, None))

        return stats
    

    def within_profile_variance(self, agg_power, prob, D):
        
        n, l = agg_power.shape  # n: # of profiles, l: # of steps per profile
        
        # FIXME: say l=96, D=24, then: [0:25), [24:49), [48: 73) are calculated, [72, 97) is omiteed to avoid [index 96 out of range]

        X = agg_power.copy()    # X.shape = (n,l)
        Y = np.array(           # Y.shape = (n, floor(l+1/D), D+1)
            [np.vstack([X[j, i*D: (i+1)*D+1] for i in range(int((l-1)/D))])
                for j in range (n)]
        )
        diff = (Y.max(axis=-1) - Y.min(axis=-1)).mean(axis=1)   # diff.shape = (n,)

        return utils.quick_stats(diff, prob=prob, key_prefix=f"WPV_{D}")
    

    def cross_profile_variance(self, agg_power, prob):
        
        n, l = agg_power.shape  # n: # of profiles, l: # of steps per profile

        key = "CPV_unweight" if prob is None else "CPV"
        if prob is None:
            prob = np.array([1/n] * n)

        # FIXME: check if the following are also correct. there's a bug if prob=None is passed !
        mu = utils.mean_with_prob(agg_power, prob, axis=0)
        std = utils.std_with_prob(agg_power, prob, axis=0)

        return {key: std}




    def save(self, save_fn=None, save_fn_prefix=None, save_path=None):
        
        if save_path is None:
            save_path = self.save_config["save_path"]
        assert os.path.exists(save_path)
        
        ec = self.extract_config

        if save_fn is None:

            if save_fn_prefix is None:
                save_fn_prefix = self.save_config["save_fn_prefix"]
            if save_fn_prefix is None:
                save_fn_prefix = ""

            fn = "{}_ALG_{}_K_{}_EV_{}_{}_Dist_{}".format(
                save_fn_prefix, ec.get("alg"), ec.get("K"), 
                ec.get("ev_incl"), ec.get("ev_how_to"), ec.get("dist_metric")
            )

            exist_fns = os.listdir(save_path)
            i = 1
            while True:
                suffix = "{}_{:03d}".format(datetime.today().date(), i)
                save_fn = "{}-{}.xlsx".format(fn, suffix)
                if save_fn not in exist_fns:
                    break
        
        writer = pd.ExcelWriter(os.path.join(save_path, save_fn))
        self.load_df.to_excel(writer, "load", index=True)
        if self.ev_df is not None:
            self.ev_df.to_excel(writer, "EV", index=True)
        
        prob = pd.DataFrame(self.prob, columns=["prob"])
        config = pd.DataFrame.from_dict(self.extract_config, orient="index")
        stats = pd.DataFrame.from_dict(self.stats, orient = "index")
        
        prob.to_excel(writer, "prob", index=True)
        config.to_excel(writer, "config", index=True)
        stats.to_excel(writer, "stats", index=True)
        writer.save()
            
        
    def profiles_to_params(self):
        data = self.data

        S = len(data)
        S_prob = self.prob
        load_bld, load_pv, K, delta = [], [], [], []
        ev_I, ev_ta, ev_td, ev_e_init, ev_e_targ, ev_p_max = [], [], [], [], [], []

        for i in range(S):
            
            load_df = self.load_df.loc[self.load_df["group"]==i]

            load_bld.append(load_df["load_bld"].values)
            load_pv.append(load_df["load_pv"].values)

            l = data[i]["load_bld"]
            K.append(len(l))
            d = (l.index[1] - l.index[0]).total_seconds() / 3600
            if d not in [0.25, 0.5, 1, 2]:
                warnings.warn(f"ATTENTION! delta = {d} for {i}-th profile.")
            delta.append(round(d, 2))


            if self.ev_df is None:
                # FIXME: if params has no keys e.g., "ev_ta", or the value is None, it should be interpreted as no EV, and is valide
                ev_I.append(0)
                for l in [ev_ta, ev_td, ev_e_init, ev_e_targ, ev_p_max]:
                    l.append(np.array([], dtype=float))
                continue

            ev_sessions = self.ev_df.loc[self.ev_df["group"] == i]
            ev_I.append(len(ev_sessions))
            ev_e_init.append(ev_sessions["e_init"].values)
            ev_e_targ.append(ev_sessions["e_targ"].values)
            ev_p_max.append(ev_sessions["Pmax"].values)
            
            if len(ev_sessions) == 0:
                # FIXME: not an elegant way, but otherwise, ta.dtype = '<m8[ns]'
                ev_ta.append(np.array([], dtype=float))
                ev_td.append(np.array([], dtype=float))
                continue

            ta = ev_sessions["ta"]
            td = ev_sessions["td"]
            day = ta.dt.floor(freq="D")

            func_td_to_hr = lambda x: x.total_seconds()/3600 # convert timedelta to hours
            ev_ta.append((ta-day).apply(func_td_to_hr).values/d)
            ev_td.append((td-day).apply(func_td_to_hr).values/d)

        params = {
            "K": K, "delta": delta, "S": S, "S_prob": S_prob,
            "load_bld": load_bld, "load_pv": load_pv, 
            "ev_I": ev_I, "ev_ta": ev_ta, "ev_td": ev_td, 
            "ev_e_init": ev_e_init, "ev_e_targ": ev_e_targ, "ev_p_max": ev_p_max
        }

        return params



class ProfileExtraction():

    def __init__(self, 
                data, 
                tstart = None, 
                tend = None,
                rand_seed = 2023,
                save_fn_prefix = None,
                save_path = None 
                ):
        
        
        self.rand_seed = rand_seed

        self.delta = 0.25 # FIXME: it should be a variable
        
        self.data_prep(data, tstart=tstart, tend=tend)
        # self.data: dict, keys: [data_agg, ev_sessions]
        # self.tstart, self.tend will also be set (inside the method)

        self.save_config = {"save_fn_prefix": save_fn_prefix,
                            "save_path": save_path}
        self.cache = {}

        
    def rep_profile_extraction(self,
            alg = "cluster",    # "cluster" or "sample" 
            K = 20,      # number of typical profiles
            tstart = None, 
            tend = None,
            dist_metric = None,   # used for "importance sampling"
                # support: "l1", "l2", "l1_center", "l2_center", "l2_squared", "l2_center_squared"
            importance_func = None,    # used for "imporance sampling"
                # support: pass a lambda function, default use dist_metric directly, if not None 
            incl_ev = False,    # whether to have EVs in the representative profile
            ev_how_to = None,   # if include EV load, how to add it into each time step
                # support: None, "unif", "asap"
            incl_tou = False,   # whether to consider TOU in extraction, require a key "TOU" in data
            tou_how_to = None,  # if consider TOU, how to integrate it
                # support: FIXME
            rand_seed = None):


        # [Yi, 2023/03/09]
        self.tstart = max(tstart, self.tstart) if tstart is not None else self.tstart
        self.tend = min(tend, self.tend) if tend is not None else self.tend

        if rand_seed is not None:
            self.rand_seed = rand_seed

        if alg not in ["cluster", "sample"]:
            raise Exception("Now we only support [cluster] [sample] two algorithm for [alg] arg")


        df = self.data["data_agg"]
        
        # FIXME: now we assume incl_tou == False
        data_1D = df["load_bld"] - df["load_pv"]
        if ev_how_to is not None:
            data_1D += df[f"load_ev_{ev_how_to}"]   # "unif" or "asap"
        data_1D = pd.DataFrame(data_1D, columns = ["load"])

        data_1D["day"] = data_1D.index.floor(freq="D")
        data_1D["minu"] = data_1D.index - data_1D["day"]
        data_2D = data_1D.pivot(index="day", columns="minu", values="load")

        self.cache["data_2D"] = data_2D.copy()

        # [Yi, 2023/03/09] return a [RepProfiles] object 

        if alg == "cluster":
            # "data": list, each a dict: "load_bld", "load_pv", "ev_sessions"
            # "prob": weights of each profile
            res = self.cluster_profiles(data_2D, K=K, incl_ev=incl_ev, )
        elif alg == "sample":
            res = self.sample_profiles(data_2D, K=K, incl_ev=incl_ev, dist_metric=dist_metric, importance_func=importance_func)


        extract_config = {
            "alg": alg, "K": K, "tstart": self.tstart, "tend": self.tend,
            "dist_metric": dist_metric, "importance_func": importance_func, # it may be a function, need to manually write as a string
            "incl_ev": incl_ev, "ev_how_to": ev_how_to,
            "incl_tou": incl_tou, "tou_how_to": tou_how_to,
            "rand_seed": rand_seed
        }

        return RepProfiles(rep_profiles=res,
                           extract_config = extract_config,
                           save_config=self.save_config)


    def data_prep(self, data, tstart, tend):
        
        ## input check
        for k in ["load_bld", "load_pv"]:
            if k not in data.keys():
                raise Exception(f"Key {k} not found in [data]")

        load_bld = data["load_bld"]
        load_pv = data["load_pv"]

        have_ev = ("ev_sessions" in data.keys()) and (data["ev_sessions"] is not None)
        have_tou = "TOU" in data.keys() 

        tstart_bld, tend_bld = load_bld.index.min(), load_bld.index.max()        
        tstart_pv, tend_pv = load_pv.index.min(), load_pv.index.max()
        
        tstart = max(tstart, tstart_bld) if tstart is not None else tstart_bld
        tend = min(tend, tend_bld) if tend is not None else tend_bld
        self.tstart, self.tend = tstart, tend
        
        # warnings: if there seems a significant missing part of some columns
        # TODO: also need to check EV & TOU if they are provided
        thres = 30 # days
        if tstart_pv - tstart >= timedelta(days=thres):
            warnings.warn(f"The earliest available PV data is more than {thres} days later than [tstart]\n It is {tstart_pv}. You may want a check")
        if tend - tend_pv >= timedelta(days=thres):
            warnings.warn(f"The latest available PV data is more than {thres} days earlier than [tend]\n It is {tstart_pv}. You may want a check")


        df_cols = ["load_bld", "load_pv"]
        if have_ev:
            df_cols += ["load_ev_unif", "load_ev_asap"]
        if have_tou:
            df_cols += ["tou_buy", "tou_sell"]
        
        df = pd.DataFrame(columns=df_cols)
        eps = timedelta(seconds=1)  # loc of datetime closed on both sides, but wee do not want to include the first step of the next day
        df["load_bld"] = load_bld.loc[tstart:tend-eps]
        df["load_pv"] = load_pv.loc[tstart:tend-eps]


        if have_ev:
            # [Yi, 2023/03/09] drop ev_sessions whose duration is 0
            min_duration = timedelta(minutes=1)
            ev_session = data["ev_sessions"]
            sig = (ev_session["td"] - ev_session["ta"] >= min_duration)
            data["ev_sessions"] = ev_session.loc[sig].copy()

            ev_sig = (data["ev_sessions"]["ta"] >= tstart) & (data["ev_sessions"]["td"] <= tend)
            ev_sessions = data["ev_sessions"].loc[ev_sig].copy()
            df["load_ev_unif"] = self.convert_ev_load(ev_sessions, mode="unif", delta=self.delta)
            df["load_ev_asap"] = self.convert_ev_load(ev_sessions, mode="asap", delta=self.delta)

        if have_tou:
            pass # TODO

        # FIXME: how to deal with missing data ?
        # an idea: use forecast model to fill in
        # TODO: need to check the following cases work:
        # (i) 

        # TODO: need to check how many NA, and send warnings
        df.fillna(value={"load_pv": 0}, inplace=True)
        if have_ev:
            df.fillna(value={"load_ev_unif": 0, "load_ev_asap": 0}, inplace=True)


        self.data =  {
            "data_agg": df,
            "ev_sessions": ev_sessions.copy() if have_ev else None
        }


    def convert_ev_load(self, ev_sessions, mode, delta = 0.25):
        # FIXME: explore several ways to accelarate this
        # (1) calculate "asap" and "unif" together
        # (2) move everything out of loop if possible
        # (3) first cut ev_sessions by the year (time increase seems not linear)
        # (4) try longer step than 1 day
        # (5) if groupby & agg can provide better speed ? 
        
        load_ev_to_concat = []

        date = self.tstart.replace(minute=0, second=0)
        dt = timedelta(days=1)
        tend = self.tend.replace(minute=0, second=0)

        while date <= tend:
            
            date += dt
            ev_day = ev_sessions.loc[ev_sessions.ta.dt.floor(freq="D") == date]
                                    
            I = len(ev_day)
            if I == 0:
                load_ev_to_concat.append(
                    pd.Series(0, 
                    index=pd.date_range(date, date+timedelta(days=1), freq=f"{delta}H", inclusive="left"))
                )
                continue
            
            ref_matrix = np.vstack([np.arange(24/delta)]*I)

            ta = ((ev_day["ta"] - date).dt.total_seconds()/3600/delta).values
            edem = (ev_day["e_targ"] - ev_day["e_init"]).values
            pmax = ev_day["Pmax"].values
            
            if mode == "unif":
                td = ((ev_day["td"] - date).dt.total_seconds()/3600/delta).values
            elif mode == "asap":
                td = ta + edem / pmax / delta
            else:
                raise Exception("Now we only accept [mode] to be [unif] or [asap]")

            ev_pmax_frac = np.clip(ref_matrix+1 - ta.reshape(-1,1), 0, 1) +\
                            np.clip(td.reshape(-1,1) - ref_matrix, 0, 1) - 1
            duration = ev_pmax_frac.sum(axis=1)
            duration[duration == 0] = 1     # possibly td-ta = 0, in that case, avg_power can be arbitrary
            avg_power = edem / duration
            
            unif_power_tot = (ev_pmax_frac * avg_power.reshape(-1,1)).sum(axis=0)

            load_ev_to_concat.append(
                pd.Series(unif_power_tot, 
                index=pd.date_range(date, date+timedelta(days=1), freq=f"{delta}H", inclusive="left"))
            )

        load_ev = pd.concat(load_ev_to_concat)
        return load_ev

    def cluster_profiles(self, data_2D, K, incl_ev):
        
        data_2D_dropna = data_2D.dropna()
        dates = data_2D_dropna.index
        X = data_2D_dropna.values

        cluster = KMeans(n_clusters=K, random_state=self.rand_seed, n_init="auto").fit(X)
        cluster_labels = cluster.labels_
        cluster_centers = cluster.cluster_centers_
        idx, freq = np.unique(cluster_labels, return_counts=True)
        prob = freq / freq.sum()    # weights / probability for each cluster

        cluster_res = {
            "data": [],
            "prob": prob,
            "agg_power": cluster_centers}

        for i in range(K):
            d = dates[cluster_labels == i]
            cluster_res["data"].append(self.sample_postprocess(d, sample_ev=incl_ev, is_cluster=True, rand_seed=self.rand_seed+i))

        self.cache["cluster_res"] = cluster_res.copy()        
        return cluster_res
    

    def sample_postprocess(self, dates, sample_ev, is_cluster=False, rand_seed=0):
        df = self.data["data_agg"]

        df_sub = df[df.index.floor(freq="D").isin(dates)].copy()

        if is_cluster:
            df_sub["day"] = df_sub.index.floor(freq="D")
            df_sub["minu"] = df_sub.index - df_sub["day"]
            load_bld = df_sub.pivot(index="day", columns="minu", values="load_bld").mean(axis=0)
            load_pv = df_sub.pivot(index="day", columns="minu", values="load_pv").mean(axis=0)
        else:
            load_bld = df_sub["load_bld"]
            load_pv = df_sub["load_pv"]

        if sample_ev:
            ev_sessions = self.data["ev_sessions"]
            ev_sub = ev_sessions.loc[ev_sessions["ta"].dt.floor(freq="D").isin(dates)]
            
            if is_cluster:
                N_sample = round(len(ev_sub) / len(dates))     # num of sessions per day
                
                np.random.seed(rand_seed)
                ev_idx = list(ev_sub.index)
                np.random.shuffle(ev_idx)
                sampled_idx = ev_idx[:N_sample]
                ev_sample = ev_sub.loc[sampled_idx].copy()
            else:
                ev_sample = ev_sub.copy()
        else:
            ev_sample = None

        return {"load_bld": load_bld, "load_pv": load_pv, "ev_sessions": ev_sample}
    

    def sample_profiles(self, data_2D, K, incl_ev, dist_metric, importance_func):

        data_2D_dropna = data_2D.dropna()
        dates = data_2D_dropna.index
 
        if dist_metric is None: # randomly sample
            p = np.array([1]*len(dates))
        else:
            center = data_2D_dropna.mean(axis=0)
            if dist_metric == "l1":
                dist = abs(data_2D_dropna).sum(axis=1)
            elif dist_metric == "l2":
                dist = np.sqrt((data_2D_dropna ** 2).sum(axis=1))
            elif dist_metric == "l1_center":
                dist = abs(data_2D_dropna - center).sum(axis=1)
            elif dist_metric == "l2_center":
                dist = np.sqrt(((data_2D_dropna-center) ** 2).sum(axis=1))
            elif dist_metric == "l2_squared":
                dist = (data_2D_dropna ** 2).sum(axis=1)
            elif dist_metric == "l2_center_squared":
                dist = ((data_2D_dropna-center) ** 2).sum(axis=1)
            else:
                raise Exception("only support [l1] [l2] [l1_center] [l2_center] [l2_squared] [l2_center_squared] as [dist_metric]")
            p = dist / dist.max()   # always normalize the maximum weights to 1

            if importance_func is not None:
                # importance function needs to be a function, e.g., lambda x: x+1
                p = importance_func(p)
        
        p = p / p.sum()
        self.cache["sample_weight"] = p
        
        np.random.seed(self.rand_seed)
        # [Yi, 2023/03/12] with replacement: replace = False -> True
        sample_idx = np.random.choice(np.arange(len(dates)), size=K, replace=True, p=p)
        sample_dates = dates[sample_idx]
        sample_centers = data_2D_dropna.loc[sample_dates].values


        # [Yi, 2023/03/12] do not normalize weights
        # prob = (KN p)^{-1}

        prob = 1/(len(dates) * K * p[sample_idx])
        
        eps = 0.2
        if abs(prob.sum() - 1) >= eps:
            warnings.warn("The sum of weights are {:.2f}. You may want to check.".format(prob.sum()))


        sample_res = {
            "data": [],
            "prob": np.array(prob),    # [Yi, 2023/03/04] prob: pd.Series -> np.ndarray
            "agg_power": sample_centers,
            "sample_dates": sample_dates}   # [Yi, 2023/03/04] add "sample_dates" key
        
        for i in range(K):
            d = [sample_dates[i]]
            sample_res["data"].append(self.sample_postprocess(d, sample_ev=incl_ev, is_cluster=False))

        self.cache["sample_res"] = sample_res.copy()
        return sample_res

