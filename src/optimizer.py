import os, sys
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import copy
import gurobipy as gp
from gurobipy import GRB
import warnings
from battery_model import *
from predictor import Predictor_tou_CAISO, Predictor_tou_SDGE_DA
import math

"""
============================= UPDATES =============================
(write down updates on the code - the last updated one at the first)
===================================================================
# 2022/09/03, Yi
1. add arg: "mute" to [optimize_battery_size] & [get_control_sequence]
    so that users can choose to mute gurobi output.

# 2022/08/26, Yi
1. deal with EVs whose td > t+K in [Battery_optimizer.params_reg]

"""


class Battery_solution():
    def __init__(self, sol, params, battery, save_config=None):
        
        # [sol] has keys: "bat_capacity", "p_grid", "bat_p", "bat_e", "ev_p", "ev_e"
        self.sol = sol

        # [params] has keys:
        #   "S", "S_prob"
        #   "load_bld", "load_pv", "energy_price_buy", "energy_price_sell", "dc_price"
        #   "deg_model_opt", "deg_thre_opt", "deg_lambda_opt"
        self.params = params

        # This is the "actual" battery model
        self.battery = battery.copy_params()
        self.battery.set_params(bat_capacity = sol["bat_capacity"])
        
        # This is the model used in optimization
        self.battery_est = self.battery.copy_params(capacity=True)
        deg_model_opt = params["deg_model_opt"]
        self.battery_est.set_params(deg_model = deg_model_opt)
        if deg_model_opt not in ["unconscious", "throughput"]:
            self.battery_est.set_params(
                **{f"deg_{deg_model_opt}_thres": params["deg_thres_opt"],
                   f"deg_{deg_model_opt}_lambda": params["deg_lambda_opt"]}
            )

        self.save_config = save_config
        self.summary = None

        self.cycle_history = None
        self.cycle_history_est = None

        self.load_df = None
        self.ev_df = None

    def sol_summary(self, combine_df = True):
        
        if self.summary is not None and self.load_df is not None:
            return self.summary
        
        params = self.params
        sol = self.sol
        
        S = params["S"]
        
        index = ["unit", "All"] + list(range(S))
        columns = {"TCO": "$/day",    # "All" only
                 "TCO_est": "$/day",    # "All" only
                 "OPEX": "$/day", 
                 "CAPEX": "$/day",      # "All" only
                 "CAPEX_est": "$/day",      # "All" only
                #  TODO: "recover": "yr",           # "All" only    how many years the CAPEX can be recovered
                #  TODO: "recover_est": "yr",       # "All" only
                 "bat_capacity": "kWh",     # "All" only
                 "eq_bat_capacity": "hr",   # "All" only
                 "eq_cycles": "1/day",
                 "eq_cycles_est": "1/day",
                 "eq_rate": "$/kWh",        # "All" only
                 "eq_rate_est": "$/kWh",        # "All" only
                 "eq_rate_opex": "$/kWh", 
                 "eq_rate_null": "$/kWh",
                 "demand_charge": "$/day", 
                 "tou_cost": "$/day", 
                 "prob": "1", 
                 "ev_I": "1/day", 
                 "load_bld": "kWh/day",
                 "load_ev": "kWh/day", 
                 "load_tot": "kWh/day", 
                 "load_pv": "kWh/day",
                 "grid_import": "kWh/day",
                 "grid_export": "kWh/day",
                 "grid_max": "kW",
                 "import_cost": "$/day",
                 "export_revenue": "$/day",
                 }

        self.cycle_history = []
        self.cycle_history_est = []

        load_to_concat = []
        ev_to_concat = []

        df = pd.DataFrame(index=index, columns=columns.keys())
        for k in columns.keys():
            df.loc["unit", k] = columns[k]
        
        bat_capacity = sol["bat_capacity"]
        df.loc["All", "bat_capacity"] = bat_capacity
        prob = np.array(params["S_prob"])
        prob /= prob.sum()          # FIXME: normalize to sum 1 ?
        df.loc[range(S), "prob"] = prob     

        for s in range(S):
            
            K, delta = params["K"][s], params["delta"][s]
            
            load_bld = params["load_bld"][s].mean() * 24   
            load_pv = params["load_pv"][s].mean() * 24  
            load_ev = sol["ev_p"][s].sum() * (24/K)
            load_tot = load_bld + load_ev
            ev_I = params["ev_I"][s] * 24 / (K*delta)

            df.loc[s, "ev_I"] = ev_I
            df.loc[s, "load_bld"] = load_bld
            df.loc[s, "load_ev"] = load_ev
            df.loc[s, "load_tot"] = load_tot
            df.loc[s, "load_pv"] = load_pv

            p_grid = sol["p_grid"][s]
            grid_import = np.maximum(p_grid, 0)
            grid_export = np.maximum(-p_grid, 0)
            tou_import = params["energy_price_buy"][s]
            tou_export = params["energy_price_sell"][s]
            import_cost = grid_import @ tou_import * (24/K)
            export_revenue = grid_export @ tou_export * (24/K)
            tou_cost = import_cost - export_revenue

            df.loc[s, "grid_import"] = grid_import.mean() * 24
            df.loc[s, "grid_export"] = grid_export.mean() * 24
            df.loc[s, "import_cost"] = import_cost
            df.loc[s, "export_revenue"] = export_revenue
            df.loc[s, "tou_cost"] = tou_cost

            p_dc = grid_import.max()
            df.loc[s, "grid_max"] = p_dc
            demand_charge = p_dc * params["dc_price"] * 24/(K*delta)
            opex = tou_cost + demand_charge
            eq_rate_opex = opex / load_tot
            
            """
            # null strategy: the most naive strategy
            #   no battery
            #   PV first satisfy the load, then sell to the grid

            # FIXME: this section can be modularized - it is also used in [] & []
            ev_rule = params["ev_charge_rule_default"]
            ev_efficacy = params["ev_efficacy"]
            ev_p_max = params["ev_pmax_frac"][s] * params["ev_p_max"][s][:,None]    # shape = (I, K)
            
            e_init = params["ev_e_init"][s][:,None] # shape = (I, 1)
            e_targ = params["ev_e_targ"][s][:,None] # shape = (I, 1)

            if ev_rule == "asap":
                ev_e = np.hstack([e_init, e_init + ev_p_max.cumsum(axis=1) * delta * ev_efficacy])
                ev_e = np.minimum(ev_e, e_targ)
                ev_p = (ev_e[:,1:] - ev_e[:,:-1]) / (delta * ev_efficacy)  # shape = (I, K)
            if ev_rule == "unif":
                ev_e_max = ev_p_max.sum(axis=1)[:,None] * (delta * ev_efficacy)     # shape = (I, K)
                ev_p = ev_p_max * ((e_targ - e_init) / ev_e_max)
                ev_e = np.hstack([e_init, e_init + ev_p.cumsum() * delta * ev_efficacy])
            
            load_ev_null = ev_p.sum(axis=0)
            load_tot_null = load_ev_null + params["load_bld"][s]
            pv_to_load_null = np.minimum(load_tot_null, params["load_pv"][s])
            pv_to_grid_null = params["load_pv"][s] - pv_to_load_null
            grid_to_load_null = load_tot_null - pv_to_load_null
            tou_cost_null = (grid_to_load_null @ tou_import -\
                                pv_to_grid_null @ tou_export ) * (24/K)
            demand_charg_null = grid_to_load_null.max() * params["dc_price"] * 24 / (K*delta)
            df.loc[s, "eq_rate_null"] = (tou_cost_null+demand_charg_null) / load_tot
            """

            df.loc[s, "demand_charge"] = demand_charge
            df.loc[s, "OPEX"] = opex
            df.loc[s, "eq_rate_opex"] = eq_rate_opex


            bat_e = sol["bat_e"][s]

            if bat_capacity != 0:

                self.battery.update_soc(e=bat_e)
                eq_cycles = self.battery.get_states(key = "cycles_equiv") * 24/(K*delta)
                self.battery_est.update_soc(e=bat_e)
                eq_cycles_est = self.battery_est.get_states(key = "cycles_equiv") * 24/(K*delta)
                df.loc[s, "eq_cycles"] = eq_cycles
                df.loc[s, "eq_cycles_est"] = eq_cycles_est

                self.cycle_history.append(self.battery.cycle_history)
                self.cycle_history_est.append(self.battery_est.cycle_history)

                self.battery = self.battery.copy_params(capacity=True)   # start from a new battery
                self.battery_est = self.battery_est.copy_params(capacity=True)
            else:
                df.loc[s, "eq_cycles"] = 0
                df.loc[s, "eq_cycles_est"] = 0


            if combine_df == False:
                continue

            load_df = pd.DataFrame(
                index = np.arange(0, K*delta, delta) + 100 * s,
                columns = ["group", 
                "load_bld", "load_pv", "p_grid", "bat_p", "bat_e", "ev_p", "ev_I", 
                "tou_import", "tou_export", "eq_cycles_est", "eq_cycles"]
            )

            load_df["group"] = s
            load_df["load_bld"] = params["load_bld"][s]
            load_df["load_pv"] = params["load_pv"][s]
            load_df["p_grid"] = sol["p_grid"][s]
            load_df["bat_p"] = sol["bat_p"][s]
            load_df["bat_e"] = sol["bat_e"][s][1:]
            load_df["ev_p"] = sol["ev_p"][s].sum(axis=0)
            load_df["ev_I"] = (params["ev_pmax_frac"][s] > 0).sum(axis=0)[:K]
            load_df["tou_import"] = tou_import
            load_df["tou_export"] = tou_export
            if bat_capacity == 0:
                load_df["eq_cycles"] = 0
                load_df["eq_cycles_est"] = 0
            else:
                load_df["eq_cycles"] = np.array(self.cycle_history[s][self.battery.get_params("deg_model")])
                load_df["eq_cycles_est"] = np.array(self.cycle_history_est[s][params["deg_model_opt"]])

            load_to_concat.append(load_df)


            ev_df = pd.DataFrame(
                index = range(params["ev_I"][s]),
                columns = ["group", "ta", "td", "e_init", "e_targ", "Pmax"] + [f"p_{k}" for k in range(K)]
            )

            ev_df["group"] = s

            ev_df["ta"] = params["ev_ta"][s]    # FIXME: this has already been rounded
            ev_df["td"] = params["ev_td"][s]
            ev_df["e_init"] = params["ev_e_init"][s]
            ev_df["e_targ"] = params["ev_e_targ"][s]
            ev_df["Pmax"] = params["ev_p_max"][s]
            ev_df.loc[:, [f"p_{k}" for k in range(K)]] = sol["ev_p"][s]
            
            ev_to_concat.append(ev_df)


        for k in ["OPEX",
                 "eq_cycles", "eq_cycles_est",
                 "eq_rate_opex", #"eq_rate_null", 
                 "demand_charge", "tou_cost", 
                 "ev_I", "load_bld", "load_ev", "load_tot", "load_pv",
                 "grid_import", "grid_export", "import_cost", "export_revenue",]:
            
            df.loc["All",k] = df.loc[range(S), k].values @ prob
        
        df.loc["All","grid_max"] = df.loc[range(S), "grid_max"].max()
        
        df.loc["All", "eq_bat_capacity"] = bat_capacity / df.loc["All", "load_bld"] * 24    # kWh / (kWh/day) * (hr/day) -> hr


        eq_cycles = df.loc["All", "eq_cycles"]
        eq_cycles_est = df.loc["All", "eq_cycles_est"]
        bat_cycle_price = bat_capacity * params["bat_price"] / params["bat_cycle_0"]
        bat_day_price = bat_capacity * params["bat_price"] / params["bat_life_0"]

        df.loc["All", "CAPEX"] = max(eq_cycles*bat_cycle_price, bat_day_price)
        df.loc["All", "CAPEX_est"] = max(eq_cycles_est*bat_cycle_price, bat_day_price)

        df.loc["All", "TCO"] = df.loc["All", "CAPEX"] + df.loc["All", "OPEX"]
        df.loc["All", "TCO_est"] = df.loc["All", "CAPEX_est"] + df.loc["All", "OPEX"]

        df.loc["All", "eq_rate"] = df.loc["All", "TCO"] / df.loc["All", "load_tot"]
        df.loc["All", "eq_rate_est"] = df.loc["All", "TCO_est"] / df.loc["All", "load_tot"]

        self.summary = df.T
        
        if combine_df:
            self.load_df = pd.concat(load_to_concat, axis=0)
            self.ev_df = pd.concat(ev_to_concat, axis=0, ignore_index=True)
        return df
      
    def save(self, save_fn=None, save_fn_prefix=None, save_path=None):
        
        self.sol_summary()

        if save_path is None:
            save_path = self.save_config["save_path"]
        assert os.path.exists(save_path)
        
        writer = pd.ExcelWriter(os.path.join(save_path, save_fn))
        self.summary.to_excel(writer, "summary", index=True)
        
        self.load_df.to_excel(writer, "load", index=True)
        self.ev_df.to_excel(writer, "EV", index=True)
        writer.save()

class Battery_optimizer():

    def __init__(self, battery=None):
        self.battery_sample = battery   # a battery instance
        
        self.solvers = {
            "gurobi": self.solve_model_gurobi,
        }

        # cache is used to temporarily store some intermidiate results
        # so if the program fail, you can find them, which may help you to debug
        self.cache = {"params": None,
                      "solution": None,}
        
        # update 2022/09/03: store some running seetings here
        self.settings = {
            "mute_solver": False,
        }

        self.__params_sample = {
            "K": 96,
            "delta": 0.25,
            "S": 1, # number of scenarios (energy profiles). can omit, default 1
            "S_prob": None, 
            # if S != 1, params in "loads", "ev" should be list of length S
            ### loads
            "load_bld": 5 * np.sin(np.linspace(0, 2*np.pi, 96)) + 10,
            "load_pv": np.maximum(0, -np.linspace(-6,6,96)**2+20),
            "energy_price_buy": 2 + np.cos(np.linspace(0, 2*np.pi, 96)), # $/kWh
            "energy_price_sell": None, # if None, sell is not allowed. can omit
            ### demand charge
            "dc_price": 0.3, # $/day. ref: 18 $/mon
            "dc_prev_max": None, # track p_grid_max in the same billing cycle. can omit
            ### battery (here I only consider one battery)
            "bat_capacity": None, # if none, capacity is optimized
            "bat_p_max": 3, # i.e., capacity (kWh) / p_bat_max (kW) = 3 (h)
            "bat_p_min": 3, # can omit, then p_bat_min = p_bat_max
            "bat_price": 100, # $/kWh (old: 1000, ref: Tesla Powerwall)
            "bat_efficacy": 0.98, 
            "bat_life_0": 3650, # days.
            "bat_cycle_0": 5000, # cycles in lifetime (old: 10000)
            # [Yi, 2023/03/15]  add degradation model params
            "deg_model_opt": "throughput",  # valid values: "throughput", "Crate", "rainflow", "DOD"
            "deg_thres_opt": None,
            "deg_lambda_opt": None,
            # [OLD] "reg_lambda": 0, # can omit, then reg_lambda = 0
            # [OLD] "reg_term": "p_norm", # options: "p", "p_norm", "e", "e_norm", ...
            "bat_soc_0": 0.5, # B0, BT have to be fractions (SoC indeed). 0.5 if omit
            "bat_soc_K": None, # if None, default is the same as bat_soc_0
            ### EVs
            "ev_I": 20,
            "ev_ta": np.linspace(0, 48, 20), # ta, td can be floats
            "ev_td": np.linspace(36, 96, 20),
            "ev_e_init": np.array([0]*20),
            "ev_e_targ": np.array([10]*20),
            "ev_capacity": None, # can omit, default as e_targ (useful only when aloow discharge)
            "ev_p_max": 6.6,
            "ev_p_min": 0, # can omit, default as 0
            "ev_efficacy": 0.98,
            "ev_charge_rule": "optimal",
            "ev_charge_rule_default": "unif"
        }


    """ the following methods are PUBLIC methods
        i.e., you can call them directly outside    
    """

    def optimize_battery_size(self, params, strategy="optimal", mute=False, **kw):

        # FIXME: properties specified in "self.battery_sample" 
        #   has higher priority than "params" here (i.e., will cover)
        params.update(self.battery_sample.get_params())
        params = self.params_reg(params)
        self.params_sanity_check(params)
        # [Yi, 2023/03/17] Attention!
        #   in battery.params, key [deg_model] indicates the "actual" degradation model
        #   in params here, key [deg_model_opt] indicates the model used for degradation-aware optimization


        # update 2022/09/03
        self.settings["mute_solver"] = mute

        if strategy == "optimal":
            return self.optimize_size_optimal(params=params, **kw)
        elif strategy == "MSC":
            return self.optimiza_size_rule(params=params, rule="MSC", **kw)
        elif ...:
            ...
        else:
            raise Exception("Strategy \"{}\" is not implemented".format(strategy))
    
    def estimate_daily_TCO(self, params, sol):  # FIXME: OLD !
        
        S = params["S"]
        tco = 0
        cycle_history_S = []

        for s in range(S):
            K, delta = params["K"][s], params["delta"][s]
            energy_cost =  delta / ((K*delta)/24) *\
                (np.maximum(0, sol["p_grid"][0]) @ params["energy_price_buy"][s] +\
                    np.minimum(0, sol["p_grid"][0]) @ params["energy_price_sell"][s])
            demand_charge = np.max(sol["p_grid"][0]) * params["dc_price"] 
            
            if sol["bat_capacity"] != 0:
                b = self.battery_sample.copy_params()
                b.set_capacity(sol["bat_capacity"])
                b.update_soc(e=sol["bat_e"][s], delta=delta)
                capex = b.get_states()["expected_daily_cost"]
            else:
                capex = 0
            
            tco += params["S_prob"][s] * (energy_cost + demand_charge + capex)
            cycle_history_S.append(dict(b.cycle_history))

        self.cache["cycle_history"] = cycle_history_S   # FIXME: a tentative solution
        return tco

    def get_control_sequence(self, params, strategy="optimal", mute=False, **kw):
        bat_capacity = params["bat_capacity"]
        params.update(self.battery_sample.get_params())
        params["bat_capacity"] = bat_capacity
        
        params = self.params_reg(params)
        #print("params:",params)
        self.params_sanity_check(params)
        assert params["bat_capacity"] is not None
        
        # update 2022/09/03
        self.settings["mute_solver"] = mute

        if strategy == "optimal":
            # update 2022/09/03: fix bug: no return
            # update 2022/10/08: replace "optimize_size_optimal" with "optimize_control_sequence"
            #   seems a typo
            return self.optimize_control_sequence(params=params, **kw)
        elif strategy == "MSC":
            return self.rule_base_control_sequence(params=params, rule="MSC")
        # update 2023/05/27: add rule: TOU
        elif strategy == "TOU":
            return self.rule_base_control_sequence(params=params, rule="TOU")
        elif ...:
            ...
        else:
            raise Exception("Strategy \"{}\" is not implemented".format(strategy))


    """ the following methods should be treated as PRIVATE methods
        i.e., don't call them directly outside   """

    def optimize_size_optimal(self, params, language="gurobi"):
        # [Yi, 2023/03/18] regularization terms are no longer used
        #   the model will always directly return the optimal capacity

        opt_solver = self.solvers[language]
        self.params_sanity_check(params)
        sol = opt_solver(params)
        # sol["TCO"] = self.estimate_daily_TCO(params, sol)
        
        # sol has keys:
        #   TODO: list solution keys
        return Battery_solution(sol, params, self.battery_sample)
    
    # L?: Why this method is not implemented
    def optimiza_size_rule(self, params, rule="MSC"):
        
        sol = ...
        return sol

    def optimize_control_sequence(self, params, language="gurobi"):
        opt_solver = self.solvers[language]
        sol = opt_solver(params)
        # sol["TCO"] = self.estimate_daily_TCO(params, sol)
        # [OLD] sol["reg"] = tuple([params["reg_lambda"], params["reg_term"]])
        return Battery_solution(sol, params, self.battery_sample)
    
    def rule_base_control_sequence(self, params, rule="MSC"):
        
        # update 2023/05/17: add strategy "TOU"
        if rule == "MSC":
            sol = self.msc_control_sequence(params)
        elif rule == "TOU":
            sol = self.tou_control_sequence(params)
        return Battery_solution(sol, params, self.battery_sample)
    
    def msc_control_sequence(self, params):      
        self.params_sanity_check(params)
        self.cache["params"] = copy.deepcopy(params)
        
        S = params["S"]
        sol = {k: [None]*S for k in ["p_grid", "bat_p", "bat_e", "ev_p", "ev_e"]}

        bat_capacity = params["bat_capacity"]
        bat_p_max, bat_p_min = bat_capacity / params["bat_p_max"], bat_capacity / params["bat_p_min"]
        bat_efficacy = params["bat_efficacy"]
        ev_efficacy = params["ev_efficacy"]
        ev_rule = params["ev_charge_rule"]  # "asap" or "unif", "flex" & "V2G" are not valid
        assert ev_rule in ["asap", "unif"]
        
        for s in range(S):
            K = params["K"][s]
            delta = params["delta"][s]

            load_bld = params["load_bld"][s]
            ev_p_max = params["ev_pmax_frac"][s] * params["ev_p_max"][s][:,None]    # shape = (I, K)
            
            e_init = params["ev_e_init"][s][:,None] # shape = (I, 1)
            e_targ = params["ev_e_targ"][s][:,None] # shape = (I, 1)

            # [Yi, 2023/03/20]
            ev_p = ev_p_max
            ev_e = np.hstack([e_init, e_init + ev_p.cumsum(axis=1) * delta * ev_efficacy])
            """ following processing has been done in [params_reg] 
            if ev_rule == "asap":
                ev_e = np.hstack([e_init, e_init + ev_p_max.cumsum(axis=1) * delta * ev_efficacy])
                ev_e = np.minimum(ev_e, e_targ)
                ev_p = (ev_e[:,1:] - ev_e[:,:-1]) / (delta * ev_efficacy)  # shape = (I, K)
            if ev_rule == "unif":
                ev_e_max = ev_p_max.sum(axis=1)[:,None] * (delta * ev_efficacy)     # shape = (I, K)
                ev_p = ev_p_max * ((e_targ - e_init) / ev_e_max)
                ev_e = np.hstack([e_init, e_init + ev_p.cumsum() * delta * ev_efficacy])
            """

            load_ev = ev_p.sum(axis = 0)
            load_tot = load_bld + load_ev

            load_pv = params["load_pv"][s]

            # PV -> Load = min {PV, Load}
            P_pv_to_load = np.minimum(load_pv, load_tot)
            
            P_bat_to_load = np.zeros(K)
            P_pv_to_bat = np.zeros(K)
            bat_e = np.zeros(K+1)
            bat_p = np.zeros(K)
            bat_e[0] = bat_capacity * params["bat_soc_0"][s]
            for k in range(K):
                P_bat_to_load[k] = min(
                    load_tot[k] - P_pv_to_load[k], # if load_pv cant cover all load_tot, this term will be 0
                    bat_p_max,                           # then remained part will be provided by the bat
                    bat_e[k] * bat_efficacy / delta      # limited by the following two terms
                )
                P_pv_to_bat[k] = min(
                    load_pv[k] - P_pv_to_load[k],  # if load_pv cover all load_tot, this term will be 0
                    bat_p_min,                                           # then remained part will be for charging the bat
                    (bat_capacity - bat_e[k]) / (delta*bat_efficacy)     # limited by the following two terms
                )

                bat_p[k] = P_bat_to_load[k] - P_pv_to_bat[k]    # only one of the terms can be non-zero, negative represents being charged
                bat_e[k+1] = bat_e[k] + delta * (P_pv_to_bat[k] * bat_efficacy - P_bat_to_load[k] / bat_efficacy)
            
            P_grid_to_load = load_tot - P_pv_to_load - P_bat_to_load
            P_pv_to_grid = load_pv - P_pv_to_load - P_pv_to_bat

            p_grid = P_grid_to_load - P_pv_to_grid

            sol["p_grid"][s] = p_grid
            sol["bat_p"][s] = bat_p
            sol["bat_e"][s] = bat_e
            sol["ev_p"][s] = ev_p
            sol["ev_e"][s] = ev_e
            sol["bat_capacity"] = bat_capacity
        
        return sol

    # update 2023/05/27: add the method of tou
    def tou_control_sequence(self, params):
        self.params_sanity_check(params)
        self.cache["params"] = copy.deepcopy(params)
        
        S = params["S"]
        # construct empty table for control steps
        sol = {k: [None]*S for k in ["p_grid", "bat_p", "bat_e", "ev_p", "ev_e"]}

        bat_capacity = params["bat_capacity"]
        bat_p_max, bat_p_min = bat_capacity / params["bat_p_max"], bat_capacity / params["bat_p_min"]
        bat_efficacy = params["bat_efficacy"]
        ev_efficacy = params["ev_efficacy"]
        ev_rule = params["ev_charge_rule"]  # "asap" or "unif", "flex" & "V2G" are not valid
        assert ev_rule in ["asap", "unif"]

        # get the gap value of the price according to the threshold
        #price_buy=params["energy_price_buy"]

        if params["energy_price_sell"] is None:
            raise Exception("When TOU is chosen, price_sell can not be None")
        else:
            price_sell=params["energy_price_sell"]
            
        '''        
        if params["price_buy_kws"]["threshold"] is None or params["price_buy_kws"]["threshold"].size() != 2:
            threshold=[40,60]
            print('='*20+"Attention"+'='*20)
            print("The price_buy threshold is not well specified")
        else:
            threshold=params["price_buy_kws"]["threshold"]
        '''
        
        threshold=[40,60]   



        for s in range(S):
            price_buy=params["energy_price_buy"][s]
            #price_sell=np.asarray(price_sell)
            

            ########################## Same part taken from msc_control_sequence START ##########################
            K = params["K"][s]
            delta = params["delta"][s]

            load_bld = params["load_bld"][s]
            ev_p_max = params["ev_pmax_frac"][s] * params["ev_p_max"][s][:,None]    # shape = (I, K)
            
            e_init = params["ev_e_init"][s][:,None] # shape = (I, 1)
            e_targ = params["ev_e_targ"][s][:,None] # shape = (I, 1)

            # [Yi, 2023/03/20]
            ev_p = ev_p_max
            ev_e = np.hstack([e_init, e_init + ev_p.cumsum(axis=1) * delta * ev_efficacy])
            """ following processing has been done in [params_reg] 
            if ev_rule == "asap":
                ev_e = np.hstack([e_init, e_init + ev_p_max.cumsum(axis=1) * delta * ev_efficacy])
                ev_e = np.minimum(ev_e, e_targ)
                ev_p = (ev_e[:,1:] - ev_e[:,:-1]) / (delta * ev_efficacy)  # shape = (I, K)
            if ev_rule == "unif":
                ev_e_max = ev_p_max.sum(axis=1)[:,None] * (delta * ev_efficacy)     # shape = (I, K)
                ev_p = ev_p_max * ((e_targ - e_init) / ev_e_max)
                ev_e = np.hstack([e_init, e_init + ev_p.cumsum() * delta * ev_efficacy])
            """

            load_ev = ev_p.sum(axis = 0)
            load_tot = load_bld + load_ev
            load_pv = params["load_pv"][s]

            # PV -> Load = min {PV, Load}
            P_pv_to_load = np.minimum(load_pv, load_tot)
            
            # init table for saving the control steps
            P_bat_to_load = np.zeros(K)
            P_pv_to_bat = np.zeros(K)

            P_grid_to_bat=np.zeros(K) #added 
            P_pv_to_grid=np.zeros(K) #added
            P_grid_to_load=np.zeros(K)

            bat_e = np.zeros(K+1)
            bat_p = np.zeros(K)
            ### init the state of the battery
            bat_e[0] = bat_capacity * params["bat_soc_0"][s]

            ########################## Same part taken from msc_control_sequence END ##########################
            #FIXME: L? how to choose the method of getting the sell_price?
            def valley_period():
                if load_pv[k]>load_tot[k]: # load_pv is still surplus for battery
                    if (bat_capacity - bat_e[k])>0:
                        P_pv_to_bat[k] = min(
                            load_pv[k] - P_pv_to_load[k],
                            bat_p_max,
                            (bat_capacity - bat_e[k]) / (delta*bat_efficacy)
                        )
                        P_pv_to_grid[k]=max(
                            #surplus pv after load_tot and max_bat_charge
                            load_pv[k]-load_tot[k]-min(bat_p_max,(bat_capacity - bat_e[k]) / (delta*bat_efficacy)),
                            0
                        )
                        P_grid_to_bat[k]=max(
                            #max_bat_charge-surplus_pv_after_tot
                            bat_p_max,
                            min(bat_p_max,(bat_capacity - bat_e[k]) / (delta*bat_efficacy))-(load_pv[k]-load_tot[k]), 
                            0
                        )
                    else:
                        P_pv_to_grid[k]=max(
                            load_pv[k]-load_tot[k],
                            0
                        )
                else:
                    P_grid_to_load[k]=max(
                        #remaining part to be imported from grid
                        load_tot[k]-load_pv[k], 0
                    )
                    P_grid_to_bat[k]=max(
                        #max_bat_charge
                        min(bat_p_max,(bat_capacity - bat_e[k]) / (delta*bat_efficacy)),0
                    )

            def flat_period():
                if load_pv[k]>load_tot[k]: # load_pv is still surplus for battery
                    if (bat_capacity - bat_e[k])>0:
                        P_pv_to_bat[k] = min(
                            load_pv[k] - P_pv_to_load[k],
                            bat_p_max,
                            (bat_capacity - bat_e[k]) / (delta*bat_efficacy)
                        )
                        P_pv_to_grid[k]=max(
                            #surplus pv after load_tot and max_bat_charge
                            load_pv[k]-load_tot[k]-min(bat_p_max,(bat_capacity - bat_e[k]) / (delta*bat_efficacy)),
                            0
                        )
                    else:
                        P_pv_to_grid[k]=max(
                            load_pv[k]-load_tot[k],
                            0
                        )
                else:
                    P_grid_to_load[k]=max(
                        #remaining part to be imported from grid
                        load_tot[k]-load_pv[k], 0
                    )    

            def peak_period():
                if load_pv[k]>load_tot[k]: # load_pv is still surplus for battery
                    if (bat_capacity - bat_e[k])>0:
                        P_pv_to_bat[k] = min(
                            load_pv[k] - P_pv_to_load[k],
                            bat_p_max,
                            (bat_capacity - bat_e[k]) / (delta*bat_efficacy)
                        )
                        P_pv_to_grid[k]=max(
                            #surplus pv after load_tot and max_bat_charge
                            load_pv[k]-load_tot[k]-min(bat_p_max,(bat_capacity - bat_e[k]) / (delta*bat_efficacy)),
                            0
                        )
                    else:
                        P_pv_to_grid[k]=max(
                            load_pv[k]-load_tot[k],
                            0
                        )
                else:
                    # FIXME: do we need to set a lower bound for discharing of battery?
                    if bat_e[k]>0:
                        P_bat_to_load[k]=min(
                            load_tot[k]-load_pv[k],
                            bat_p_max,
                            bat_e[k] * bat_efficacy / delta     
                        )
                        P_grid_to_load[k]=max(
                            #remaining part to be imported from grid
                            load_tot[k]-load_pv[k]-min(bat_p_max,bat_e[k] * bat_efficacy / delta), 0
                        ) 
                    else:
                        P_grid_to_load[k]=max(
                            load_tot[k]-load_pv[k],0
                        )

            for k in range(K):
                valley_flat=np.percentile(price_buy, threshold[0])
                flat_peak=np.percentile(price_buy, threshold[1])
                '''                
                print("="*15+" Debugging output "+"="*15)
                print("price_buy[k]:{}".format(price_buy))
                print("type price_buy[k]: {}".format(type(price_buy)))
                print("valley_flat:{}".format(valley_flat))
                print("type valley_flat: {}".format(type(valley_flat)))
                print("flat_peak:{}".format(flat_peak))
                print("="*15+" Debugging output "+"="*15)'''
                
                current_price_buy=price_buy[0]
                #print("current price buy :{}, type:{}".format(current_price_buy,type(current_price_buy)))
                if current_price_buy<valley_flat:
                    valley_period()
                elif current_price_buy>=valley_flat and current_price_buy<=flat_peak:
                    flat_period()
                else:
                    peak_period()
                bat_p[k] = P_bat_to_load[k] - P_pv_to_bat[k] + P_grid_to_bat[k]   # only one of the terms can be non-zero, negative represents being charged
                bat_e[k+1] = bat_e[k] + delta * (P_pv_to_bat[k] * bat_efficacy - P_bat_to_load[k] / bat_efficacy+P_grid_to_bat[k]*bat_efficacy)
          
            #P_grid_to_load = load_tot - P_pv_to_load - P_bat_to_load
            #P_pv_to_grid = load_pv - P_pv_to_load - P_pv_to_bat          
            p_grid=P_grid_to_load+P_grid_to_bat-P_pv_to_grid

            sol["p_grid"][s] = p_grid
            sol["bat_p"][s] = bat_p
            sol["bat_e"][s] = bat_e
            sol["ev_p"][s] = ev_p
            sol["ev_e"][s] = ev_e
            sol["bat_capacity"] = bat_capacity            
        
        return sol

    def solve_model_gurobi(self, params):
        
        #print("params passed to gurobi:",params)
        
        self.params_sanity_check(params)
        self.cache["params"] = copy.deepcopy(params)

        # update 2022/09/03: allow user to mute gurobi output
        with gp.Env(empty=True) as env:
            if self.settings["mute_solver"] == True:
                env.setParam('OutputFlag', 0)
            env.start()
            # with gp.Model(env=env) as m:
            m = gp.Model("battery_sizing_model", env=env)

        if params["bat_capacity"] is None:
            # then, battery capacity is to be optimized
            bat_capacity = m.addVar(name="bat_capacity")
        else:
            bat_capacity = params["bat_capacity"]

        S = params["S"]
        opex_S = [0] * S  # opex of sub-scenario s (normalized to one day)

        deg_model = params["deg_model_opt"]
        if deg_model not in ["unconscious", "throughput"]:
            deg_thres, deg_lambda = params["deg_thres_opt"], params["deg_lambda_opt"]
        eq_cycles_S = [0] * S
        
        # [Lunlong, 2023/08/01] add new methods for dc_formulation
        assert "dc_formulation" in params.keys()
        assert "penalty_coef" in params.keys()
        dc_formulation=params["dc_formulation"]
        penalty_coef=float(params["penalty_coef"])

        for s in range(S):
            
            K = params["K"][s]
            delta = params["delta"][s]
            
            """ grid power """
            # [Yi, 2023/03/19] add p_grid_max: tranformer capacity constraint
            # L.Notes: here the p_grid_max is culculated in params_reg()
            p_grid_pos = m.addVars(K, ub=params["p_grid_max"]) # pos means import/buy
            p_grid_neg = m.addVars(K, ub=params["p_grid_max"]) # pos means export/sell
            # only "name" those that you wish to find them again out the loop
            p_grid = m.addVars(K, lb=-float('inf'), name="p_grid_{}".format(s))
            p_grid_max = m.addVar(name="p_grid_max_{}".format(s))
            
            # [Constr] p_grid = [p_grid]^+ - [p_grid]^-
            m.addConstrs((p_grid[k] == p_grid_pos[k]-p_grid_neg[k] for k in range(K)))
            # [Constr] p_grid_max >= p_grid[k]
            m.addConstrs((p_grid_max >= p_grid[k] for k in range(K)))
            # L.Notes: ? How was params["dc_prev_max"] initiated?
            #print("params[\"dc_prev_max\"]:",params["dc_prev_max"])
            

            # [obj] energy cost
            energy_cost = m.addVar(lb=-float('inf'))
            m.addConstr((energy_cost == delta *\
                sum(p_grid_pos[k] * params["energy_price_buy"][s][k] -\
                    p_grid_neg[k] * params["energy_price_sell"][s][k] for k in range(K))))
            
            
            # [obj] demand charge
            demand_charge = m.addVar()
            dc_price = params["dc_price"] * (K*delta/24) # it will later be normalized back to one day
            #print(dc_price)
            # [Lunlong, 2023/08/01] add new methods for dc_formulation
            if dc_formulation=="moving":
                m.addConstr((p_grid_max >= params["dc_prev_max"][s]))
                # L.Notes: the dc_price here is taken from xlsx containing params
                m.addConstr((demand_charge == p_grid_max * dc_price))
            elif dc_formulation=="add_on":
                #increment=m.addVar()
                #m.addConstr((increment >= 0))
                #optimizing_window=None
                #print("add_on dc called.")
                p_max_addon = m.addVar()
                m.addConstr((p_max_addon >= 0))
                m.addConstr((p_max_addon==p_grid_max - params["dc_prev_max"][s]))
                #print("add on :",p_max_addon)
                #dc_price = params["dc_price"] * (K*delta/24)
                m.addConstr((demand_charge == p_max_addon * dc_price))
            # [Lunlong 2023/08/04 new dc formulation ] 
            elif dc_formulation=="step":
                # scale dc price for each step
                dc_price_step=dc_price/K
                m.addConstr((p_grid_max >= params["dc_prev_max"][s]))
                # add variables of dc_charge for each step
                dc_step = m.addVars(K, name="dc_step_{}".format(s))
                dc_step_extra = m.addVars(K, name="dc_step_extra_{}".format(s)) 
                # calculate the sum of dc_charge
                m.addConstrs((dc_step[k] == p_grid_pos[k]*dc_price_step for k in range(K))) #+p_grid_neg[k]*dc_price_step
                m.addConstrs((dc_step_extra[k] == (p_grid_max-p_grid_pos[k])*dc_price_step for k in range(K)))
                m.addConstr((demand_charge == sum(dc_step[k]+dc_step_extra[k] for k in range(K))))
                
            """ battery """
            p_bat_pos = m.addVars(K) # pos means discharging
            p_bat_neg = m.addVars(K) # neg means charging
            p_bat = m.addVars(K, lb=-float('inf'), name="bat_p_{}".format(s))
            e_bat = m.addVars(K+1, name="bat_e_{}".format(s))
            
            # [Constr] p_bat = [p_bat]^+ - [p_bat]^-
            m.addConstrs((p_bat[k] == p_bat_pos[k] - p_bat_neg[k] for k in range(K)))
            # [Constr] inital charge & target charge
            m.addConstr((e_bat[0] == params["bat_soc_0"][s] * bat_capacity))
            m.addConstr((e_bat[K] >= params["bat_soc_K"][s] * bat_capacity))
            # [Constr] battery charge update
            bat_efficacy = params["bat_efficacy"]
            m.addConstrs((e_bat[k+1] == e_bat[k] - delta *\
                (p_bat_pos[k]/bat_efficacy - p_bat_neg[k]*bat_efficacy) for k in range(K)))
            # [Constr] battery charging power is limited
            m.addConstrs((p_bat_pos[k] <= bat_capacity / params["bat_p_max"] for k in range(K)))
            m.addConstrs((p_bat_neg[k] <= bat_capacity / params["bat_p_min"] for k in range(K)))
            # [Constr] battery charge is limited (e_bat >=0 has been enforced by default)
            m.addConstrs((e_bat[k] <= bat_capacity for k in range(K+1)))
            # [Yi, 2023/02/10] a redundant constr, 
            #   (to some extend) avoid both p_bat_pos & p_bat_neg are positive
            #   in theory, it should never happen, but it happens (maybe the suboptimal is very tiny)
            m.addConstrs((p_bat_pos[k] + p_bat_neg[k] <=
                    bat_capacity / min(params["bat_p_max"], params["bat_p_min"]) for k in range(K)))

            '''            
            """ penalty """
            penalty=m.addVar()
            if float(penalty_coef) > 0:
                params_temp=params.copy() # get a copy of original params 
                params_temp["ev_charge_rule"]="unif" # modify for MSC
                ref=self.rule_base_control_sequence(params_temp,"MSC").sol_summary() # get the solution summary of MSC
                ref=ref.T
                msc_tou=ref.loc["tou_cost","All"] 
                msc_p_max=ref.loc["grid_max","All"]
                # extra demand charge
                extra_dc=m.addVar(lb=-float('inf')) 
                m.addConstr((extra_dc==(p_grid_max-msc_p_max)*dc_price))
                # tou_cost savings
                saved_tou=m.addVar(lb=-float('inf'))
                m.addConstr((saved_tou==(msc_tou-energy_cost)))
                # penalty
                penalty_base=m.addVar(lb=-float('inf'))
                m.addConstr((penalty_base==(extra_dc-saved_tou)))
                m.addConstr((penalty==(penalty_coef*penalty_base)))
            else:
                m.addConstr((penalty==0))
            '''

            # [Obj] battery operation cost
            
            eq_cycles = m.addVar(name="eq_cycles_{}".format(s))
            # [Yi, 2023/03/19] add a "deg_model": "unconscious", i.e.,
            #   do not consider cycling degradation in either sizing or control
            if params["bat_capacity"] ==0 or deg_model == "unconscious":
                m.addConstr((eq_cycles==0))
            elif deg_model == "throughput":
                self.gurobi_eq_cycles_throughput(m, eq_cycles, p_bat_pos, p_bat_neg, delta, K)
            elif deg_model == "Crate":
                self.gurobi_eq_cycles_Crate(m, eq_cycles, p_bat_pos, p_bat_neg, 
                               deg_thres, deg_lambda, delta, K, bat_capacity, params["bat_p_max"])
            elif deg_model == "rainflow":
                self.gurobi_eq_cycles_rainflow(m, eq_cycles, p_bat_pos, p_bat_neg, deg_thres, deg_lambda, 
                                               delta, K, bat_capacity, params["bat_soc_0"][s], bat_efficacy)
            elif deg_model == "DOD":
                big_M = 100 * params["load_bld"][s].sum() * delta   # a very large constant
                self.gurobi_eq_cycles_DOD(m, eq_cycles, p_bat_pos, p_bat_neg, deg_thres, deg_lambda, 
                                          delta, K, bat_capacity, params["bat_soc_0"][s], bat_efficacy, big_M)
            
            eq_cycles_S[s] = eq_cycles

            """ THIS IS OLD VERSION
            reg_lambda = params["reg_lambda"]
            reg_term = params["reg_term"]
            bat_opex = m.addVar()
            if reg_lambda != 0:
                if reg_term == "p":
                    m.addConstr((bat_opex == reg_lambda * delta *\
                        sum(p_bat_pos[k] + p_bat_neg[k] for k in range(K))))
                elif reg_term == "p_norm":
                    p_bat_pos_frac = m.addVars(K)
                    p_bat_neg_frac = m.addVars(K)
                    m.addConstrs((p_bat_pos_frac[k] * bat_capacity / params["bat_p_max"] ==
                        p_bat_pos[k] for k in range(K)))
                    m.addConstrs((p_bat_neg_frac[k] * bat_capacity / params["bat_p_min"] ==
                        p_bat_neg[k] for k in range(K)))
                    m.addConstr((bat_opex == reg_lambda * delta *\
                        sum(p_bat_pos_frac[k] + p_bat_neg_frac[k] for k in range(K))))    
                elif reg_term == "e":
                    m.addConstr((bat_opex == reg_lambda *\
                        sum(bat_capacity - e_bat[k] for k in range(K+1))))
                elif reg_term == "e_norm":
                    soc_bat = m.addVars(K+1)
                    m.addConstrs((soc_bat[k] * bat_capacity == e_bat[k] for k in range(K+1)))
                    m.addConstr((bat_opex == reg_lambda *\
                        sum(1-soc_bat[k] for k in range(K+1))))
                else:
                    raise Exception("Rule for reg term \"{}\" is not specified".format(reg_term))
                # FIXME: "p_norm", "e_norm" includes multiplication between variables - non-convex
                if reg_term in ["p_norm", "e_norm"] and params["bat_capacity"] is None:
                    m.params.NonConvex = 2
            else:
                m.addConstr((bat_opex == 0))
            """



            """ EV """
            I = params["ev_I"][s]
            pmax_frac = params["ev_pmax_frac"][s]
            ta, td = params["ev_ta"][s], params["ev_td"][s]
            e_init, e_targ = params["ev_e_init"][s], params["ev_e_targ"][s]

            p_ev_pos = m.addVars(I, K) # pos means charging
            p_ev_neg = m.addVars(I, K) # neg means discharging
            p_ev = m.addVars(I, K, lb=-float('inf'), name="ev_p_{}".format(s))
            e_ev = m.addVars(I, K+1, name="ev_e_{}".format(s))

            # [Constr] p_ev = [p_ev]^+ - [p_ev]^-
            m.addConstrs((p_ev[i,k] == p_ev_pos[i,k] - p_ev_neg[i,k] 
                for i in range(I) for k in range(K)))
            # [Constr] inital charge & target charge
            m.addConstrs((e_ev[i,ta[i]] == e_init[i] for i in range(I)))
            m.addConstrs((e_ev[i,td[i]] == e_targ[i] for i in range(I)))
            # [Constr] EV charge update
            ev_efficacy = params["ev_efficacy"]
            m.addConstrs((e_ev[i, k+1] == e_ev[i, k] + delta *\
                (p_ev_pos[i,k]*ev_efficacy - p_ev_neg[i,k]/ev_efficacy)
                for i in range(I) for k in range(K)))
            # [Constr] EV charging power is limited
            m.addConstrs((p_ev_pos[i,k] <= params["ev_p_max"][s][i] * pmax_frac[i,k]
                for i in range(I) for k in range(K)))
            m.addConstrs((p_ev_neg[i,k] <= params["ev_p_min"][s][i] * pmax_frac[i,k]
                for i in range(I) for k in range(K)))
            # [Constr] EV charge is limited (e_ev >=0 has been enforced by default)
            m.addConstrs((e_ev[i,k] <= params["ev_capacity"][s][i] for i in range(I) for k in range(K+1)))


            """ energy balance & cost """
            load_bld = params["load_bld"][s]
            #print("load_bld: ",load_bld)
            load_pv = params["load_pv"][s]
            #print("load_pv: ",load_pv)

            m.addConstrs((load_pv[k] + p_grid[k] + p_bat[k] >=
                load_bld[k] + sum(p_ev[i,k] for i in range(I)) for k in range(K)))
            
            # opex is normalized to one day, regardless how K actually is
            # [Yi, 2023/03/15] the normalization is moved outside
            opex_s = m.addVar(lb=-float('inf'), name="opex_{}".format(s))
            m.addConstr((opex_s == demand_charge + energy_cost))
            opex_S[s] = opex_s


        """ set objective"""
        capex = m.addVar()
        opex = m.addVar(lb=-float('inf'))

        # [obj] CAPEX: cost of battery normalized to one day
        #       CAPEX >= calendar cost, CAPEX >= cycle cost
        cycle_price = 0.5 * params["bat_price"] / params["bat_cycle_0"] # unit: $ per cycle per kWh
        day_price = params["bat_price"] / params["bat_life_0"] # unit: $ per day per kWh
        day_inv_S = [24 / (params["K"][s]*params["delta"][s]) for s in range(S)]    # normalize to 1 day
        m.addConstr((capex >= bat_capacity * day_price))
        m.addConstr((capex >= cycle_price * sum(day_inv_S[s] * eq_cycles_S[s] * params["S_prob"][s] for s in range(S))))
        
        # [obj] OPEX: normalized to one day (probs should sum up to be 1)  
        m.addConstr((opex == sum(day_inv_S[s] * opex_S[s] * params["S_prob"][s] for s in range(S))))

        # set objective
        m.setObjective(capex + opex, sense=GRB.MINIMIZE) # + penalty


        """ solve model """
        m.optimize()


        """ get solution """
        solution = dict()

        # bat_capacity
        if params["bat_capacity"] is not None:
            solution["bat_capacity"] = params["bat_capacity"]
        else:
            solution["bat_capacity"] = m.getVarByName("bat_capacity").x

        # p_grid, bat_p, bat_e, ev_p, ev_p
        solution["p_grid"] = {s: np.zeros(params["K"][s]) for s in range(S)}
        solution["bat_p"] = {s: np.zeros(params["K"][s]) for s in range(S)}
        solution["bat_e"] = {s: np.zeros(params["K"][s]+1) for s in range(S)}
        solution["ev_p"] = {s: np.zeros((params["ev_I"][s], params["K"][s])) for s in range(S)}
        solution["ev_e"] = {s: np.zeros((params["ev_I"][s], params["K"][s]+1)) for s in range(S)}
        solution["eq_cycles"] = np.zeros(S)
        
        for s in range(S):
            K = params["K"][s]
            I = params["ev_I"][s]
            for k in range(K+1):
                if k < K:
                    solution["p_grid"][s][k] = m.getVarByName("p_grid_{}[{}]".format(s,k)).x
                    solution["bat_p"][s][k] = m.getVarByName("bat_p_{}[{}]".format(s,k)).x
                solution["bat_e"][s][k] = m.getVarByName("bat_e_{}[{}]".format(s,k)).x
                for i in range(I):
                    if k < K:
                        solution["ev_p"][s][i,k] = m.getVarByName("ev_p_{}[{},{}]".format(s,i,k)).x
                    solution["ev_e"][s][i,k] = m.getVarByName("ev_e_{}[{},{}]".format(s,i,k)).x
            
            # [Attention]: this [eq_cycles] is not precise, 
            #   since usually cycles do not affect cost dominately (calendar cost weighs more) 
            #   thus it will not always minimize the cycle numbers
            # the precise cycle numbers is calculated using the battery object
            if solution["bat_capacity"] == 0:
                solution["eq_cycles"][s] = 0
            else:
                eq_cycles = m.getVarByName("eq_cycles_{}".format(s)).x
                solution["eq_cycles"][s] = eq_cycles / (2*solution["bat_capacity"])
        
        #print("current sol: \n sol[\"bat_e\"][s][K]: ", solution["bat_e"][S-1][K])
        #print("params[\"bat_soc_K\"][s] ", params["bat_soc_K"][S-1]*bat_capacity)
        self.cache["solution"] = copy.deepcopy(solution)
        self.solution_sanity_check(params, solution)
        #print(solution)
        return solution

    def gurobi_eq_cycles_throughput(self, m, eq_cycles, p_bat_pos, p_bat_neg, delta, K):
        m.addConstr((eq_cycles == delta * sum(p_bat_pos[k]+p_bat_neg[k] for k in range(K))))
    
    def gurobi_eq_cycles_Crate(self, m, eq_cycles, p_bat_pos, p_bat_neg, 
                               thres, lams, delta, K, bat_capacity, bat_p_max):
        J = len(thres)
        p_bat_pos_js = m.addVars(K, J)
        p_bat_neg_js = m.addVars(K, J)
        
        # [Constr] sum(p_bat_js) = p_bat
        m.addConstrs((p_bat_pos[k] == sum(p_bat_pos_js[k,j] for j in range(J)) for k in range(K)))
        m.addConstrs((p_bat_neg[k] == sum(p_bat_neg_js[k,j] for j in range(J)) for k in range(K)))

        # [Constr] p_bat_js[j] <= thres[j] * p_max
        m.addConstrs((p_bat_pos_js[k, j] <= thres[j] * bat_capacity / bat_p_max for j in range(J) for k in range(K)))
        m.addConstrs((p_bat_neg_js[k, j] <= thres[j] * bat_capacity / bat_p_max for j in range(J) for k in range(K)))

        # [Constr] cycle count
        m.addConstr((eq_cycles == delta * sum((p_bat_pos_js[k,j]+p_bat_neg_js[k,j])*lams[j] 
                                              for j in range(J) for k in range(K))))
    
    def gurobi_eq_cycles_rainflow(self, m, eq_cycles, p_bat_pos, p_bat_neg, 
                               thres, lams, delta, K, bat_capacity, bat_soc_0, bat_efficacy):
        J = len(thres)
        p_bat_pos_js = m.addVars(K, J)
        p_bat_neg_js = m.addVars(K, J)
        e_bat_js = m.addVars(K+1, J)
        
        # [Constr] initialize e_bat_js
        m.addConstrs((e_bat_js[0,j] == thres[j] * bat_capacity * bat_soc_0 for j in range(J)))

        # [Constr] sum(p_bat_js) = p_bat
        m.addConstrs((p_bat_pos[k] == sum(p_bat_pos_js[k,j] for j in range(J)) for k in range(K)))
        m.addConstrs((p_bat_neg[k] == sum(p_bat_neg_js[k,j] for j in range(J)) for k in range(K)))

        # [Constr] e_bat_js[j] <= thres[j] * b_capacity
        m.addConstrs((e_bat_js[k, j] <= thres[j] * bat_capacity for j in range(J) for k in range(K+1)))
        
        # [Constr] e_bat_js[k+1,j] = e_bat_js[k,j] - p_bat_pos_js[j]/eta + p_bat_neg_js[j]*eta
        m.addConstrs((e_bat_js[k+1,j] == e_bat_js[k,j] + delta *\
                     (p_bat_neg_js[k,j]*bat_efficacy - p_bat_pos_js[k,j]/bat_efficacy)
                      for j in range(J) for k in range(K))) 

        # [Constr] cycle count
        m.addConstr((eq_cycles == delta * sum((p_bat_pos_js[k,j]+p_bat_neg_js[k,j])*lams[j] 
                                              for j in range(J) for k in range(K))))
    
    def gurobi_eq_cycles_DOD(self, m, eq_cycles, p_bat_pos, p_bat_neg, 
                               thres, lams, delta, K, bat_capacity, bat_soc_0, bat_efficacy, M):
        J = len(thres)
        p_bat_pos_js = m.addVars(K, J)
        p_bat_neg_js = m.addVars(K, J)
        e_bat_js = m.addVars(K+1, J)
        w = m.addVars(K+1, J, vtype=GRB.BINARY) # binary variable
        z1 = m.addVars(K+1, J)   # auxiliary variable to linearize w[k,j] * E_j
        z0 = m.addVars(K+1, J)   # auxiliary variable to linearize w[k,j-1] * E_j
        
        # [Constr] linearize w[k,j] * bat_capacity using big-M method
        # z <= w * M, z <= thres*b, z >= 0, z >= thres*b - (1-x) * M
        m.addConstrs((z1[k,j] <= w[k,j] * M for j in range(J) for k in range(K+1)))
        m.addConstrs((z1[k,j] <= thres[j] * bat_capacity for j in range(J) for k in range(K+1)))
        m.addConstrs((z1[k,j] >= thres[j] * bat_capacity - (1-w[k,j]) * M for j in range(J) for k in range(K+1)))
        
        m.addConstrs((z0[k,j] <= w[k,j-1] * M for j in range(1,J) for k in range(K+1)))
        m.addConstrs((z0[k,j] <= thres[j] * bat_capacity for j in range(J) for k in range(K+1)))
        m.addConstrs((z0[k,j] >= thres[j] * bat_capacity - (1-w[k,j-1]) * M for j in range(1,J) for k in range(K+1)))
        m.addConstrs((z0[k,0] == thres[0] * bat_capacity for k in range(K)))    # w[k,-1] = 1

        # [Constr] z1 <= e_bat_js <= z0
        m.addConstrs((e_bat_js[k,j] >= z1[k,j] for j in range(J) for k in range(K+1)))
        m.addConstrs((e_bat_js[k,j] <= z0[k,j] for j in range(J) for k in range(K+1)))

        # [Constr] initialize e_bat_js
        m.addConstr((sum(e_bat_js[0,j] for j in range(J)) == bat_capacity * bat_soc_0))

        # [Constr] sum(p_bat_js) = p_bat
        m.addConstrs((p_bat_pos[k] == sum(p_bat_pos_js[k,j] for j in range(J)) for k in range(K)))
        m.addConstrs((p_bat_neg[k] == sum(p_bat_neg_js[k,j] for j in range(J)) for k in range(K)))
        
        # [Constr] e_bat_js[k+1,j] = e_bat_js[k,j] - p_bat_pos_js[j]/eta + p_bat_neg_js[j]*eta
        m.addConstrs((e_bat_js[k+1,j] == e_bat_js[k,j] + delta *\
                     (p_bat_neg_js[k,j]*bat_efficacy - p_bat_pos_js[k,j]/bat_efficacy)
                      for j in range(J) for k in range(K))) 

        # [Constr] cycle count
        m.addConstr((eq_cycles == delta * sum((p_bat_pos_js[k,j]+p_bat_neg_js[k,j])*lams[j] 
                                              for j in range(J) for k in range(K))))

    def params_reg(self, params):
        # [Yi, 2023/03/15] staticmethod -> method (i.e., can call self.attr/methods)

        params = copy.deepcopy(params)
        
        def check_none(key):
            return (key not in params) or (params[key] is None)
        
        def fill_none(key, fill):
            return fill if check_none(key) else params[key]
        
        def len_S_list(value, copy=False):
            if isinstance(value, str) and (value in params.keys()):
                value = params[value]
            if not isinstance(value, list):
                if copy == True:
                    return [value] * S
                if S == 1:
                    return [value]
                raise Exception("{} length not equal to S={}".format(key, S))
            return value

        S = fill_none("S", fill=1)
        params["S"] = S
        params["S_prob"] = fill_none("S_prob", fill=[1/S]*S)
        
        for key in ["K", "delta", "energy_price_buy"]:
            params[key] = len_S_list(key, copy=True)
        for key in ["load_bld", "load_pv", 
                    "ev_I", "ev_ta", "ev_td", "ev_e_init", "ev_e_targ"]:
            params[key] = len_S_list(key, copy=False)
        
        key = "energy_price_sell"
        value = fill_none(key, 0)
        if isinstance(value, float) or isinstance(value, int):
            # 0726 LunLong:
            # add type checking to deal with the error occur in naive prediction
            #assert isinstance(params["energy_price_buy"][0][0],(float,int))
            params[key] = [params["energy_price_buy"][s] * value for s in range(S)]
        else:
            params[key] = len_S_list(key, copy=True)
        
        params["dc_prev_max"] = len_S_list(fill_none("dc_prev_max", fill=0), copy=True)
        params["dc_price"] = fill_none("dc_price", 0.6)

        # [Yi, 2023/03/19] add [p_grid_max]
        p_grid_max = fill_none("p_grid_max", float('inf'))  # default is no limit
        if isinstance(p_grid_max, str):     # use string, e.g., "1.5" to define 1.5 x max(load_bld)
            # L.Notes: here the p_grid_max is the max bld_load given by \
                # the "predictor" in the fiture K steps, the multiplies a give coef p_grid_max
            p_grid_max = np.array(params["load_bld"]).max() * float(p_grid_max)
        params["p_grid_max"] = p_grid_max

        params["bat_p_min"] = fill_none("bat_p_min", params["bat_p_max"])
        params["bat_soc_0"] = len_S_list(fill_none("bat_soc_0", fill=0.5), copy=True)
        params["bat_soc_K"] = len_S_list(fill_none("bat_soc_K", fill=params["bat_soc_0"]), copy=True)

        # [Yi, 2023/03/15] "reg_lambda", "reg_term" is no longer used
        # [OLD] params["reg_lambda"] = fill_none("reg_lambda", 0)
        # [OLD] params["reg_term"] = fill_none("reg_term", "p_norm")
        # instead, use equiv cycle models
        deg_model = fill_none("deg_model_opt", "unconscious")
        params["deg_model_opt"] = deg_model
        if deg_model not in ["unconscious", "throughput"]:
            params["deg_thres_opt"] = fill_none("deg_thres_opt", self.battery_sample.get_params(f"deg_{deg_model}_thres"))
            params["deg_lambda_opt"] = fill_none("deg_lambda_opt", self.battery_sample.get_params(f"deg_{deg_model}_lambda"))

        for key in ["ev_p_max", "ev_p_min"]:
            value = fill_none(key, 0)
            if isinstance(value, int) or isinstance(value, float):
                params[key] = [np.array([value]*params["ev_I"][s]) for s in range(S)]
            else:
                params[key] = len_S_list(key, copy=False)

        params["ev_capacity"] = fill_none("ev_capacity", params["ev_e_targ"].copy())
        params["ev_efficacy"] = fill_none("ev_efficacy", 1)

        ev_charge_rule = fill_none("ev_charge_rule", "flex")
        assert ev_charge_rule in ["flex", "V2G", "unif", "asap"]

        # round ev_ta & ev_td
        #   ev_ta & ev_td can be floats
        #   e.g: ta = 1.5, Pmax = 6.6, then at t=1, pmax = 6.6*0.5=3.3
        # FIXME: if td > K? ta < 0?
        
        #print("menthod called out.")
        
        params["ev_pmax_frac"] = [0] * S
        for s in range(S):
            K = params["K"][s]
            delta = params["delta"][s]
            I = params["ev_I"][s]
            ta = params["ev_ta"][s]
            td = params["ev_td"][s]
            e_init = params["ev_e_init"][s]
            e_targ = params["ev_e_targ"][s]
            p_max = params["ev_p_max"][s]
            eta = params["ev_efficacy"]

            # [Yi, 2023/03/22] fix condition when I=0
            if I == 0:
                params["ev_pmax_frac"][s] = np.zeros((I, K))
                continue

            # [Yi, 2022/08/26]: check feasibility here
            # [Yi, 2023/02/10]: raise warning instead of error
            #   and automatically adjust e_targ if infeasible
            eps = 10 ** (-5)
            feas = (e_targ - e_init) / (p_max*eta) <= (td-ta)*delta + eps
            infeas_idx = list(np.argwhere(feas==False).flatten())
            if len(infeas_idx) > 0:
                warnings.warn("EV demand infeas: s={}, idx={}".format(s, infeas_idx))
            
            e_targ = np.minimum(e_targ, e_init + (td-ta)*delta*p_max*eta)

            # update 08/26 - Yi: (averagely) cut sessions with td>K apart
            assert sum(ta < 0) == 0
            # [Lunlong, 2023/07/28]: check if td or ta contains nan
            try:
                #print("menthod called.")
                td_cut = np.minimum(td, K)
                #print("td:",len(td),"td_cut:",len(td_cut),"ta:",len(ta))
                #print("td-ta: ",td-ta,"\n t_cut-ta",td_cut-ta)
                e_targ_cut = e_init + (e_targ - e_init) * (td_cut-ta)/(td-ta)
            except Exception as e:
                print("td:",td,"\ntd_cut:",td_cut,"\nta:",ta)
                print("reason:",e)

            
            # [Yi, 2023/02/02] correct typo "e_targ" -> "ev_e_targ"
            params["ev_e_targ"][s] = e_targ_cut

            td = td_cut
            ta_int = np.floor(ta).astype(int)
            td_int = np.ceil(td).astype(int)

            ref_matrix = np.vstack([np.arange(K)]*I)
            ev_pmax_frac = ((ref_matrix >= ta_int.reshape(-1,1)) &
                            (ref_matrix < td_int.reshape(-1,1))).astype(float)
            for i in range(I):
                # [Yi, 2023/2/10] correct a bug:
                # ta[i] - ta_int[i] -> 1 - (ta[i]-ta_int[i])
                ev_pmax_frac[i, ta_int[i]] = 1 - (ta[i] - ta_int[i])
                ev_pmax_frac[i, td_int[i]-1] = td[i]+1 - td_int[i]
            
            params["ev_ta"][s] = ta_int
            params["ev_td"][s] = td_int


            # [Yi, 2023/03/20] 
            if ev_charge_rule == "flex":
                # discharge is disabled
                params["ev_p_min"][s] *= 0
            
            if ev_charge_rule == "unif":
                e_req = e_targ_cut - e_init # energy required
                e_max = ev_pmax_frac.sum(axis=1) * p_max * delta * eta  # maximum deliverable energy
                ev_pmax_frac *= (e_req/e_max)[:,None]
                # thus the only way to meet required energy is to keep charging uniformly
            
            if ev_charge_rule == "asap":
                e_req = e_targ_cut - e_init
                t_cum = np.hstack([np.zeros((I,1)), ev_pmax_frac.cumsum(axis=1)])
                t_req = e_req / (p_max * delta * eta)
                t_asap = np.minimum(t_cum, t_req[:,None])
                ev_pmax_frac = (t_asap[:,1:] - t_asap[:,:-1])  # shape = (I, K)


            params["ev_pmax_frac"][s] = ev_pmax_frac


        return params

    @staticmethod
    def params_sanity_check(params):

        S = params["S"]

        # TODO: check params contain all the keys
        ...

        def shape_S_assert(key, level2=None):
            try:
                assert len(params[key]) == S
            except:
                raise Exception("{} in wrong length. should be {}".format(key, S))
            if level2 is None:
                return
            
            for s in range(S):
                n = params[level2][s]
                try:
                    assert len(params[key][s]) == n
                except:
                    raise Exception("{} at s={} in wrong length. should be {}".format(s, key, n))
        

        for key in ["K", "delta", "bat_soc_0", "bat_soc_K", "ev_I"]:
            shape_S_assert(key)
        
        for key in ["load_bld", "load_pv", "energy_price_buy", "energy_price_sell"]:
            if key == "energy_price_sell" and params[key] is None:
                continue
            shape_S_assert(key, level2="K")

        for key in ["ev_ta", "ev_td", "ev_e_init", "ev_e_targ", "ev_p_max", "ev_p_min", "ev_capacity"]:
            shape_S_assert(key, level2="ev_I")
            
        for key in ["bat_soc_0", "bat_soc_K"]:
            assert params[key]>=np.zeros(len(params[key]))
            assert params[key]<=np.ones(len(params[key]))
        
        # TODO: check the shape of params["ev_pmax_frac"]
        ...

        # TODO: check charge sessions are feasible
        ...

    @staticmethod
    def solution_sanity_check(params, sol):
        S=params['S']
        bat_capacity = params["bat_capacity"]
        bat_efficacy = params["bat_efficacy"]
        # sol keys: p_grid, bat_p, bat_e, ev_p, ev_e
        for s in range(S):
            K = params["K"][s]
            delta = params["delta"][s]
            
            """ BAT """
            # assert sol["bat_e"][0] == params["bat_soc_0"][s] * bat_capacity
            assert np.isclose(sol["bat_e"][s][K], params["bat_soc_K"][s] * bat_capacity, rtol=1e-05, atol=1e-05, equal_nan=False) ,\
                f"Infeasible when checking bat terminal states: , got bat_e_actuall:"+str(sol["bat_e"][s][K])+"bat_e_target"+str(params["bat_soc_K"][s] * bat_capacity)
            for k in range(K):
                if sol["bat_p"][s][k]>0:
                    assert (sol["bat_e"][s][k+1] == sol["bat_e"][s][k] - delta *\
                        sol["bat_p"][s][k]/bat_efficacy for k in range(K)),\
                           f"Infeasible when checking bat_e of step k:"+str(k)\
                               +str(sol["bat_e"][s][k+1])+"!="\
                               +str(sol["bat_e"][s][k])+'-'\
                               +str( delta)+'*'+str(sol["bat_p"][s][k])+'/'+str(bat_efficacy)
                else:
                    assert (sol["bat_e"][s][k+1] == sol["bat_e"][s][k] - delta *\
                        sol["bat_p"][s][k]*bat_efficacy for k in range(K)),\
                           f"Infeasible when checking bat_e of step k:"+str(k)\
                               +str(sol["bat_e"][s][k+1])+"!="\
                               +str(sol["bat_e"][s][k])+'-'\
                               +str( delta)+'*'+str(sol["bat_p"][s][k])+'*'+str(bat_efficacy)
                   
            # bat power limit
            assert (sol["bat_p"][s][k]>= -1*(bat_capacity / params["bat_p_min"])&\
                sol["bat_p"][s][k]<=bat_capacity / params["bat_p_max"] for k in range(K))
            # bat capacity
            assert (sol["bat_e"][s][k] <= bat_capacity for k in range(K+1))
            
            """ EV """
            I = params["ev_I"][s]
            pmax_frac = params["ev_pmax_frac"][s]
            ta, td = params["ev_ta"][s], params["ev_td"][s]
            e_init, e_targ = params["ev_e_init"][s], params["ev_e_targ"][s]
            # ev e state
            assert (sol["ev_e"][s][i,ta[i]] == e_init[i] for i in range(I))
            assert (sol["ev_e"][s][i,td[i]] == e_targ[i] for i in range(I))
            # ev charge
            ev_efficacy = params["ev_efficacy"]
            #print("EV solutions: ",sol["ev_e"],sol["ev_p"])
            for i in range(I):
                for k in range(K):
                    if sol["ev_p"][s][i,k]>0:
                        assert np.isclose(sol["ev_e"][s][i, k+1] ,sol["ev_e"][s][i, k] + delta*\
                            sol["ev_p"][s][i,k]*ev_efficacy,rtol=1e-05, atol=1e-05, equal_nan=False),\
                                f"Infeasible when checking ev charging state: ev:"+\
                                str(sol["ev_e"][s][i, k+1])+"!="+str( sol["ev_e"][s][i, k])+"+"+str(delta*sol["ev_p"][s][i,k]*ev_efficacy)
                        assert (sol["ev_p"][s][i,k]-params["ev_p_max"][s][i] * pmax_frac[i,k]) <= 1e-05 ,\
                            f"Infeasible when checking ev charging state: ev:"+\
                                str(sol["ev_p"][s][i,k])+"!<="+str( params["ev_p_max"][s][i])+"*"+str(pmax_frac[i,k])
                    else:
                        assert np.isclose( sol["ev_e"][s][i, k+1] ,sol["ev_e"][s][i, k] + delta*\
                            sol["ev_p"][s][i,k]/ev_efficacy,rtol=1e-05, atol=1e-05, equal_nan=False),\
                                f"Infeasible when checking ev charging state: ev:"+\
                                str(sol["ev_e"][s][i, k+1])+"!="+str( sol["ev_e"][s][i, k])+"+"+str(delta*sol["ev_p"][s][i,k]/ev_efficacy)
                        assert (sol["ev_p"][s][i,k]-(-1)*params["ev_p_min"][s][i] * pmax_frac[i,k]>= 1e-05*(-1)),\
                            f"Infeasible when checking ev charging state: ev:"+\
                                str(sol["ev_p"][s][i,k])+"!>="+str( params["ev_p_min"][s][i])+"*"+str(pmax_frac[i,k])
            assert (sol["ev_e"][s][i,k]-params["ev_capacity"][s][i] <= 1e-05 for i in range(I) for k in range(K+1)),f"Infeasible when checking ev charging state: ev:"+\
                    +sol["ev_e"][s]+"should all be <="+params["ev_capacity"][s]
            
            # energy balence
            load_bld = params["load_bld"][s]
            load_pv = params["load_pv"][s]
            assert (load_pv[k] + sol["p_grid"][k] + sol["bat_p"][k] >=
                load_bld[k] + sum(sol["ev_p"][s][i,k] for i in range(I)) for k in range(K))                   

    @staticmethod
    def search_minimum(lim, func, split=5, tol=0.01, **kw):

        xs = np.linspace(lim[0], lim[1], split+1)
        y0 = np.inf
        i0, i1 = 0, split
        for i in range(len(xs)):
            x = xs[i]
            y = func(x, **kw)
            if y <= y0-tol:
                y0 = y
                i0 = i
            elif y >= y0 + tol:
                i1 = i
                break
        # true only if split > 3
        if i0 == 0 and i1 == split:
            print("Found")
            return (xs[i0]+xs[i1])/2
        else:
            lim = (xs[i0], xs[i1])
            print("X", end=" ")
            return Battery_optimizer.search_minimum(lim, func, split, tol, **kw)

    def get_params_sample(self):
        return self.__params_sample