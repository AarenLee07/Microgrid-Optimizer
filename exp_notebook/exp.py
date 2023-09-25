import os
import sys
src_path = sys.path[0].replace("exp_notebook", "src")
#replace notebook as scripts
data_path = sys.path[0].replace("exp_notebook", "data")
if src_path not in sys.path:
    sys.path.append(src_path)
out_path = sys.path[0].replace("exp_notebook", "output")
#sys.path.append("..")

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import copy

from grid_manager import MPC_op
from data_loader import UCSD_dataloader
from battery_model import Battery_base
from optimizer import Battery_optimizer
from predictor import *
from exp_manager import ExperimentManager
from threading import Thread
from time import sleep,ctime

# from ev_manager import EVmanager
# import copy

bat_params_sample ={
    "bat_capacity": None,
    "bat_p_max": 3, # i.e., capacity (kWh) / p_bat_max (kW) = 3 (h)
    "bat_p_min": 3, # can omit, then p_bat_min = p_bat_max
    "bat_price": 150, # $/kWh (old: 1000, ref: Tesla Powerwall)
    "bat_efficacy": 0.98, 
    "bat_life_0": 3650, # days.
    "bat_cycle_0": 3000, # cycles in lifetime
    # battery degradation params
    "deg_model": "throughput",  
        # valid values: "throughput", "Crate", "rainflow", "DOD"
    #   [1. degradation ~ high C-rate]
    "deg_Crate_thres": (0.25, 0.25, 0.25, 0.25),
    "deg_Crate_lambda": (0.8, 1, 1.5, 2),
    #   [2. degradation ~ large cycle depth]
    "deg_rainflow_thres": (0.2, 0.2, 0.2, 0.4),
    "deg_rainflow_lambda": (0.6, 1, 1.5, 1.8),
    #   [3. degradation ~ low SoE range]
    "deg_DOD_thres": (0.5, 0.2, 0.2, 0.1),
    "deg_DOD_lambda": (1.3, 1.15, 0.85, 0.6),
}

op_params_sample = {"K": 96,
            "dc_price": 0.6,
            "ev_efficacy": 0.98,
            "energy_price_sell": 0.6, 
            "deg_model_opt": "unconscious",
            "ev_charge_rule": "flex",
            "ev_charge_rule_default": "unif",
            "p_grid_max": "1.5",}





class MPC_ExperimentManager(ExperimentManager):
    """
    Main variable: battery size (normalized to hr: to bld_load)
    """

        
    def run_one_trial(self, params, save_fn, fork_id):
        
        # params: keys: "pred_model", "strategy", "B_kWh", "deg_model_opt", "deg_model", 
        #   "start", "end", "p_grid_max", "price_dc", "price_sell", "ev_charge_rule"

        op_params = copy.deepcopy(op_params_sample)
        bat_params = copy.deepcopy(bat_params_sample)

        bat_params["deg_model"] = params.get("deg_model", "DOD")
        bat_params["bat_capacity"] = params.get("B_kWh", 350)
        strategy = params.get("strategy", "optimal")
        op_params["deg_model_opt"] = params.get("deg_model_opt", "rainflow") 
        op_params["ev_charge_rule"] = params.get("ev_charge_rule", "flex")
        op_params["energy_price_sell"] = params.get("price_sell", "rainflow")
        op_params["dc_price"] = params.get("price_dc", 0.6) 
        p_grid_max = params.get("p_grid_max", "1.5") 
        op_params["penalty_coef"]=params.get("penalty_coef", 0)
        op_params["sol_save_steps"] = params.get("sol_save_steps",0)
        op_params["dc_formulation"]=params.get("dc_formulation", "moving")
        op_params["disturbance_rule"]=params.get("disturbance_rule", "uniform")
        op_params["disturbance_scale"]=params.get("disturbance_scale", 0.03)
        op_params["p_grid_max_method"]=params.get("p_grid_max_method","by_execution")
        op_params["p_grid_max"] = None if p_grid_max is None else str(p_grid_max)
        op_params["shift"]=params.get("shift",False)
        op_params["shift_ratio"]=params.get("shift_ratio",0)
        pred_model = params.get("pred_model", "GT") 
        print("pred_model:",pred_model)
        if pred_model=='GT':
            op_params["check_inconsistency"]=params.get("check_inconsistency",True)
        else:
            op_params["check_inconsistency"]=False
        pv_to_bld=params.get("pv_to_bld",0.5)
        ev_to_bld=params.get("ev_to_bld",0.25)
        
        # 2023/05/30 LunLong
        # Trying to add bld pv ev into grid search parmas
        bld=params.get("bld", "Hopkins")
        pv=params.get("pv", "Hopkins")
        ev=params.get("ev","OSLER")   
        
        # 2023/07/21 LunLong
        # Check params sanity with XGBoost prediction added
        #if params.get("pred_model")=='Prediction':
        #    if not params.get("bld")=="Sum" & params.get("pv")=='Sum':
        #        raise Exception("Incompatible pred_model and bld/pv setting.")
        
        if bld is None or pv is None or ev is None:
            raise Exception("Building related params are not prepoperly initiated")

        def convert_time(s):
            idx_hyphen = s.index("-")
            month, day = int(s[:idx_hyphen]), int(s[idx_hyphen+1:])
            return datetime(2019, month, day, 0, 0)
        t_start = convert_time(params.get("start", "10-1"))
        t_end = convert_time(params.get("end", "10-8"))
        
        mpc = MPC_op()

        # Step 1: load data
        mpc.load_data(loader=UCSD_dataloader, 
            tstart_historical=datetime(2018,1,1,0,0),
            tstart_execution=datetime(2019,1,1,0,0),
            tend=datetime(2019,12,31,23,59), delta=0.25,
            bld=bld, pv=pv, ev=ev, pv_to_bld=pv_to_bld, ev_to_bld=ev_to_bld, Pmax=10,
            pred_model=pred_model)

        '''
        # Step 2: Load historical data
        # ! add pred_model here
        if pred_model=="GT":
            mpc.init_historical_data(loader=UCSD_dataloader,
                tstart=datetime(2019,1,1,0,0), tend=datetime(2019,12,31,23,59), delta=0.25,
                bld=bld, pv=pv, ev=ev, pv_to_bld=0.5, ev_to_bld=0.25, Pmax=10, pred_model=pred_model)
        else:
            mpc.init_historical_data(loader=UCSD_dataloader,
                tstart=datetime(2018,11,30,0,0), tend=datetime(2019,12,31,23,59), delta=0.25,
                bld=bld, pv=pv, ev=ev, pv_to_bld=0.5, ev_to_bld=0.25, Pmax=10, pred_model=pred_model)
        '''

        # Step 3: specify other operational params
        optimizer_params = {"strategy": strategy, "language":"gurobi"}

        mpc.init_op_params(optimizer=Battery_optimizer, optimizer_params=optimizer_params, delta_0=0.25, **op_params)

        # Step 4: specify battery
        mpc.init_battery(model=Battery_base, params=bat_params, delta_0=0.25)

        # Step 5: initialize predictor
        # [Yi, 2023/03/08] modify predictor def
        if pred_model=="Disturbance":
            bld_kws={
                "rule":op_params["disturbance_rule"],
                "loc":0,
                "scale":op_params["disturbance_scale"],
            }
            mpc.init_predictor(shortcut=pred_model, bld_kws=bld_kws,shift=op_params["shift"],shift_ratio=op_params["shift_ratio"],delta=0.25)
        else:
            mpc.init_predictor(shortcut=pred_model, bld_kws=None,shift=op_params["shift"],shift_ratio=op_params["shift_ratio"],delta=0.25)

        # Step 6: initialize save_config
        mpc.init_save_config(save_fn=save_fn[:-5],  # FIXME: remove ".xlsx"
            folder_path=self.save_path, log_pred_action=True,
            checkpoints="1D", recovery=True, recovery_from=None,
            )

        mpc.run(tstart=t_start, tend=t_end, exe_K=4, save=True, fork_id=fork_id)
        
        stats = dict(mpc.summary["All"])
        
        return stats

