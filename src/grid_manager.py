import os
import sys

out_path = sys.path[0].replace("notebooks", "output")

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from battery_model import Battery_base
from optimizer import Battery_optimizer
from data_loader import UCSD_dataloader, XGB_dataloader
from ev_manager import EVmanager
from predictor import *
from data_pool import DataPool
import xlsxwriter
import copy

first_start=True




"""
============================= UPDATES =============================
(write down updates on the code - the last updated one at the first)
===================================================================
# [Yi, 2023/03/22]
- add arg [log_pred_action] for [init_save_config]
    if True, the scheduled actions will be logged
    modifications are across [run / run_k_steps / save / recovery]

# [Yi, 2023/03/08]
- modify [init_predictor] corresponding to change in [Predictor]
- TODO: add more info in [op_log], e.g., load, price, pred_load, etc. ?
- TODO: check price_sell setting works properly

# [Yi, 2023/03/04]
 - FIXME: short-circuit [data_pool.update] method in [run_k_steps]
    currently, always load all data at start, so the historical data need not to be updated
    this should be modified for real-world implmentation purpose !

    # [Yi, 2023/02/11] copy from notebook
"""


class MPC_op():

    def __init__(self):
        
        self.data = None        # dict, keys: "load_bld", "load_pv", "ev_sessions"
                                # set by [load_data]
        self.battery = None     # a battery object, e.g., Battery_base
                                # set by [init_battery]
        
        self.op_params = None   # dict, set by [init_op_params]
        self.delta_0 = None     # unit: hour, set by [init_op_params]
        self.optimizer = None   # dict, keys: "optimizer", "params", set by [init_op_params]
        
        self.data_pool = None   # 
        self.data_pool_xgb = None
        self.predictor = None   # a Predictor object, set by [init_predictor]
        
        self.save_config = None # dict, set by [init_save_config]
        self.checkpoints = None # pd.data_range, set in [run]

        self.ev_log = EVmanager()
        self.op_log = None

        # [Yi, 2023/03/19]
        self.summary = None     # will be updated by method [op_summary]
        # [Lunlong, 2023/08/08] to calculate the summary of first solved step
        self.summary_one_step = None # will be updated by method [op_summary]
        self.summary_one_step_flag = False # work as a mark of whether saved summary one step

        # [Lunlong, 2023/08/08] add "curr_sol" to save sol for calculating summary_one_step
        self.cache = {"op_params": None,
                      "ev_p_sol": None, "ev_index": None,
                      "EV_onsite": self.ev_log.onsite_table.copy(),
                      "EV_log": self.ev_log.logging.copy(),
                      "curr_sol":None}
    
        # [Lunlong, 2023/08/21] add params to log the latest pred_error
        self.latest_max_bld_error_neg=0
        self.latest_max_bld_error_pos=0
        self.latest_max_pv_error_neg=0
        self.latest_max_pv_error_pos=0
        self.latest_max_ev_error_neg=0
        self.latest_max_ev_error_pos=0
        self.latest_max_net_error_neg=0
        self.latest_max_net_error_pos=0
        
        self.first_start_flag=0
        self.lastest_p_prev_max=0


    """ the following methods are PUBLIC methods
        i.e., you can call them directly outside    
    """

    def run(self, tstart=None, tend=None, exe_K=1, t_cut=False, save=False, fork_id=None, run_bat_as_sol=False):
        
        # check source data & predictor has been initialized
        if self.data is None:
            raise Exception("please call method \'load_data\' first")
        if self.predictor is None:
            raise Exception("please call method \'init_predictor\' first")
        if self.optimizer is None or self.op_params is None:
            raise Exception("please call method \'init_op_params\' first")
        if self.save_config is None and save==True:
            raise Exception("please call method \'init_save_config\' first")

        tstart = self.data["load_bld"].index.min() if tstart is None else tstart
        tend = self.data["load_bld"].index.max() if tend is None else tend
        
        self.op_log = pd.DataFrame(
            index = pd.date_range(tstart, tend, freq="{}H".format(self.delta_0)),
            columns = ["p_grid","sol_p_grid", "bat_p", "bat_e", "ev_p", "ev_I", "unseen_ev_I", "sol_ev_p",
                        "load_bld", "load_pv","net_load", "tou_import", "tou_export",
                        "opex", "solve_time","latest_p_grid_max",
                        "load_bld_error","load_pv_error",
                        "ev_error","net_load_error"],
        )


        if self.log_pred_action:
            self.pred_action_log = dict()
            for key in ["load_bld", "load_pv", "p_grid", "bat_p", "ev_p"]:
                self.pred_action_log[key] = pd.DataFrame(
                    index = pd.date_range(tstart, tend, freq="{}H".format(self.delta_0*exe_K)),
                    columns = range(self.op_params.get("K")),
                )

        if save == True:
            recovery = self.save_config["recovery"]
            checkpoints = self.save_config["checkpoints"] # ? shape like
            if recovery:
                tstart_r = self.recovery()
                tstart = tstart_r if tstart_r is not None else tstart
            if checkpoints is not None:
                self.checkpoints = pd.date_range(tstart, tend, freq=checkpoints) # deleted closed="right"
        
        # [Yi, 2023/03/19] # use string, e.g., "1.5" to define 1.5 x max(load_bld)
        p_grid_max = self.op_params.get("p_grid_max")
        if isinstance(p_grid_max, str):
            self.op_params["p_grid_max"] = float(p_grid_max) * self.data["load_bld"].max()


        t = tstart
        """ records will be saved under following circumstances:
            - t is a checkpoint
            - there is an exception raised when running
            - all steps have completed
        """
        while t < tend:
            if save == True and \
             self.checkpoints is not None and t in self.checkpoints:
                # save records at checkpoints
                op_log_curr = self.op_log.dropna()
                if len(op_log_curr) > 0:
                    self.op_summary(op_log_curr)
                ''''
                if len(op_log_curr) > 0 and t == tstart:
                    self.summary_one_step=op_summary_one_step_temp
                    print("summary saved at:",t)
                    self.save()
                    #self.summary_one_step_flag==False
                    self.summary_one_step=None
                '''
                self.save()
                
            
            #try:
            
            # [Lunlong, 2023/08/08] to get the summary at first time step
            if save == True and t == tstart:
                temp_MPC_op=copy.deepcopy(self)
                print("self.op_params[K]:",self.op_params["K"])
                temp_MPC_op.run_k_steps(t=t, exe_K=self.op_params["K"]-1, t_cut=t_cut, fork_id=fork_id, tstart=tstart,run_bat_as_sol=run_bat_as_sol)
                op_log_temp = temp_MPC_op.op_log.dropna().copy()
                assert len(op_log_temp)>0
                self.summary_one_step=temp_MPC_op.op_summary(op_log_temp).copy().T
                self.save()
                #print("op_log_curr:",op_log_temp)
                print("summary saved at:",t)
                del temp_MPC_op
                
            '''            
            except Exception as e:
            # ev_log has already been edited - set them back
            self.ev_log.onsite_table = self.cache["EV_onsite"]
            self.ev_log.logging = self.cache["EV_log"]
            # save results when there's an exception
            if save == True:
                self.save()
            print(e)
            raise KeyboardInterrupt()
            '''
            self.run_k_steps(t, exe_K=exe_K, t_cut=t_cut, fork_id=fork_id, tstart=tstart, run_bat_as_sol=run_bat_as_sol)
            t += timedelta(hours=self.delta_0*exe_K)
        
        # save records when all complete
        if save == True:
            op_log_curr = self.op_log.dropna()
            if len(op_log_curr) > 0:
                self.op_summary(op_log_curr)
            self.save()
        print("="*20, "FINISH", "="*20)
         
    # [LunLong 2023/08/20] merge load_data() and init_historical_data() as one      
    # tstart_historical_data, tstart_execution_data, tend # ! should share the same end 
    '''
    def load_data(self, loader=UCSD_dataloader, **kw):
        # [Yi, 2023/03/13] force to fill na
        
        loaded = loader(fillna=True, **kw)
        self.data = loaded.get_data()
    '''
    def load_data(self, loader=UCSD_dataloader, 
                             tstart_historical=datetime(2018,1,1,0,0),
                             tstart_execution=datetime(2019,1,1,0,0),
                             tend=datetime(2019,12,31,23,59),**kw):
        # check feasibility of input time
        assert tstart_historical<=tstart_execution
        assert tstart_execution<tend
        
        # old method of loading data, loading data seperated of two periods
        '''
        # load prediction ref, including load_bld, load_pv and ev_sessions
        kw["tstart"]=tstart_historical
        kw["tend"]=tstart_execution
        loaded_pred_ref = loader(fillna=True, **kw).get_data()
        
        # load GT, including load_bld, load_pv and ev_sessions
        kw["tstart"]=tstart_execution
        kw["tend"]=tend
        loaded_GT = loader(fillna=True, **kw).get_data()
        self.data=loaded_GT.copy() # ground_truth for all cases as the ref of execution
        
        # init a tmp dict for the prediction dict
        loaded_pred=loaded_GT.copy()
        
        # common part for all pred_models  
        for key in ["load_bld","load_pv"]:
            assert (key in list(loaded_pred_ref.keys()) and key in list(loaded_pred.keys()))
            # concat two part of the load
            load_tmp=pd.concat([loaded_pred[key],loaded_pred_ref[key]],axis=0)
            # sort by timeindex
            load_tmp=load_tmp.sort_index()
            # drop duplicates based on timeindex, however only few values may be duplicated 
            #   due to the overlap on the boundary 
            load_tmp = load_tmp[~load_tmp.index.duplicated(keep='first')]
            loaded_pred[key]=load_tmp.copy()
        
        # load XGB static data into a saperate table
        if kw["pred_model"]=="Prediction":
            kw["tstart"]=tstart_execution
            kw["tend"]=tend
            load_XGB=XGB_dataloader(fillna=True, **kw).get_data() # load_XGB contains only load_bld and load_pv
            loaded_XGB=loaded_pred.copy()
            # copy from the static data directly
            for key in ["load_bld","load_pv"]:
                assert key in list(load_XGB.keys())
                loaded_XGB[key]=load_XGB[key].copy()
            self.data_pool_xgb = DataPool(loaded_XGB.copy()) 
                
        # all pred_models share the same set of ev_sessions
        if kw["ev"] is not None:
            assert ("ev_sessions" in list(loaded_pred_ref.keys()) and "ev_sessions" in list(loaded_pred.keys()))
            ev_tmp=loaded_pred_ref["ev_sessions"].copy()
            count_ev=len(loaded_pred["ev_sessions"])
            ev_tmp.index=ev_tmp.index+count_ev
            ev_tmp_to_sort=pd.concat([loaded_pred["ev_sessions"],ev_tmp],axis=0)
            ev_tmp_sorted=ev_tmp_to_sort.sort_values(by="ta")
            loaded_pred["ev_sessions"]=ev_tmp_sorted

        self.data_pool = DataPool(loaded_pred.copy()) 
        '''
        
        
    
        # new method of loading data
        
        # get data of the full duration ["load_bld","load_pv","ev_sessions"] for gt
        #   as well as for prediction models excluding "XGBoost"
        kw["tstart"]=tstart_historical
        kw["tend"]=tend
        loaded_GT = loader(fillna=True, **kw).get_data()
        self.data=loaded_GT.copy()
        self.data_pool = DataPool(loaded_GT.copy()) 
        
        # for xgb prediction, load_bld and load_pv are loaded from static file but ev_sessions not implemented
        #   as for load_bld, it's the benchmark of recaling other load, inconsistency of loading period in gt and pred_ref doesnt matter
        #   while not the same story for load_pv
        #   so that we need to calculate the average bld_load of [tstart_execution,tend] in the gt as rescaling reference
        
        bld_load=self.data["load_bld"]
        execution_period=bld_load[tstart_execution:tend]
        load_ave_exec=np.mean(execution_period,axis=0)
        load_ave_all=np.mean(bld_load,axis=0)
        
        if kw["pred_model"]=="Prediction":
            kw["tstart"]=tstart_execution
            kw["tend"]=tend
            folder=r'load_forecast\\'+kw["pred_method"]
            load_XGB=XGB_dataloader(fillna=True, bld_load_mean=load_ave_exec,folder=folder, **kw).get_data() # load_XGB contains only load_bld and load_pv
            loaded_XGB=loaded_GT.copy()
            # copy from the static data directly
            for key in ["load_bld","load_pv"]:
                assert key in list(load_XGB.keys())
                loaded_XGB[key]=load_XGB[key].copy()
            self.data_pool_xgb = DataPool(loaded_XGB.copy()) 
        
        return
        

    def init_battery(self, model=Battery_base, **kw):
        # [Yi, 2023/03/19]
        self.battery = model(deg_model_only=True, **kw)
        bat_capacity = self.battery.get_params("bat_capacity")
        
        # [LunLong, 2023/07/23]
        # set the coef of e_curr to be 0 to avoid msc "Cheating" by selling all capacity
        self.battery.set_params(e_curr=0*bat_capacity)
        self.battery_est = self.battery.copy_params(states=True, capacity=True, deg_model_only=True)
        self.battery_est.set_params(deg_model = self.op_params["deg_model_opt"])
        # FIXME: set battery capacity, set curr_e
    
    def init_op_params(self, optimizer=Battery_optimizer, optimizer_params=None, delta_0=0.25, **op_params):
        self.optimizer = {
            "optimizer": optimizer,
            "params": optimizer_params 
        }
        self.delta_0 = delta_0
        
        op_params_default = {
            "K": 96,
            "dc_price": 0.6,
            "ev_efficacy": 1,
            "delta": delta_0,
            # [Yi, 2023/03/19] new keys to specify
            "energy_price_sell": 0, 
            "deg_model_opt": "unconscious",
            "ev_charge_rule": "flex",
            "ev_charge_rule_default": "unif",
            "p_grid_max": None,
            # [Lunlong, 2023/08/01] new keys for demand charge
            #"dc_formulation":"moving",
            #"penalty_coef":"0",
        }
        self.op_params = op_params_default
        self.op_params.update(op_params)
        # in self.run, update battery related params
       
    def init_save_config(self, save_fn=None, folder_path=None, log_pred_action=True,
                            checkpoints=None, recovery=True, recovery_from=None):
        if folder_path is None:
            folder_path = out_path
        assert os.path.exists(folder_path)
        
        # [Yi, 2023/02/11]
        if recovery:
            if recovery_from is None:
                recovery_from = os.path.join(folder_path, save_fn+".xlsx")
            # [Yi, 2023/02/11] no need to assert
            # assert os.path.exists(recovery_from)

        self.save_config = {
            "save_fn": save_fn,
            "folder_path": folder_path,
            "checkpoints": checkpoints,
            "recovery": recovery,
            "recovery_from": recovery_from
        }

        # [Yi, 2023/03/22] can log scheduled actions (but may not actually be executed)
        #   this functionality is implemented in
        #   - run / run_k_steps / save / recovery
        self.log_pred_action = log_pred_action

    def init_predictor(self, **kw):
        # [Yi, 2023/03/08] modify predictor def
        
        if kw["shortcut"]=="Prediction":
            assert self.data_pool_xgb is not None
            self.predictor = Predictor(data_pool=self.data_pool,
                                       data_pool_xgb=self.data_pool_xgb, **kw)
        else:
            self.predictor = Predictor(data_pool=self.data_pool, **kw)


    """ the following methods should be treated as PRIVATE methods
        i.e., don't call them directly outside
    """

    def run_k_steps(self, t, exe_K=1, t_cut=False, fork_id=None, tstart=None, run_bat_as_sol=False):
        # 0. start clock
        t_clock = time.perf_counter()
        if t.hour == 0:
            print("="*20, t,"thread_id:",str(fork_id), "="*20)    # FIXME: do not print too frequent
        
        
        """ STEP-1. update EV new_arrivals & new_departs """
        if self.data["ev_sessions"] is not None:
            self.update_onsite_ev(t)
        
        
        """ STEP-2. prepare params for CFTOC """
        # For more advanced battery model, battery parameters may update
        # for key in ["bat_capacity", "bat_p_max", "bat_p_min", "bat_efficacy"]:
        #     self.op_params[key] = self.battery.get_params(key)
        self.op_params.update(self.battery.get_params())
        
        params = self.op_params.copy()
        K, delta_0 = params["K"], self.delta_0
        
        pred = self.predictor.get_prediction(t, K, delta=delta_0)
        # ! check if ev_sessions in the pred
        # print(pred)

        # load_bld, pv, energy_price
        for key in ["load_bld", "load_pv",
                    "energy_price_buy", "energy_price_sell"]:
            
            if pred[key] is not None:
                # "energy_price_sell" is likely to be None
                # then it will use the setting in self.op_params
                params[key] = pred[key]
                # FIXME: if sell=0.9 ?
        #print("params before solving mpc: load_bld",params["load_bld"])
        #print("params before solving mpc: load_pv",params["load_pv"])
        
        # track demand charge
        mo = t.month
        sig_bill = ((self.op_log.index.month == mo) & ~pd.isna(self.op_log["p_grid"]))
        #dc_prev_max=None
        if sum(sig_bill) > 0:
            # [Yi, 2023/02/02] correct typo: loc[..., "p"] -> loc[..., "p_grid"]
            # p_grid_max of all executed steps
            p_grid_exe_prev_max = max(0, self.op_log.loc[sig_bill,"p_grid"].max())
            # cal the previous timestep
            t_prev=t-timedelta(hours=self.delta_0*exe_K)
            # solution given by the last opt step
            sol_last_step=self.pred_action_log["p_grid"].loc[t_prev][:exe_K]
            # a temp container of p_grid_max
            dc_prev_max=0
            if self.first_start_flag>=1:
                if self.op_params['p_grid_max_method']=='minimize':
                    dc_prev_max = min(p_grid_exe_prev_max,max(sol_last_step))
                elif self.op_params['p_grid_max_method']=='minimize_cap':
                    if p_grid_exe_prev_max>max(sol_last_step):
                        dc_prev_max = min(p_grid_exe_prev_max,max(sol_last_step))
                    else:
                        dc_prev_max=p_grid_exe_prev_max
                elif self.op_params['p_grid_max_method']=='zero':
                    dc_prev_max=0
                elif self.op_params['p_grid_max_method']=='by_execution':
                    dc_prev_max=p_grid_exe_prev_max
                elif self.op_params['p_grid_max_method']=='by_solution':
                    assert self.lastest_p_prev_max != None
                    dc_prev_max=max(max(sol_last_step),self.lastest_p_prev_max)
                else :
                    raise Warning("unimplemented p_grid_max_method: ",self.op_params['p_grid_max_method']) 
            # for all methods the first step goes the same
            else:
                if self.op_params['p_grid_max_method']=='zero':
                    dc_prev_max=0
                else:
                    dc_prev_max=p_grid_exe_prev_max 
                self.first_start_flag=self.first_start_flag+1 
            params["dc_prev_max"] = dc_prev_max
            self.lastest_p_prev_max = dc_prev_max
        #else:
            #dc_prev_max=None
        # battery soc_0
        bat_capacity = params["bat_capacity"]
        bat_e_curr = self.battery.get_states("e_curr")
        if bat_e_curr is not None and bat_capacity != 0:
            params["bat_soc_0"] = bat_e_curr / bat_capacity
        else:
            self.battery.set_params(e_curr=0.5*bat_capacity)
            params["bat_soc_0"] = 0.5


        # EV
        ev_params = self.get_ev_params(t, pred["ev_sessions"])
        
        # [Yi, 2023/03/20] FIXME: do not know why, but this works
        #   otherwise, some of them will be in type "O"
        for key in ["ev_ta", "ev_td", "ev_e_init", "ev_e_targ", "ev_p_max"]:
            ev_params[key] = ev_params[key].astype(float)
        
        params.update(ev_params)
        
        self.cache["op_params"] = params.copy()
        #print("params load_bld: ",params["load_bld"])
        """ STEP-3. solve CFTOC """
        # [Yi, 2022/12/22] battery = self.battery (used as battery_sample)
        # see "optimizer" ln 25, ln 117
        opt = self.optimizer["optimizer"](battery = self.battery)
        # [Yi, 2023/2/10] cache the optimizer
        self.cache["opt"] = opt
        # [Yi, 2023/03/19] add ".sol": get_control_sequence returns a [Battery_solution] object, here we only use the solution part
        sol = opt.get_control_sequence(params, mute=True, **self.optimizer["params"]).sol
        self.cache["opt_sol"] = copy.deepcopy(sol)
        
        # [Lunlong, 2023/08/06] save solutions as csv for desiganed steps
        sol_k=self.op_params["sol_save_steps"]
        #print(sol)
        if sol_k>0:
            self.op_params["sol_save_steps"]=sol_k-1
            
            sol_df=pd.DataFrame(columns=["bat_p","bat_e","p_grid","load_bld","load_pv"])
            sol_df_ev=pd.DataFrame(columns=["ev_p","ev_e"])

            for k_2 in range(K):
                sol_df.loc[k_2]={
                    "bat_p":sol["bat_p"][0][k_2],
                    "p_grid":sol["p_grid"][0][k_2],
                    "bat_e":sol["bat_e"][0][k_2],
                    "load_bld":params["load_bld"][k_2],
                    "load_pv":params["load_pv"][k_2],
                }
            for k_3 in range(len(sol["ev_p"][0])):
                sol_df_ev.loc[k_3]={
                    "ev_p":sol["ev_p"][0][k_3],
                    "ev_e":sol["ev_e"][0][k_3],
                }
            sol_path=self.save_config['folder_path']+'\\'+self.save_config['save_fn'][:-4]+"sol_"+str(sol_k)+'.xlsx'
            writer = pd.ExcelWriter(sol_path, engine="xlsxwriter")
            sol_df.to_excel(writer,sheet_name='op_log')
            sol_df_ev.to_excel(writer,sheet_name='ev_log')
            #writer.close()
            #sol_df.to_csv(sol_path)
        #print(save_config)

        # FIXME: execution time of STEP-4 ignored
        t_last = time.perf_counter() - t_clock

        # [Yi, 2023/03/23]  can choose to log scheduled actions for next K steps
        if self.log_pred_action:
            #print("t:",t," range(K):",range(K))
            #print("self.pred_action_log\[\"load_pv\"\].loc[t, range(K)]:",list(self.pred_action_log["load_pv"].loc[t, range(K)]))
            #print("params\[\"load_pv\"\]: ",list(params["load_pv"].values))
            # [Lunlong 2023/08/23] add type check for load data,
            #   since both np.ndarray and Series occurred
            if isinstance(params["load_bld"],pd.Series):
                self.pred_action_log["load_bld"].loc[t, range(K)] = params["load_bld"].values
            elif isinstance(params["load_bld"],np.ndarray):
                self.pred_action_log["load_bld"].loc[t, range(K)] = params["load_bld"][0]
                raise Warning("params[\"load_bld\"] type error",type(params["load_bld"]))
            else:
                raise Exception("params[\"load_bld\"] type error",type(params["load_bld"]))
            
            if isinstance(params["load_pv"],pd.Series):
                self.pred_action_log["load_pv"].loc[t, range(K)] = params["load_pv"].values
            elif isinstance(params["load_pv"],np.ndarray):
                self.pred_action_log["load_pv"].loc[t, range(K)] = params["load_pv"][0]
                raise Exception("params[\"load_pv\"] type error",type(params["load_pv"]))
            else:
                raise Exception("params[\"load_pv\"] type error",type(params["load_pv"]))
            
            #self.pred_action_log["load_pv"].loc[t, range(K)] = params["load_pv"].values
            self.pred_action_log["p_grid"].loc[t, range(K)] = sol["p_grid"][0]
            self.pred_action_log["bat_p"].loc[t, range(K)] = sol["bat_p"][0]
            self.pred_action_log["ev_p"].loc[t, range(K)] = sol["ev_p"][0].sum(axis=0)

        #print("self.pred_action_log.load_bld: ",self.pred_action_log["load_bld"])

        """ STEP-4. update results """
        for k in range(exe_K):
            # execute exe_K stpes
            
            if run_bat_as_sol==True:
                '''old way of execution start'''
                # update battery charge
                #   when battery has more complicated dynamics, soc_update rule may be different
                bat_p = sol["bat_p"][0][k]
                self.battery.update_soc(p = bat_p, delta=delta_0)
                self.battery_est.update_soc(p = bat_p, delta=delta_0)
                bat_e = self.battery.get_states("e_curr")
                
                # update ev charge
                ev_p, unseen_evs = self.update_ev_charge(t, sol=sol, curr_k=k, exe_K=exe_K)
                # [Yi, 2023/02/11]: add ev_I to track
                ev_p_sum, ev_I = np.sum(ev_p), len(ev_p)

                # update p_grid
                exe_t = t + timedelta(hours=self.delta_0*k)
                load_pv = self.data["load_pv"].loc[exe_t]
                load_bld = self.data["load_bld"].loc[exe_t]

                p_grid = load_bld - load_pv + ev_p_sum - bat_p
                # bat_p neg means charging, pos means discharging
                
                sol_ev_p=0
                for i in sol["ev_p"][0]:
                    sol_ev_p+=i[k]
                
                '''old way of execution end'''
                
            elif run_bat_as_sol==False:
                
                '''new way of execution start'''
                # update battery charge
                #   when battery has more complicated dynamics, soc_update rule may be different
                
                bat_p = sol["bat_p"][0][k]
                
                # update ev charge
                ev_p, unseen_evs = self.update_ev_charge(t, sol=sol, curr_k=k, exe_K=exe_K)
                # [Yi, 2023/02/11]: add ev_I to track
                ev_p_sum, ev_I = np.sum(ev_p), len(ev_p)

                # update p_grid
                exe_t = t + timedelta(hours=self.delta_0*k)
                load_pv = self.data["load_pv"].loc[exe_t]
                load_bld = self.data["load_bld"].loc[exe_t]

                p_grid = load_bld - load_pv + ev_p_sum - bat_p
                # bat_p neg means charging, pos means discharging
                
                if p_grid>sol["p_grid"][0][k]+0.01:
                    #print("Compensation involved!")
                    if p_grid>self.lastest_p_prev_max:
                        mismatch=p_grid-self.lastest_p_prev_max
                        
                        # cal max posibile power of discharing from bat
                        bat_capacity=params["bat_capacity"]
                        bat_efficacy=params["bat_efficacy"]
                        bat_p_max=min(bat_capacity/params["bat_p_max"],bat_e_curr/delta_0)*bat_efficacy # pos means discharging
                        
                        compensation=min(mismatch,-bat_p+bat_p_max)
                        bat_p=bat_p+compensation
                        p_grid=p_grid-compensation
                
                self.battery.update_soc(p = bat_p, delta=delta_0)
                self.battery_est.update_soc(p = bat_p, delta=delta_0)
                bat_e = self.battery.get_states("e_curr")
                            
                
                sol_ev_p=0
                for i in sol["ev_p"][0]:
                    sol_ev_p+=i[k]
                
                '''new way of execution end'''
            
            
            
            
            # [Lunlong 2023/08/23]    
            # ! make an assumption that:
            # * The suboptimal of MPC under all kinds of prediction is caused by
            # *     the error of load prediction, specifically, the negative error (load_pred<load_truth),
            # *     which always makes the executed p_grid higher than the that of the solution.    
            # todo: So that the following changes should be made
            # * When p_grid in solution is lower that needed in the execution:
            # *     take extra power from the battery without violating the constraints
            # * Need to consider different cases:
            # *     A. bld_error+pv_error<0
            # *         first supply extra power from the battery then from the grid
            # *     B. bld_error+pv_error>0
            # *         if p_grid in the solution is above 0:
            # *             give it back to grid to lower p_grid in execution
            # *         else:
            # *             rule1 "battery_first": to charge battery first then export to grid
            # *             rule2 "to_grid": to export to grid
            # *             # unimplemented: rule3 "price_based": depending on the import price 
            # todo: How to decide charge battery of export to grid when scenerio A occurs
            
            # ! Such method may cause new bugs since the current exe_K=4
            # ! THIS METHOD IS NOT PLAUSIBLE IN REAL SCENERIO
            
            '''new way of execution start'''
            '''
            extra_p_rule="battery_first"
            
            # update ev charge
            # exceuted value
            ev_p = self.update_ev_charge(t, sol=sol, exe_k=k)
            # [Yi, 2023/02/11]: add ev_I to track
            ev_p_sum, ev_I = np.sum(ev_p), len(ev_p)
            # sol based on prediction
            sol_ev_p=0
            for i in sol["ev_p"][0]:
                sol_ev_p+=i[k]
            
            mismatch_ev=ev_p_sum-sol_ev_p
            assert abs(mismatch_ev)<=0.1 # currently, ev_pred are all gt, shouldnt mismatch
            
            exe_t = t + timedelta(hours=self.delta_0*k)
            load_pv = self.data["load_pv"].loc[exe_t]
            load_bld = self.data["load_bld"].loc[exe_t]
            mismatch_bld=load_bld-pred["load_bld"][exe_t]
            mismatch_pv=load_pv-pred["load_pv"][exe_t]
            
            # mismatch pos means need more power from bat and grid
            mismatch=mismatch_bld+mismatch_pv+mismatch_ev
            
            bat_p = sol["bat_p"][0][k] # get bat_p solution
            bat_capacity=params["bat_capacity"]
            bat_efficacy=params["bat_efficacy"]
            bat_p_max=min(bat_capacity/params["bat_p_max"],bat_e_curr/delta_0)*bat_efficacy # pos means discharging
            bat_p_min=max(-bat_capacity/params["bat_p_min"],-(bat_capacity-bat_e_curr)/delta_0)/bat_efficacy # neg means charging
        
            # if abs(mismatch)>=0.01:
            #    print("mismatch: ",mismatch)
            if mismatch>=0: # need to check the bat_p_max and bat_p_min here
                bat_p=min(bat_p+mismatch, bat_p_max)
            # mismatch<0 means less power needed than in solution
            elif sol["p_grid"][0][k]<=0: 
                # if p_grid in the solution is less than 0
                #   extra p be for charing bat or export to grid depending on rules
                if extra_p_rule=="battery_first":
                    bat_p=max(bat_p+mismatch, bat_p_min)
                elif extra_p_rule=="to_grid":
                    pass
                elif extra_p_rule=="price_based":
                    raise Warning("extra_p_rule unimplemented:",extra_p_rule)
            else:
                # if p_grid in the solution is above 0, give it back, lower it down
                print("p_grid in solution is higher than execution")
                # if more surplus after giving back to the grid
                if abs(mismatch)>sol["p_grid"][0][k]:
                    if extra_p_rule=="battery_first":
                        mismatch_inner=mismatch+sol["p_grid"][0][k]
                        bat_p=max(bat_p+mismatch_inner, bat_p_min)
                    elif extra_p_rule=="to_grid":
                        pass
                    elif extra_p_rule=="price_based":
                        raise Warning("extra_p_rule unimplemented:",extra_p_rule)
                else:
                    pass
            '''    
                
            '''new way of execution end'''   
            
            
            
            # calculate pred_error of current step
            self.latest_max_bld_error_neg=min(pred["load_bld"][exe_t]-load_bld,
                                            self.latest_max_bld_error_neg,0)
            self.latest_max_bld_error_pos=max(pred["load_bld"][exe_t]-load_bld,
                                            self.latest_max_bld_error_pos,0)
            self.latest_max_pv_error_neg=min(pred["load_pv"][exe_t]-load_pv,
                                            self.latest_max_pv_error_neg,0)
            self.latest_max_pv_error_pos=max(pred["load_pv"][exe_t]-load_pv,
                                            self.latest_max_pv_error_pos,0)
            self.latest_max_ev_error_neg=min(sol_ev_p-ev_p_sum,
                                            self.latest_max_ev_error_neg,0)
            self.latest_max_ev_error_pos=max(sol_ev_p-ev_p_sum,
                                            self.latest_max_ev_error_pos,0)
            net_load_error=(pred["load_bld"].loc[exe_t]-load_bld)+\
                    (pred["load_pv"].loc[exe_t]-load_pv)+(sol_ev_p-ev_p_sum)
            self.latest_max_net_error_neg=min(net_load_error,
                                            self.latest_max_net_error_neg,0)
            self.latest_max_net_error_pos=max(net_load_error,
                                            self.latest_max_net_error_pos,0)
            
            if self.op_params["check_inconsistency"]==True:
                if not np.isclose(p_grid,sol["p_grid"][0][k],rtol=0.01,atol=0.01) :
                    print(f"Infeasible when checking p_grid: "+str(p_grid)+" is not close to "+str(sol["p_grid"][0][k-1])+','+\
                        str(sol["p_grid"][0][k])+','+"exe_t:",exe_t,"\n",\
                            "load_bld vs sol_load_bld:",str(load_bld),",",str(params["load_bld"][k]),"\n",\
                            "load_pv vs sol_load_pv:",str(load_pv),",",str(params["load_pv"][k]),"\n",\
                            "load_ev vs sol_load_ev:" ,str(ev_p_sum) ,",",str(sol_ev_p),"\n",\
                            "bat_p:",str(bat_p))
            
            

            price_buy = params["energy_price_buy"][k]
            # FIXME: this can have many problems, e.g., 
            # params["energy_price_sell"]: 
            #   1. does not exist; 2. is None; 3. is a float; 4. is an array
            #   should we update by the regulated params?
            price_sell = params["energy_price_sell"] # a tentative solution
            if isinstance(price_sell, list) or isinstance(price_sell, np.ndarray):
                price_sell = price_sell[k]
            else:   # a fractional number
                price_sell = price_buy * price_sell
            opex = max(p_grid,0) * price_buy + min(p_grid,0) * price_sell
            
            # update historical data pool
            
            """  
            # [Yi, 2023/03/04] 
            # FIXME: the data_pool needs to be better designed
            #   now, always load all (incl. future) data at the beginning,
            #   and never call data_pool.update
            
            if "GT" not in self.predictor.name:
                self.data_pool.update(
                    {"load_pv": pd.Series({t: load_pv}),
                    "load_bld": pd.Series({t: load_bld}),
                    "ev_sessions": None})  # FIXME: EV sessions
            # FIXME: no need to update ground truth predictor
            #   since all data has been loaded at the beginning
            """

            # record operations in op_log (incl. correct p_grid)
            self.op_log.loc[exe_t] = {
                "sol_p_grid":sol["p_grid"][0][k],
                "p_grid": p_grid, "bat_p": bat_p, "bat_e": bat_e,
                "ev_p": ev_p_sum, "sol_ev_p": sol_ev_p, "ev_I": ev_I,
                "unseen_ev_I": unseen_evs,
                "net_load": ev_p_sum+load_bld+load_pv,
                "load_bld": load_bld, "load_pv": load_pv,
                "tou_import": price_buy, "tou_export": price_sell,
                "opex": opex, "solve_time": t_last / exe_K,
                "latest_p_grid_max": max(self.op_log["p_grid"]),
                "load_bld_error": pred["load_bld"].loc[exe_t]-load_bld,
                "load_pv_error": pred["load_pv"].loc[exe_t]-load_pv,
                "ev_error":sol_ev_p-ev_p_sum,
                "net_load_error": net_load_error
                
                #"load_ev_error": pred["ev_sessions"]
            }
            
        
        for key in ["op_params", "ev_p_sol", "ev_index"]:
            self.cache[key] = None
        # If interupted, ev_log has already been updated
        #   need to set it back
        self.cache["EV_onsite"] = self.ev_log.onsite_table.copy()
        self.cache["EV_log"] = self.ev_log.logging.copy()

        return

    def save(self):
        save_fn = self.save_config["save_fn"]
        folder_path = self.save_config["folder_path"]
        fn= os.path.join(folder_path, save_fn+".xlsx")
        #print("save method called.")
        # TODO: add a function [mpc_desc_cal] in utils:
        #   1. summarize the parameters in simulation
        #   2. compute performance stats 

        # desc = mpc_desc_cal(cftoc_params=self.cftoc_params,
        #         mpc_res_dfs={"op_log": self.op_log, "EV_log": self.ev_log.logging})
        #[Lunlong, 2023/08/08] when first write set mode as w 
        # to avoid summary_one_step being overwirtten under mode w
        # but it doesnt work properly
        '''
        if not os.path.exists(fn):
            writer = pd.ExcelWriter(fn,mode='w')
        else:
            writer = pd.ExcelWriter(fn,mode='a')'''
            
        writer = pd.ExcelWriter(fn,mode='w')  
        # [Yi, 2023/03/19] write summary
        if self.summary is not None:
            self.summary.to_excel(writer, "summary", index=True)
        # [Lunlong, 2023/08/18] write summary of first step    
        if self.summary_one_step is not None :
            self.summary_one_step.to_excel(writer, "summary_one_step", index=True)
            #print("summary_one_step saved")
            #self.summary_one_step_flag = True
        # [Yi, 2023/02/11] correct typo: opt_log -> op_log
        self.op_log.to_excel(writer, "op_log", index=True)
        self.ev_log.logging.to_excel(writer, "EV_log", index=True)
        self.ev_log.onsite_table.to_excel(writer, "EV_onsite", index=True)
        #assert self.log_pred_action==True
        #print("self.pred_action_log.load_bld: ",self.pred_action_log["load_bld"])
        if self.log_pred_action:
            #s=2
            #assert s==1
            #print("pred load save method called")
            for key, df in self.pred_action_log.items():
                df.to_excel(writer, f"pred_{key}", index=True)
        assert self.log_pred_action==True
        writer._save()
        #writer.close()
    
    def recovery(self):
        # TODO: need validation !
        
        # [Yi, 2023/02/11] use file defined in [recovery_from]
        #   instead of frome the same file as save_fn
        #   while in init_save_config, [recovery_from] is the same as save_fn by default
        fn = self.save_config["recovery_from"]
        # save_fn = self.save_config["save_fn"]
        # folder_path = self.save_config["folder_path"]
        # fn= os.path.join(folder_path, save_fn+".xlsx")

        if not os.path.exists(fn):
            return None
        log_dfs = pd.read_excel(fn, sheet_name=None, index_col=0, engine="openpyxl")
        # [Yi, 2023/02/11] correct typo: opt_log -> op_log
        op_log = log_dfs["op_log"]
        op_log.index = pd.to_datetime(op_log.index, infer_datetime_format=True)
        tstart_r = op_log.loc[~pd.isna(op_log["ev_I"])].index.max() + timedelta(hours=self.delta_0)
        self.op_log = op_log
        
        onsite_table = log_dfs["EV_onsite"]
        ev_log = log_dfs["EV_log"]
        for key in ["ta", "td", "td_actual"]:
            if key != "td_actual":
                onsite_table[key] = pd.to_datetime(onsite_table[key])
            ev_log[key] = pd.to_datetime(ev_log[key])
        # recover p_history
        def str2list(s):
            if not isinstance(s,str) or s=="":
                return s
            else:
                return [float(x) for x in list(s[1:-1].split(", "))]
        ev_log["p_history"] = ev_log["p_history"].apply(str2list)
        
        self.ev_log.onsite_table = onsite_table
        self.ev_log.logging = ev_log
        
        self.summary = log_dfs["summary"]
        e_curr = op_log.loc[~pd.isna(op_log["ev_I"])]["bat_e"].values[-1]
        days = self.summary.loc["days", "All"]
        eq_cycles = self.summary.loc["eq_cycles", "All"] * days
        eq_cycles_est = self.summary.loc["eq_cycles_est", "All"] * days
        self.battery.set_params(e_curr=e_curr, cycles_equiv=eq_cycles, working_days=days)
        self.battery_est.set_params(e_curr=e_curr, cycles_equiv=eq_cycles_est, working_days=days)

        if self.log_pred_action:
            try:
                for key in self.pred_action_log:
                    self.pred_action_log = log_dfs[f"pred_{key}"]
            except:
                print("Pred action log NOT FOUND")


        print("*"*25,"RECOVERY", "*"*25)
        print("-"*25,tstart_r, "-"*25)
        return tstart_r     

    def update_onsite_ev(self, t):
        ev_sessions = self.data["ev_sessions"]
        # 1. new_departures
        sig_depart = ev_sessions["td_actual"] <= t
        new_depart_idx = list(set.intersection(
                set(ev_sessions.loc[sig_depart].index), 
                set(self.ev_log.onsite_table.index)))
        self.ev_log.update(t, new_depart=new_depart_idx)
        # 2 new_arrivals
        # [Yi, 2023/02/10] FIXME: may not the best way
        #   only consider EVs havn't departed (departed EVs may not always in logging)
        #   old: (ta < t + delta) \ already_arrived
        #   new: (t <= ta < t + delta)
        sig_arrival = (ev_sessions["ta"] < t + timedelta(hours=self.delta_0)) &\
            (ev_sessions["ta"] >= t)
        # new_arrival_idx = ev_sessions.loc[sig_arrival].index
        # FIXME: this part can be more efficient (mark the last index ?)
        new_arrival_idx = list(set.difference(
            set(ev_sessions.loc[sig_arrival].index),
            set(self.ev_log.logging.index)))
        new_arrivals = ev_sessions.loc[new_arrival_idx].copy()
        self.ev_log.update(t, new_arrivals=new_arrivals)    # FIXME: maybe only some of the columns?

    def get_ev_params(self, t, ev_pred):

        ev_params = dict()
        
        if self.data["ev_sessions"] is None:
            ev_params["ev_I"] = 0
            for key in ["ev_ta", "ev_td", "ev_e_init", "ev_targ"]:
                ev_params[key] = np.array([])
            return ev_params
       
        # concat onsite and ev_pred
        # Note:
        # 1. for onsite EV, ta=0
        # 2. for pred EV: "e" = "e_init"
        onsite = self.ev_log.onsite_table.copy()
        onsite["ta"] = t # ! why do we need this line
        if ev_pred is not None:
            ev_pred["e"] = ev_pred["e_init"]
            ev_concat = pd.concat([onsite, ev_pred], axis=0)
        else:
            ev_concat = onsite

        td2int = lambda x: (x-t).total_seconds()/(3600*self.delta_0)
        ev_params["ev_ta"] = ev_concat["ta"].apply(td2int).values
        ev_params["ev_td"] = ev_concat["td"].apply(td2int).values
        # FIXME: now td (1) can be float; (2) td-ta may > K; (3) td may < ta 
        # check if params_reg handles all of these
        ev_params["ev_e_init"] = ev_concat["e"].values
        ev_params["ev_e_targ"] = ev_concat["e_targ"].values
        ev_params["ev_p_max"] = ev_concat["Pmax"].values

        ev_params["ev_I"] = len(ev_params["ev_ta"])
        
        # [Yi, 2023/03/19] FIXME
        self.cache["ev_index"] = list(ev_concat.index)
        
        return ev_params
        
        #   2.1 get onsite table
        onsite = self.ev_log.onsite_table
        if len(onsite) == 0:
            self.opt_log.loc[t] = {"N":0, "p":0, "c":ev_params["c"][0]}
            return
        else:
            onsite_idx = onsite.index
            ev_params["ta"] = np.array([0]*len(onsite))
            td2int = lambda x: (x-t).total_seconds()/(3600*delta_0)
            # FIXME: td = np.vectorize(td2int)(onsite["td"].values)
            if self.predictor.name == "Ground Truth" and self.predictor.td_actual == True:
                # surprisingly know the actual departure time
                td = self.source_data.loc[onsite.index, "td_actual"].apply(td2int).values
            else:
                td = onsite["td"].apply(td2int).values
            # td <= t, but td_actual > t
            ev_params["td"] = np.maximum(1, td).astype(int)
            ev_params["e_init"] = onsite["e"].values
            ev_params["e_init_0"] = onsite["e_init"].values
            ev_params["e_targ"] = onsite["e_targ"].values
            ev_params["e_targ_0"] = onsite["e_targ"].values
            ev_params["N_w"] = np.array([np.inf]* len(onsite))
            ev_params["Pmax"] = onsite["Pmax"].values
            ev_params["X_a"] = onsite["X_C"].values

            #   2.2 get predictors
            # ev_pred: keys: ["ta", "td", "e_init", "e_targ", "N_w", "Pmax", "X_a"]
            t_end = onsite["td"].max() if t_cut == True else None
            ev_pred = self.predictor.get_prediction(t, tend=t_end)
            
            #   2.3 combine to a profile
            if ev_pred is not None:
                for key in ev_params.keys():
                    if key != "c":
                        ev_params[key] = np.hstack([ev_params[key], ev_pred[key]])
            ev_params["I"] = len(ev_params["ta"])
            # FIXME: it may raise infeasible because
            #   p_tilde is not logged
            #   to deal with this, we manually check the feasibility
            ev_params["e_targ"] = np.maximum(ev_params["e_init"],
                        np.minimum(ev_params["e_targ"],
                        ev_params["e_init"]+\
                            (ev_params["td"]-ev_params["ta"])*\
                            ev_params["Pmax"]*params["eta"]*self.delta_0))
        
    def update_ev_charge(self, t, sol, curr_k=0, exe_K=1):
        
        # todo: the update index relationship of none_GT prediction havent been checked yet
        # t: use the original t, calculate here
        t_curr = t+timedelta(hours=self.delta_0*curr_k)
        # ! update_onsite_ev before updating charge
        self.update_onsite_ev(t_curr)
        
        onsite = self.ev_log.onsite_table
        redundancy=1.001
        unseen_ev_idx=[]
        unseen_ev_I=0
        
        if curr_k == 0:    
            p = list(sol["ev_p"][0][range(len(onsite)),0])
            self.cache["ev_p_sol"] = sol["ev_p"][0]
            # [Yi, 2023/03/19] FIXME
            # self.cache["ev_index"] = list(onsite.index)
        else:
            # [Lunlong, 2023/08/07 TODO:check if sol_ev was uodated ]
            ev_p_sol = self.cache["ev_p_sol"]
            ev_index = self.cache["ev_index"]
            p = []
            
            #//set_diff=set(onsite.index)-set(ev_index)
            #//if len(set_diff)>0:
            #//    print("set_diff:",set_diff)
            
            #unseen_ev_p=0
            for idx in onsite.index:
                if idx in ev_index: # if EV idx' power has been optimized
                    # FIXME
                    # [Yi, 2023/02/10] correct typo:
                    #   ev_p_sol is np.ndarray, not dataframe, thus
                    #   ev_p_sol.loc[...] -> ev_p_sol[...]
                    # // print("ev_index.index(idx):",ev_index.index(idx))
                    p.append(ev_p_sol[ev_index.index(idx), curr_k])
                else:   
                    # [Yi, 2023/03/19]: if optimizer has never seen EV idx 
                    #   rule = op_params["ev_charg_rule_default"] 
                    #   - charge with max ("asap") / unif power ("unif") 
                    # [Yi, 2023/02/10]: fix a bug here
                    #   charge at Pmax, but not exceed e_targ
                    p_max = onsite.loc[idx, "Pmax"]
                    e_req = onsite.loc[idx, "e_targ"] - onsite.loc[idx, "e"]
                    ta = onsite.loc[idx, "ta"]
                    td = onsite.loc[idx, "td"]
                    efficacy = self.op_params["ev_efficacy"]
                    unseen_ev_idx.append(idx)
                    unseen_ev_I+=1
                    if self.op_params["ev_charge_rule_default"] == "unif":
                        p_req = e_req / efficacy / self.delta_0
                    elif self.op_params["ev_charge_rule_default"] == "asap":
                        p_req = e_req / efficacy / self.delta_0
                        
                    elif self.op_params["ev_charge_rule_default"] == "alap":
                        # charge as less as possible before solving CFTOC the next time
                        # time of next optimization
                        t_next_sol=t_curr+timedelta(hours=(exe_K-curr_k+1)*self.delta_0)
                        # charging period left for next optimization 
                        t_future_sol = (td-t_next_sol).total_seconds()/3600
                        t_future_sol = max(0, t_future_sol) # may <0, if td is within current scope
                        # if e_req can be satisfied in the next optimization, omit it
                        if t_future_sol*efficacy*p_max > e_req * redundancy: 
                            p_req=0
                        # else charge the minimum but avoid infeasibility
                        else:
                            # ! Whether the executor know EV right after coming? 
                            # time left before next optimization
                            t_before_next_sol=min((t_next_sol-ta).total_seconds()/3600,\
                                (t_next_sol-t_curr).total_seconds()/3600)
                            # time left before EV departure
                            t_before_td=min((td-ta).total_seconds()/3600,\
                                (td-t_curr).total_seconds()/3600)
                            
                            t_available=min(t_before_next_sol,t_before_td)
                            p_req = (e_req-(t_future_sol*efficacy*p_max)) \
                                / t_available / efficacy
                        
                        
                    p.append(min(p_max, p_req))
        charge_log = dict(zip(onsite.index, p))    
        # [Yi, 2023/03/19] t -> t_curr
        self.ev_log.update(t_curr, charge_log=charge_log, delta=self.delta_0,
                                 eta=self.op_params["ev_efficacy"])
        # // self.update_onsite_ev(t_curr)
        self.update_onsite_ev(t_curr)
        # [Yi, 2023/02/11] return np.sum(p) -> return p
        #unseen_ev_I=len(unseen_ev_idx)
        return p, unseen_ev_I

    def op_summary(self, op_log):
        
        sol = op_log

        K, delta = self.op_params["K"], self.delta_0
        days = len(sol) * delta / 24
        days = 1 if days == 0 else days

        index = ["unit", "All"]
        columns = {
                "days": "1",
                "TCO": "$/day",    # "All" only
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
                #  "prob": "1", 
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
                 "bat_e_terminal": "kWh",
                 "bat_e_terminal_revenue": "$",
                 "load_bld_error_max_neg": "kW",
                 "load_bld_error_max_pos": "kW",
                 "load_pv_error_max_neg": "kW",
                 "load_pv_error_max_pos": "kW",
                 "load_ev_error_max_neg": "kW",
                 "load_ev_error_max_pos": "kW",
                 "load_net_error_max_neg": "kW",
                 "load_net_error_max_pos": "kW",
                 }

        df = pd.DataFrame(index=index, columns=columns.keys())
        for k in columns.keys():
            df.loc["unit", k] = columns[k]
        
        df.loc["All", "days"] = days
        df.loc["All", "bat_capacity"] = self.battery.get_params("bat_capacity")

        load_bld = sol["load_bld"].mean() * 24   
        load_pv = sol["load_pv"].mean() * 24  
        load_ev = sol["ev_p"].mean() * 24
        load_tot = load_bld + load_ev
        ev_I = len(self.ev_log.logging) / days  # FIXME: may not be a good way
        
        

        df.loc["All", "ev_I"] = ev_I
        df.loc["All", "load_bld"] = load_bld
        df.loc["All", "load_ev"] = load_ev
        df.loc["All", "load_tot"] = load_tot
        df.loc["All", "load_pv"] = load_pv

        p_grid = sol["p_grid"]
        grid_import = np.maximum(p_grid, 0)
        grid_export = np.maximum(-p_grid, 0)
        tou_import = sol["tou_import"]
        tou_export = sol["tou_export"]
        import_cost = (grid_import * tou_import).mean() * 24
        export_revenue = (grid_export * tou_export).mean() * 24
        tou_cost = import_cost - export_revenue
        #[Lunlong. 2023/08/18] add bat_e_tou
        bat_e_revenue = sol["bat_e"][-1]*sol["tou_import"].mean()/days

        df.loc["All", "grid_import"] = grid_import.mean() * 24
        df.loc["All", "grid_export"] = grid_export.mean() * 24
        df.loc["All", "import_cost"] = import_cost
        df.loc["All", "export_revenue"] = export_revenue
        df.loc["All", "tou_cost"] = tou_cost
        df.loc["All", "bat_e_terminal"] = sol["bat_e"][-1]
        df.loc["All", "bat_e_terminal_revenue"] = bat_e_revenue
        
        df.loc["All", "grid_max"] = grid_import.max()

        mo = p_grid.index.year.astype(str) + "-" + p_grid.index.month.astype(str)
        gb = p_grid.groupby(mo)
        dc_days = p_grid.groupby(mo).size().values*0.25/24
        p_dc = gb.agg(max).values
        demand_charge = (p_dc * dc_days * self.op_params["dc_price"]).sum() / dc_days.sum()
        opex = tou_cost + demand_charge - bat_e_revenue
        eq_rate_opex = opex / load_tot

        df.loc["All", "demand_charge"] = demand_charge
        df.loc["All", "OPEX"] = opex
        df.loc["All", "eq_rate_opex"] = eq_rate_opex


        bat_e = sol["bat_e"].values
        bat_p = sol["bat_p"].values
        eta = self.battery.get_params("bat_efficacy")
        e_0 = bat_e[0] + max(bat_p[0],0) / eta + min(bat_p[0], 0) * eta 

        bat_capacity = self.battery.get_params("bat_capacity")

        if bat_capacity != 0:

            # battery = self.battery.copy_params(capacity=True, deg_model_only=True)
            # battery_est = self.battery.copy_params(capacity=True, deg_model_only=True)
            # battery_est.set_params(deg_model = self.op_params["deg_model_opt"])

            # battery.update_soc(p = bat_p, e_init = e_0)
            eq_cycles = self.battery.get_states(key = "cycles_equiv") / days
            # battery_est.update_soc(p = bat_p, e_init = e_0)
            eq_cycles_est = self.battery_est.get_states(key = "cycles_equiv") / days
            df.loc["All", "eq_cycles"] = eq_cycles
            df.loc["All", "eq_cycles_est"] = eq_cycles_est

        else:
            df.loc["All", "eq_cycles"] = 0
            df.loc["All", "eq_cycles_est"] = 0
     
        
        df.loc["All", "eq_bat_capacity"] = bat_capacity / df.loc["All", "load_bld"] * 24    # kWh / (kWh/day) * (hr/day) -> hr


        eq_cycles = df.loc["All", "eq_cycles"]
        eq_cycles_est = df.loc["All", "eq_cycles_est"]
        bat_params = self.battery.get_params()
        bat_cycle_price = bat_capacity * bat_params["bat_price"] / bat_params["bat_cycle_0"]
        bat_day_price = bat_capacity * bat_params["bat_price"] / bat_params["bat_life_0"]

        df.loc["All", "CAPEX"] = max(eq_cycles*bat_cycle_price, bat_day_price)
        df.loc["All", "CAPEX_est"] = max(eq_cycles_est*bat_cycle_price, bat_day_price)

        df.loc["All", "TCO"] = df.loc["All", "CAPEX"] + df.loc["All", "OPEX"]
        df.loc["All", "TCO_est"] = df.loc["All", "CAPEX_est"] + df.loc["All", "OPEX"]

        df.loc["All", "eq_rate"] = df.loc["All", "TCO"] / df.loc["All", "load_tot"]
        df.loc["All", "eq_rate_est"] = df.loc["All", "TCO_est"] / df.loc["All", "load_tot"]
        
        df.loc["All","load_bld_error_max_neg"]= self.latest_max_bld_error_neg
        df.loc["All","load_bld_error_max_pos"]= self.latest_max_bld_error_pos
        df.loc["All","load_pv_error_max_neg"]= self.latest_max_pv_error_neg
        df.loc["All","load_pv_error_max_pos"]= self.latest_max_pv_error_pos
        df.loc["All","load_ev_error_max_neg"]= self.latest_max_ev_error_neg
        df.loc["All","load_ev_error_max_pos"]= self.latest_max_ev_error_pos
        df.loc["All","load_net_error_max_neg"]= self.latest_max_net_error_neg
        df.loc["All","load_net_error_max_pos"]= self.latest_max_net_error_pos
        
        self.summary = df.T
        return df
        



