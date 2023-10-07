import os
import sys
import exp
from multiprocessing import Process, freeze_support
from threading import Thread
from time import sleep
src_path = sys.path[0].replace("exp_notebook", "src")
#replace notebook as scripts
data_path = sys.path[0].replace("exp_notebook", "data")
if src_path not in sys.path:
    sys.path.append(src_path)
out_path = sys.path[0].replace("exp_notebook", "output")
log_folder=r'L:\Coding_project\Energy_grid_new_exp_local'
#log_folder=r'D:\Codes\Energy_grid_new_exp_local'


#L:\Coding_project\Energy_grid_new\output\experiments\thesis_track_p_grid\6h-bat-Oct-track-p-grid-2.xlsx
exp_suffix = "simple_debug"
exp_folder = os.path.join(out_path, "experiments", exp_suffix)
debug_folder = os.path.join(out_path, "debug_test")
assert os.path.exists(exp_folder)

save_path = os.path.join(log_folder,exp_suffix, "pred_bld_pv_ev_new_ev_pred_new_sol_exe")
if not os.path.exists(save_path):
    os.makedirs(save_path)
assert os.path.exists(save_path)


class MPC_parallel(exp.MPC_ExperimentManager):
        None

log_fn_= os.path.join(exp_folder, "pred_bld_pv_ev_new_ev_pred_new_sol_exe.xlsx")#_oneday_12months
em = MPC_parallel(log_fn=log_fn_, save_path=save_path, save=True, exp_prefix="MPC")
var_keys = [
            "method", "strategy",
            "pred_model", "deg_model_opt", "p_grid_max", 
            "price_dc", "price_sell", 
            "ev_charge_rule", 
            "B_kWh",  "deg_model", 
            "start", "end", 
            "bld", "ev", "pv",
            #"shift","shift_ratio",#"disturbance_rule", "disturbance_scale", #
            "simple_bld","simple_pv","simple_ev",
            "month_of_year",
            "run_bat_as_sol",
            "simple_pv_num", "simple_pv_exp_alpha",
            #"pv_to_bld","ev_to_bld",
            "p_grid_max_method",]

em.run(keys=var_keys, num_trials=1, fork_id=0)

