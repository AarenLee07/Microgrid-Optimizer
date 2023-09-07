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
#log_folder=r'L:\Coding_project\Energy_grid_new_exp_local'
log_folder=r'D:\Codes\Energy_grid_new_exp_local'


exp_suffix = "thesis_track_p_grid"
exp_folder = os.path.join(out_path, "experiments", exp_suffix)
debug_folder = os.path.join(out_path, "debug_test")
assert os.path.exists(exp_folder)
log_fn = os.path.join(exp_folder, "6h-bat-Oct-track-p-grid.xlsx")#_oneday_12months

save_path = os.path.join(log_folder,exp_suffix, "6h-bat-Oct-track-p-grid")
if not os.path.exists(save_path):
    os.makedirs(save_path)
assert os.path.exists(save_path)

def parallel(fork_id):
    class MPC_parallel(exp.MPC_ExperimentManager):
        None
    for i in range(15):
        try:
            em = MPC_parallel(log_fn=log_fn, save_path=save_path, save=True, exp_prefix="MPC")
            var_keys = [
                        "method", "strategy",
                        "pred_model", "deg_model_opt", "p_grid_max", 
                        "price_dc", "price_sell", 
                        "ev_charge_rule", 
                        "B_kWh",  "deg_model", 
                        "start", "end", 
                        "bld", "ev", "pv",
                        #"disturbance_rule", "disturbance_scale","shift","shift_ratio"
                        "p_grid_max_method",]
            em.run(keys=var_keys, num_trials=10, fork_id=fork_id)
        except Exception as e:
            print('Reason_out:', e)  
            print('Failed in '+str(i)+'nd trial')
            continue
   
if __name__ == '__main__':     
    thread_list= list()
    freeze_support()
    for i in range(2):
        
        t=Process(target=parallel,args=(str(i+1),)) #创建线程
        thread_list.append(t)
        t.start()  #启动线程
        sleep(20)
        
    for t in thread_list:
        t.join()