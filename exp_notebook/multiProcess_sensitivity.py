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


#L:\Coding_project\Energy_grid_new\output\experiments\thesis_track_p_grid\6h-bat-Oct-track-p-grid-2.xlsx
exp_suffix = "simple_debug"
exp_folder = os.path.join(out_path, "experiments", exp_suffix)
debug_folder = os.path.join(out_path, "debug_test")
assert os.path.exists(exp_folder)

save_path = os.path.join(log_folder,exp_suffix, "trend_check")
if not os.path.exists(save_path):
    os.makedirs(save_path)
assert os.path.exists(save_path)

def parallel(fork_id,log_fn_):
    class MPC_parallel(exp.MPC_ExperimentManager):
        None
    for i in range(1):
        try:
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
                        #"simple_pv_num", "simple_pv_exp_alpha",
                        #"pv_to_bld","ev_to_bld",
                        "p_grid_max_method",]
            em.run(keys=var_keys, num_trials=1, fork_id=fork_id)
        except Exception as e:
            print('Reason_out:', e)  
            print('Failed in '+str(i)+'nd trial')
            continue
   
if __name__ == '__main__':     
    thread_list= list()
    freeze_support()

    
    log_fn = os.path.join(exp_folder, "pred_bld_pv_ev.xlsx")#_oneday_12months
    for i in range(4):
        t=Process(target=parallel,args=('A'+str(i+1),log_fn)) #创建线程
        thread_list.append(t)
        t.start()  #启动线程
        sleep(25)
    '''
    log_fn = os.path.join(exp_folder, "Simple_pv_gridS_2.xlsx")#_oneday_12months
    for i in range(15):
        t=Process(target=parallel,args=('B'+str(i+1),log_fn)) #创建线程
        thread_list.append(t)
        t.start()  #启动线程
        sleep(25)
        
    log_fn = os.path.join(exp_folder, "Simple_pv_gridS_3.xlsx")#_oneday_12months
    for i in range(15):
        t=Process(target=parallel,args=('C'+str(i+1),log_fn)) #创建线程
        thread_list.append(t)
        t.start()  #启动线程
        sleep(25)
        
    log_fn = os.path.join(exp_folder, "Simple_pv_gridS_base.xlsx")#_oneday_12months
    for i in range(2):
        t=Process(target=parallel,args=('D'+str(i+1),log_fn)) #创建线程
        thread_list.append(t)
        t.start()  #启动线程
        sleep(15)
    
    log_fn = os.path.join(exp_folder, "One_step_optimal.xlsx")#_oneday_12months
    for i in range(2):
        t=Process(target=parallel,args=('D'+str(i+1),log_fn)) #创建线程
        thread_list.append(t)
        t.start()  #启动线程
        sleep(12)
    
    log_fn = os.path.join(exp_folder, "pred_pv_alpha1.xlsx")#_oneday_12months
    for i in range(24):
        t=Process(target=parallel,args=('C'+str(i+1),log_fn)) #创建线程
        thread_list.append(t)
        t.start()  #启动线程
        sleep(15)
    
    log_fn = os.path.join(exp_folder, "Shift_Dec.xlsx")#_oneday_12months
    for i in range(12):
        t=Process(target=parallel,args=('D'+str(i+1),log_fn)) #创建线程
        thread_list.append(t)
        t.start()  #启动线程
        sleep(25)'''
        
    for t in thread_list:
        t.join()