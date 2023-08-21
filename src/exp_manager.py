import os, sys
import pandas as pd
from datetime import datetime
import time
import win32file
from time import sleep,ctime

def is_used(file_name):
	try:
		vHandle = win32file.CreateFile(file_name, win32file.GENERIC_READ, 0, None, win32file.OPEN_EXISTING, win32file.FILE_ATTRIBUTE_NORMAL, None)
		return int(vHandle) == win32file.INVALID_HANDLE_VALUE
	except:
		return True
	finally:
		try:
			win32file.CloseHandle(vHandle)
		except:
			pass


class ExperimentManager():
    def __init__(self, log_fn, save_path, exp_prefix=None, 
                 save=True, cover=True, **default_params):
        
        self.save = save
        assert os.path.exists(log_fn)
        if self.save:
            assert os.path.exists(save_path)
        self.log_fn = log_fn
        self.save_path = save_path
        
        # TODO: also save [default_params] in the [log], 
        #   to make experiement reproduciable given the log file
        self.init_default_params()
        self.default_params.update(default_params)
        
        self.exp_prefix = exp_prefix if exp_prefix is not None else ""
        self.cover = cover  # whetehr to cover existing file with same filename

    def run(self, keys, num_trials=1):
        
        for _ in range(num_trials):
            
            clock = time.perf_counter()
            
            # STEP 1: find the next experiment to run
            # [Lunlong 2023/08/01] Modified for paralel threading, avoid accessing contradiction
            tw=0
            while tw<8:
                if is_used(self.log_fn):
                    if(tw%1==0):
                        print("Access denied, suspending.")
                    sleep(0.1)
                    tw=tw+0.1
                else:
                    log = pd.read_excel(self.log_fn, index_col=0)
                    break
            #log = pd.read_excel(self.log_fn, index_col=0)
            log_to_run = log.loc[log["status"]==0]

            if len(log_to_run) == 0:
                break


            trial_idx = log_to_run.index[0]
            
            params = dict(log.loc[trial_idx, keys])

            for k in params:
                v = params[k]
                if v in ["none", "None", "NONE"]:
                    params[k] = None
                if v in ["True", "TRUE", "true"]:
                    params[k] = True
                if v in ["False", "FALSE", "false"]:
                    params[k] = False

            save_fn = self.save_filename_gen(params)
            #  save_fn = 
            
            log.loc[trial_idx, "status"] = "R"
            if self.save:
                log.loc[trial_idx, "save_fn"] = save_fn
            # [Lunlong 2023/08/01] Modified for paralel threading, avoid accessing contradiction
            tw=0
            while tw<8:
                if is_used(self.log_fn):
                    if(tw%1==0):
                        print("Access denied, suspending.")
                    sleep(0.1)
                    tw=tw+0.1
                else:
                    log.to_excel(self.log_fn, index=True)
                    break

            # STEP 2: run experiment with corresponding params
            #   [run_one_trial] method will be highly case-specific, need to override
            #   save results also need to implement inside 
            
            stats = self.run_one_trial(params, save_fn)

            # STEP 3: record trial stats

            # [Lunlong 2023/08/01] Modified for paralel threading, avoid accessing contradiction
            tw=0
            while tw<8:
                if is_used(self.log_fn):
                    if(tw%1==0):
                        print("Access denied, suspending.")
                    sleep(0.1)
                    tw=tw+0.1
                else:
                    log = pd.read_excel(self.log_fn, index_col=0)
                    break

            #  log = pd.read_excel(self.log_fn, index_col=0)

            log.loc[trial_idx, "status"] = "D"
            log.loc[trial_idx, "runtime"] = time.perf_counter() - clock
            for k in stats.keys():
                if k not in log.columns:
                    log[k] = ""
                log.loc[trial_idx, k] = stats[k]
            # [Lunlong 2023/08/01] Modified for paralel threading, avoid accessing contradiction
            tw=0
            while tw<8:
                if is_used(self.log_fn):
                    if(tw%1==0):
                        print("Access denied, suspending.")
                    sleep(0.1)
                    tw=tw+0.1
                else:
                    log.to_excel(self.log_fn, index=True)
                    break
           
            #log.to_excel(self.log_fn, index=True)
            print(f"Done, trial {trial_idx}")

        print("="*20)
        print("DONE")

        
    def run_one_trial(self, params):
        """  OVERRIDE this for the specific experiment  """
        stats = {}
        return stats
    
    def retrieve(self, **kws):
        pass

    def save_filename_gen(self, params):
        
        fn = self.exp_prefix
        for k in params.keys():
            fn += "-{}".format(params[k])

        '''
        for k in params.keys():
            fn += "_{}-{}".format(k, params[k])
        if self.cover:
            return fn+".xlsx"
        '''
        # if choosing not to cover existing result files:
        #   add a unique suffix
        exist_fns = os.listdir(self.save_path)
        i = 1
        while True:
            suffix = "{}_{:03d}".format(datetime.today().date(), i)
            save_fn = "{}-{}.xlsx".format(fn, suffix)
            if save_fn not in exist_fns:
                break        
            else:
                i=i+1
        return save_fn
    
    def email_notification(self):
        pass
    
    def init_default_params(self):
        """  OVERRIDE this for the specific experiment  """
        self.default_params = {}