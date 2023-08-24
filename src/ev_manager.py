import numpy as np
import pandas as pd

class EVmanager():

    """"
    1. maintain an onsite EV table [self.onsite_table], with columns:
    "id", "ta", "td", "e_init", "e_targ", "Pmax", "e", [optional] "X_C", "N_w"
    2. and also a comprehensive log [self.logging], with extra columns:
    "td_actual", "p_history"
    """


    def __init__(self):

        cols = ["ta", "td", "e_init", "e_targ", "Pmax", "e", "X_C", "N_w", "td_actual", "p_history"]

        self.onsite_table = pd.DataFrame(columns = cols[:-2])
        self.logging = pd.DataFrame(columns = cols)

    def update(self, timestamp, delta=1, eta=1,
                new_arrivals=None, charge_log=None, new_depart=None, X_C=None):
        
        # new_arrivals: a pd.DataFrame as onsite table
        # charge_log: a dict of all onsite EVs
        # new_depart: a list of newly departed EVs
        #   (it may differ from "td" in the table - drivers may leave earlier or later)
        # X_C: charger type - can only update for new arrival EVs
        # need to update at every time step
        
        # at every time step
        # first update all onsite EV's soc, i.e., x(t) = x(t-1) + u(t) 
        # then, remove EVs depart at time t, and add EVs arrive at time t

        onsite_idx = self.onsite_table.index
        # update 08/03: won't update p_history if charge_log is None
        # so that this method can be call multiple times in one time step
        if charge_log is not None:
            for idx in onsite_idx:
                p = charge_log[idx] if idx in charge_log else 0
                e_prev = self.onsite_table.loc[idx, "e"]
                e_curr = e_prev + p * eta * delta
                self.onsite_table.loc[idx, "e"] = e_curr
                self.logging.loc[idx, "e"] = e_curr
                if self.logging.loc[idx, "p_history"] is np.nan:
                    self.logging.loc[idx, "p_history"] = [p]
                else:
                    self.logging.loc[idx, "p_history"].append(p)  
        
        if X_C is not None:
            for idx in X_C.keys():
                self.onsite_table.loc[idx, "X_C"] = X_C[idx]
                self.logging.loc[idx, "X_C"] = X_C[idx]
                
        if new_arrivals is not None:
            self.onsite_table = pd.concat([self.onsite_table, new_arrivals])
            self.logging = pd.concat([self.logging, new_arrivals])
            arrival_idx = new_arrivals.index
            self.onsite_table.loc[arrival_idx, "e"] = self.onsite_table.loc[arrival_idx, "e_init"].values.copy()
            self.logging.loc[arrival_idx, "e"] = self.onsite_table.loc[arrival_idx, "e"].values.copy()
        
        if new_depart is not None:
            self.onsite_table.drop(index = new_depart, inplace=True)
            self.logging.loc[new_depart, "td_actual"] = timestamp   
        
        
