import numpy as np
import warnings
# solving [rainblow] counts uses optimization solvers
# you may use [cvxpy] instead
import gurobipy as gp
from gurobipy import GRB

"""
============================= UPDATES =============================
(write down updates on the code - the last updated one at the first)
===================================================================

# [Yi, 2023/03/15]
- change default params: 
    - bat_price: 1000 -> 100 $ / kWh
    - bat_cycle_0: 10000 -> 5000
- [*] add params related to degradation model
    [make it clear] "deg_model" means what is the "actual" degradation mechanism
                    however, equiv cycles with all avail models will be calculated and stored, for comparison purpose
    "deg_model": valid values: "unconscious", "throughput", "Crate", "rainflow", "DOD"
    e.g., for power degradation model ("C-rate"):
    - "deg_C-rate_thres": (0.25, 0.25, 0.25, 0.25)
    - "deg_C-rate_lambda": (0.8, 1, 1.5, 2)
    where △P_j = (bat_capacity / bat_p_max) [kW] * deg_power_thres[j]
- TODO: add private method [params_sanity_check] to check model params
- [*] modify the [equiv_cycles] method

# [Yi, 2022/09/03]
1. [get_params] / [get_status] can specify the key(s) to get

"""

class Battery_base():

    name = "battery_base"

    def __init__(self, params=None, delta_0=0.25, deg_model_only=False, **kw):

        default_params = {
            "bat_capacity": None,
            "bat_p_max": 3, # i.e., capacity (kWh) / p_bat_max (kW) = 3 (h)
            "bat_p_min": 3, # can omit, then p_bat_min = p_bat_max
            "bat_price": 150, # $/kWh (old: 1000, ref: Tesla Powerwall)
            "bat_efficacy": 0.98, 
            "bat_life_0": 3650, # days.
            "bat_cycle_0": 3000, # cycles in lifetime
            # battery degradation params
            "deg_model": "throughput",  
                # valid values: "unconscious", "throughput", "Crate", "rainflow", "DOD"
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

        self.valid_deg_models = ["unconscious", "throughput", "Crate", "rainflow", "DOD"]
        # [Yi, 2023/03/19] if True, only eq_cycle of [deg_model] will be calculated
        self.deg_model_only = deg_model_only    

        # Note: only keys exist in default_params will be considered
        # properties will be replaced by the priority:
        #   **kw > params > default_params
        self.params = dict(default_params)
        if params is not None:
            self.set_params(**params)
        self.set_params(**kw)
        
        self.states = {
            "e_curr": None,
            "cycles_equiv": 0,
            "working_days": 0,
            "expected_lifetime": self.params["bat_life_0"],
            "expected_lifetime_remain": self.params["bat_life_0"],
            "expected_daily_cost": None
        }

        self.delta_0 = delta_0
        self.p_history = list()
        
        """ [Yi, 2023/03/15] 
            every [self.cycle_update_freq] steps, it will update the equiv cycles, by calling [self.equIv_cycle_cal]
            [equiv_cycle_cal] will calculate the equivalent cycles of every step in [ self.p_history[self.counted_t:] ]
            and store into [self.cycle_history]
        """
        self.cycle_history = {k: list() for k in self.valid_deg_models}
        self.cycle_update_freq = 24
        self.cycle_prev_state = {"t": 0, "e": None} # "e" is initialized at the first step of soc update
    

    """ the following methods are PUBLIC methods
        i.e., you can call them directly outside    
    """

    def update_soc(self, p=None, e=None, e_init=None, delta=None):

        if p is not None and e is not None:
            warnings.warn("when both \"p\" and \"e\" passed, only use \"p\" for update")
        if delta is None:
            delta = self.delta_0

        if p is not None:
            p = self.float2array(p)
            delta = self.float2array(delta, copy=len(p))
            if self.states["e_curr"] is None:
                self.states["e_curr"] = 0 if e_init is None else e_init
            if self.cycle_prev_state["e"] is None:
                self.cycle_prev_state["e"] = self.states["e_curr"]
            for k in range(len(p)):
                # FIXME: test this part - e.g., delta != delta_0
                for _ in range(round(delta[k]/self.delta_0)):
                    self.p_history.append(p[k])
                e_0 = self.states["e_curr"]
                e_1 = e_0 + self.soc_change_by_p(p[k], delta[k])
                # [Yi, 2023/02/10] It is wiered, but it does happen
                #   we separate p_bat = p_bat^+ - p_bat^- in optimization
                #   they should never be both positive
                #   however, it does happen (don't know why, maybe it is just a little bit sub-optimal)
                #   so, add the following line as a makeup for that situation
                e_1 = np.clip(e_1, 0, self.params["bat_capacity"])
                self.update_battery_states(e_0=e_0, e_1=e_1, delta=delta[k])
                
        elif e is not None: 
            # FIXME: here, I suppose "e" be an array, and e[0]=e_curr
            e = self.float2array(e)
            delta = self.float2array(delta, copy=len(e))
            if self.states["e_curr"] is None:
                self.states["e_curr"] = e[0]
            if self.cycle_prev_state["e"] is None:
                self.cycle_prev_state["e"] = self.states["e_curr"]
            assert abs(e[0] - self.states["e_curr"]) <= 0.001
            for k in range(1, len(e)):
                e_0, e_1 = e[k-1], e[k]
                p_k = self.p_by_soc_change(e_0, e_1, delta[k])
                # FIXME: test this part - e.g., delta != delta_0
                for _ in range(round(delta[k]/self.delta_0)):
                    self.p_history.append(p_k)
                self.update_battery_states(e_0=e_0, e_1=e_1, delta=delta[k])
  
    def set_capacity(self, capacity):
        self.params["bat_capacity"] = capacity
    
    def set_params(self, **kw):
        for key, value in kw.items():
            if key in self.params.keys():
                self.params[key] = value
            elif key in self.states.keys():
                self.states[key] = value
            elif key == "delta_0":
                self.delta_0 = value
            else:
                warnings.warn("\"{}\" is not a battery parameter / state".format(key))

        # [Yi, 2023/03/15] call sanity check
        self.params_sanity_check()
    
    def get_params(self, key=None):
        # update 2022/09/03
        if key is None:
            return self.params
        if isinstance(key, list):
            return {k: self.params[k] for k in key}
        return self.params[key]
            
    def get_states(self, key=None):
        # update 2022/09/03
        if key is None:
            return self.states
        if isinstance(key, list):
            return {k: self.states[k] for k in key}
        return self.states[key]

    def copy_params(self, states=False, capacity=False, **kw):
        b1 = Battery_base(**kw)
        b1.set_params(**self.params)
        b1.set_params(delta_0 = self.delta_0)
        if capacity == False:
            b1.set_capacity(None)
        if states == True:
            b1.set_params(**self.states)
        
        return b1

    def save_records(self, fn):
        # TODO: save params & records to a file (recommend format: pkl)
        ...
    
    def recover_records(self, fn):
        # TODO: recover params & records from a file
        ...

    def should_renew(self):
        return self.states["expected_lifetime_remain"] <= 0


    """ the following methods should be treated as PRIVATE methods
        i.e., don't call them directly outside 
        
        I'd suggest also treating all attributes as PRIVATE, 
        but you can get them by calling "get_params()", etc.   """

    def params_sanity_check(self):
        """ [Yi, 2023/03/15] check all params are in the valid formats """
        params = self.params

        deg_model = params.get("deg_model")
        if deg_model not in self.valid_deg_models:
            raise Exception(f"Key [deg_model] only accept {self.valid_deg_model}. Not [{deg_model}].")

    def update_battery_states(self, e_0, e_1, delta):
        self.states["e_curr"] = e_1
        self.states["working_days"] += delta/24

        if len(self.p_history) - self.cycle_prev_state["t"] >= self.cycle_update_freq:
            cycles = self.equiv_cycles_cal()
            self.states["cycles_equiv"] += sum(cycles)
        
        self.update_expected_lifetime()

    def equiv_cycles_cal(self):
        """
        Equivalent cycles will be calculated based on self.p_history

        """
        t_prev = self.cycle_prev_state["t"]
        e_prev = self.cycle_prev_state["e"]
        ps = self.p_history[t_prev:]

        cal_methods = {
            "unconscious": self.equiv_cycles_unconscious_cal,
            "throughput": self.equiv_cycles_throughput_cal,
            "Crate": self.equiv_cycles_Crate_cal,
            "rainflow": self.equiv_cycles_rainflow_cal,
            "DOD": self.equiv_cycles_DOD_cal
        }

        if self.deg_model_only:
            m = self.params["deg_model"]
            cycles_deg = cal_methods[m](e_prev, ps)
            self.cycle_history[m] += cycles_deg
        else:
            for m in self.valid_deg_models:
                cycles = cal_methods[m](e_prev, ps)
                self.cycle_history[m] += cycles
                if m == self.params["deg_model"]:
                    cycles_deg = cycles


        self.cycle_prev_state["t"] = len(self.p_history)
        self.cycle_prev_state["e"] = self.states["e_curr"]
        return cycles_deg

    def equiv_cycles_unconscious_cal(self, e_prev, ps):
        return [0] * len(ps)

    def equiv_cycles_throughput_cal(self, e_prev, ps):
        bat_capacity = self.params["bat_capacity"]
        if bat_capacity == 0:
            return [0] * len(ps)
        eq_cycles = np.abs(np.array(ps)) * (0.5 * self.delta_0 / bat_capacity)
        return eq_cycles.tolist()

    def equiv_cycles_Crate_cal(self, e_prev, ps):
        bat_capacity = self.params["bat_capacity"]
        if bat_capacity == 0:
            return [0] * len(ps)
        p_max = bat_capacity / self.params["bat_p_max"]
        
        thres = np.array(self.params["deg_Crate_thres"]) * p_max    # shape = (J,)
        lams = np.array(self.params["deg_Crate_lambda"])
        thres_cum = thres.cumsum() - thres

        p_js = np.clip(np.abs(ps) - thres_cum[:,None], 0, thres[:,None]) # shape = (J, N)
        eq_cycles = (p_js * lams[:,None]).sum(axis=0) * (0.5 * self.delta_0 / bat_capacity)
        return eq_cycles.tolist()

    def equiv_cycles_rainflow_cal(self, e_prev, ps):
        # FIXME
        bat_capacity = self.params["bat_capacity"]
        if bat_capacity == 0:
            return [0] * len(ps)
        bat_efficacy = self.params["bat_efficacy"]
        delta = self.delta_0
        thres = np.array(self.params["deg_rainflow_thres"])    # shape = (J,)
        lams = np.array(self.params["deg_rainflow_lambda"])

        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.start()
            m = gp.Model("rainflow", env=env)
        
        J = len(thres)
        K = len(ps)

        ps = np.array(ps)
        p_bat_pos, p_bat_neg = np.maximum(ps, 0), np.maximum(-ps, 0)

        eq_cycles = m.addVars(K, name="eq_cycle")    # note: different from that in [battery_optimizer], we keep eq_cycle for every step
        p_bat_pos_js = m.addVars(K, J)
        p_bat_neg_js = m.addVars(K, J)
        e_bat_js = m.addVars(K+1, J, name="bat_e_js")
        
        # [Constr] initialize e_bat_js
        m.addConstrs((e_bat_js[0,j] == thres[j] * e_prev for j in range(J)))

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
        m.addConstrs((eq_cycles[k] >= 0.5 / bat_capacity * delta * sum((p_bat_pos_js[k,j]+p_bat_neg_js[k,j])*lams[j] 
                                              for j in range(J)) for k in range(K)))
        
        # [Obj] minimize the total eq_cycles
        m.setObjective(sum(eq_cycles[k] for k in range(K)), sense=GRB.MINIMIZE)
        
        # solve model
        m.optimize()

        # bat_e_js_res = [[m.getVarByName(f"bat_e_js[{k},{j}]").x for j in range(J)] for k in range(K)]
        # bat_e_js_res = np.array(bat_e_js_res)


        return [m.getVarByName(f"eq_cycle[{k}]").x for k in range(K)]


    def equiv_cycles_DOD_cal(self, e_prev, ps):
        bat_capacity = self.params["bat_capacity"]
        if bat_capacity == 0:
            return [0] * len(ps)
        eta = self.params["bat_efficacy"]
        ps = np.array(ps)
        ps_eta = np.maximum(ps, 0)/eta + np.minimum(ps, 0)*eta
        es = np.zeros(len(ps)+1)
        es[0] = e_prev
        es[1:] = e_prev - ps_eta.cumsum() * self.delta_0
        
        thres = np.array(self.params["deg_DOD_thres"]) * bat_capacity    # shape = (J,)
        lams = np.array(self.params["deg_DOD_lambda"])
        thres_cum = thres.cumsum() - thres

        e_js = np.clip(es - thres_cum[:,None], 0, thres[:,None]) # shape = (J, N+1)
        p_js_eta = (e_js[:,1:] - e_js[:,:-1]) / self.delta_0     # shape = (J, N)
        p_js = np.maximum(p_js_eta, 0)/eta + np.minimum(p_js_eta, 0)*eta
        eq_cycles = (np.abs(p_js) * lams[:,None]).sum(axis=0) * (0.5 * self.delta_0 / bat_capacity)
        return eq_cycles.tolist()
        


    def soc_change_by_p(self, p, delta):
        efficacy = self.params["bat_efficacy"]
        # pos: discharge / neg: charge
        return - delta * (max(p,0)/efficacy + min(p,0)*efficacy)
    
    def p_by_soc_change(self, e_0, e_1, delta):
        efficacy = self.params["bat_efficacy"]
        # pos: discharge / neg: charge
        p = - (e_1 - e_0) / delta
        return max(p,0)*efficacy + min(p,0)/efficacy
      
    def update_expected_lifetime(self):
        cycles_equiv = self.states["cycles_equiv"]
        working_days = self.states["working_days"]
        if cycles_equiv == 0:
            expected_lifetime = self.params["bat_life_0"]
        else:
            expected_cycle_days = self.params["bat_cycle_0"]/(cycles_equiv/working_days)
            expected_calendar_days = self.params["bat_life_0"]
            expected_lifetime = min(expected_cycle_days, expected_calendar_days)
        self.states["expected_lifetime"] = expected_lifetime
        self.states["expected_lifetime_remain"] = expected_lifetime - working_days

        bat_capacity = self.params["bat_capacity"]
        if bat_capacity is not None:
            self.states["expected_daily_cost"] =\
                bat_capacity * self.params["bat_price"] / expected_lifetime
        if self.should_renew():
            warnings.warn("The current battery should be renewed!")


    @staticmethod
    def float2array(x, copy=None):
        if isinstance(x, float) or isinstance(x, int):
            copy = 1 if copy is None else copy
            return np.array([x] * copy)
        assert isinstance(x, np.ndarray)
        return x