import os
import sys

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

"""
============================= UPDATES =============================
(write down updates on the code - the last updated one at the first)
===================================================================
# 2023/02/11, Yi


"""


class DataPool():
    
    def __init__(self, data, update_rate=1):
        self.data = data    # keys: load_pv, load_bld, ev_sessions
        self._reset_cache()
        self.update_rate = update_rate # avoid frequently concat
    
    # [Yi, 2023/02/16] add a get_data method
    def get_data(self):
        copy_none = lambda x: x if x is None else x.copy()
        data = {
            "load_bld": copy_none(self.data["load_bld"]),
            "load_pv": copy_none(self.data["load_pv"]),
            "ev_sessions": copy_none(self.data["ev_sessions"])
        }
        return data

    def update(self, new_data):
        for k in new_data.keys():   # keys: load_pv, load_bld, ev_sessions
            self.cache[k].append(new_data[k])
        self.cache_len += 1
        if self.cache_len >= self.update_rate:
            for k in self.cache.keys():
                self.data[k] = pd.concat([self.data[k]]+self.cache[k], axis=0)
            self._reset_cache()
    
    def _reset_cache(self):
        self.cache = {k: [] for k in self.data.keys()}
        self.cache_len = 0
