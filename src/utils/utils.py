import os
import sys
import numpy as np
import pandas as pd
import itertools

# fix: reference before statement : data_path
#data_path = sys.path[0].replace("notebooks", "data")

'''    
    data_path = sys.path[0].replace("notebooks", "data")
    try:
        os.path.exists(data_path)
    except:
        print("error: data_path not exist:{}".format(data_path))
        curr_path = sys.path[0]
        idx = curr_path.find("\\Energy_grid")
        if idx == -1:
            raise Exception("cannot locate current repository")
        data_path = os.path.join(curr_path[:idx+len("\\Energy_grid")], "data")
        
    finally:
        print("path get in the utils.py: ")
        print(data_path)
        return data_path'''


def get_data_path():

    try:
        os.path.exists(data_path)
    except:
        curr_path = sys.path[0]
        idx = curr_path.find("\\Microgrid-Optimizer")
        if idx == -1:
            raise Exception("cannot locate current repository")
        data_path = os.path.join(curr_path[:idx+len("\\Microgrid-Optimizer")], "data")
    finally:
        return data_path
    


def mean_with_prob(x, prob=None, axis=None):
    if prob is None:
        return x.mean(axis=axis)
    prob = prob / prob.sum()    # normalize prob
    if axis == 0:
        prob = prob[:,None]
    return (x * prob).sum(axis=axis)

def std_with_prob(x, prob=None, axis=None):
    if prob is None:
        return x.std(axis=axis)
    prob = prob / prob.sum()    # normalize prob
    mu = mean_with_prob(x, prob, axis=axis)
    if axis == 1:
        mu = mu[:,None]
    if axis == 0:
        prob = prob[:,None]
    return np.sqrt(((x - mu) ** 2 * prob).sum())

def percentile_with_prob(x, percentiles, prob=None):
    if prob is None:
        return np.percentile(x, percentiles)
    
    prob = prob / prob.sum()    # normalize prob

    order = np.argsort(x)
    x = x[order]    # order x from min to max
    prob = prob[order]
    prob_cum = np.cumsum(prob)

    pcts = []
    for q in percentiles:
        q = q/100
        idx = sum(prob_cum<=q)
        if idx == len(x)-1:
            pct = x[-1]
        else:
            q0 = prob_cum[idx-1]
            q1 = prob_cum[idx]
            pct = ((q1-q) * x[idx-1] + (q-q0) * x[idx]) / (q1 - q0)
        pcts.append(pct)
    return np.array(pcts)


def quick_stats(x, percentiles=(5,25,50,75,95), prob=None, key_prefix = None):
        # FIXME: current only support 1-D x

        assert len(x.shape) == 1

        key_prefix = "" if key_prefix is None else key_prefix + "_"
             
        stats = {}
        stats[key_prefix+"mu"] = mean_with_prob(x, prob)
        stats[key_prefix+"std"] = std_with_prob(x, prob)
        if percentiles is not None:
            pct = percentile_with_prob(x, percentiles, prob)
            for i in range(len(percentiles)):
                stats[key_prefix+"pct_{}".format(percentiles[i])] = pct[i]
        
        return stats
    
def generate_exp_table(params_dic,save_path):
    
    # Get all possible combinations
    combinations = list(itertools.product(*params_dic.values()))
    # Convert to DataFrame
    df = pd.DataFrame(combinations, columns=params_dic.keys())
    df.index.name='id'
    df.to_excel(save_path, index=True)
    
    return df
