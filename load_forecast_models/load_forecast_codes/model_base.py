import numpy as np
import optuna
import torch
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.preprocessing import MaxAbsScaler

from darts.dataprocessing.transformers import Scaler
from darts.datasets import AirPassengersDataset
from darts.metrics import smape
from darts.models import TCNModel
from darts.utils.likelihood_models import GaussianLikelihood

import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt

from darts.dataprocessing.transformers import Scaler
from darts.models import RNNModel
from darts.datasets import WeatherDataset
from darts.models import LinearRegressionModel
import darts.metrics as metrics
from darts.datasets import AirPassengersDataset
import datetime

import numpy as np
import optuna
import torch
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.preprocessing import MaxAbsScaler

from darts.dataprocessing.transformers import Scaler
from darts.datasets import AirPassengersDataset
from darts.models import TCNModel
from darts.utils.likelihood_models import GaussianLikelihood
from optuna.terminator import report_cross_validation_scores

from darts.models import TFTModel
from darts.utils.likelihood_models import QuantileRegression
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from darts import TimeSeries

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold






class Model_base():
    # set basic settings:
    # 1. model identification
    # 2. params of model, and whether call optuna
    # 3. prediction settings
    # 2. saving settings
    def __init__(self,id, model_type,
                 call_optuna, optuna_settings, optuna_params_dic,
                 call_one_trial, one_trial_settings, one_trial_params_dic,
                 load_from_trained_model,load_settings,
                 load_from_checkpoint,load_checkpoint_settings,
                 predict_settings,metric_settings, save_settings):
        # for optuna params_dic should be like:
        #   dic={
        #       'param_key':[tpye, lower_bound, upper_bound, bool_log], #params to search
        #       'param_key':[value] #params not to search
        #   }
        # check if necessary key pairs are passed
        
        for i in ['folder','save_model','save_prediction','save_metrics']:
            assert i in save_settings.keys()
            
        self.model_type=model_type
        self.model=None # tmp model for optuna / model load from file
        
        self.save_settings=save_settings
        self.save_path=os.path.join(save_settings['folder'],id)
        self.id=id
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        self.call_optuna=call_optuna
        if call_optuna==True:
            assert optuna_settings!=None
            assert optuna_params_dic!=None
        self.optuna_params_dic=optuna_params_dic
        self.optuna_settings=optuna_settings
        self.optuna_model_name=None
        
        self.call_one_trial=call_one_trial
        if call_one_trial==True:
            assert one_trial_settings!=None
            assert one_trial_params_dic!=None
        self.one_trial_params_dic=one_trial_params_dic
        self.one_trial_settings=one_trial_settings
        self.one_trial_model_name=None
        
        self.load_from_trained_model=load_from_trained_model
        if load_from_trained_model==True:
            assert load_settings!=None
            self.load_settings=load_settings
            self.load_from_model()
            
        self.load_from_checkpoint=load_from_checkpoint
        if load_from_checkpoint==True:
            assert load_checkpoint_settings!=None
            self.load_checkpoint_settings=load_checkpoint_settings
            self.load_from_checkpoints()
        
        
        self.predict_settings=predict_settings
        self.metric_settings=metric_settings
        

        
        self.optuna_study=None
        self.best_params=None
        self.best_model=None
        self.best_model_name=None
        
        self.one_trial_model=None # model from run_one_trial
        
        ...
    def load_from_model(self):
        self.model=self.model_type.load(**self.load_settings)
        ...
        
    def load_from_checkpoint(self):
        self.model=self.model_type.load_from_checkpoint(**self.load_checkpoint_settings)
    # run one trial on designated params
    def run_one_trial(self):
        now=datetime.datetime.now()
        self.one_trial_model_name='-'.join([str(now.month),str(now.day),str(now.hour),str(now.minute),'one_trial'])
        
        self.one_trial_model=self.model_type(
                                             **self.one_trial_params_dic)
        '''work_dir=self.save_path,
        model_name=self.one_trial_model_name, log_tensorboard=True,
        save_checkpoints=True,'''
        #self.one_trial_model, optimizer = amp.initialize(self.one_trial_model, optimizer, opt_level='O1')

        self.one_trial_model.fit(**self.one_trial_settings)
        
    # run hyperparams searching
    def run_optuna(self):
        p=self.optuna_params_dic
        
        X_train=self.optuna_settings['X_train']
        y_train=self.optuna_settings['y_train']
        metric_cv=self.optuna_settings['metric_cv']
        n_trials=self.optuna_settings['n_trials']
        stop_threshold=self.optuna_settings['stop_threshold']
        direction=self.optuna_settings['direction']
        
        def objective(trial,p):
            params={}
            for i in p:
                
                if len(p[i])==1:
                    params.update({i,p[i][0]})
                elif len(p[i])==4:
                    assert isinstance(p[i][1],p[i][0])
                    assert isinstance(p[i][2],p[i][0])
                    if p[i][0]==int:
                        params.update({i,trial.suggest_int(i,p[i][1],p[i][2],p[i][3])})
                    elif p[i][0]==float:
                        params.update({i,trial.suggest_float(i,p[i][1],p[i][2],p[i][3])})
                    elif p[i][0]==bool:
                        params.update({i,trial.suggest_float(i,p[i][1],[p[i][2],p[i][3]])})
                    else:
                        Warning("Invalid param type!")
            self.model=self.model_type(**params)
            self.model.fit(**self.one_trial_settings)
            
            scores=cross_val_score(self.model, X_train, y_train,
                                   cv=KFold(n_split=5,shuffle=True),
                                   scoring=metric_cv)
            
            report_cross_validation_scores(trial, scores)
            return scores.mean()
        
        def callback(study, trial):
            for ii, t in enumerate(study.trials):
                if t.value >= stop_threshold:
                    study.stop()
                    
        study = optuna.create_study(directions=direction)
        
        study.optimize(objective, n_trials=n_trials, gc_after_trial=True,
                   callbacks=[callback])
    
        print('Number of finished trials: {}'.format(len(study.trials)))
        print('Best trial:')
        trial = study.best_trial

        print('  Value: {}'.format(trial.value))
        print('  Params: ')

        for key, value in trial.params.items():
            print('    {}: {}'.format(key, value))
            
        self.optuna_study=study
        return study
    
    def refit_best_trial(self):
        assert self.optuna_study!=None
        assert self.one_trial_params_dic!=None
        
        now=datetime.datetime.now()
        self.best_model_name='-'.join([str(now.month),str(now.day),str(now.hour),str(now.minute),'best_trial'])
        params=self.one_trial_params_dic.copy()
        best_trail=self.optuna_study.best_trial
        best_params=best_trail.params
        for i in best_params:
            if i=='lr':
                params['optimizer_kwargs'].update({i:best_params[i]})
            else:
                params.update({i:best_params[i]})
        #best_params['tree_method']='gpu_hist'
        self.best_params=params
        model=self.model_type(**self.best_params)
        model.fit(**self.one_trial_settings)
        self.best_model=model
        return model
    
    def save_one_trial_model(self):
        assert self.one_trial_model !=None
        model_path=os.path.join(self.save_path,self.one_trial_model_name,'trained_model')
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_name=self.one_trial_model_name
        model_name=os.path.join(model_path,model_name+'.pt')
        self.one_trial_model.save(model_name)
        
    def save_best_model(self):
        assert self.best_model !=None
        model_path=os.path.join(self.save_path,self.best_model_name,'trained_model')
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_name=self.best_model_name
        model_name=os.path.join(model_path,model_name+'.pt')
        self.best_model.save(model_name)
        
    def predict_on_one_trial_model(self):
        self.prediction=self.one_trial_model.predict(**self.predict_settings)
        self.prediction.to_csv(os.path.join(self.save_path,self.one_trial_model_name+'prediction.csv'))
        metrics_df=self.cal_metrics()
        metrics_df.to_csv(os.path.join(self.save_path,self.one_trial_model_name+'prediction_metrics.csv'))
        
    def predict_on_best_model(self):
        self.prediction=self.best_model.predict(**self.predict_settings)
        self.prediction.to_csv(os.path.join(self.save_path,self.best_model_name+'prediction.csv'))
        
    def predict_on_loaded_model(self):
        self.prediction=self.model.predict(**self.predict_settings)
        self.prediction.to_csv(os.path.join(self.save_path,'loaded_model_prediction.csv'))

    def cal_metrics(self):
        assert self.prediction!=None
        metrics_method_dic={
            'CV':metrics.coefficient_of_variation,
            'MAE':metrics.mae,
            'MAPE':metrics.mape,
            'OPE':metrics.ope,
            'RMSE':metrics.rmse,
            'MSE':metrics.mse,
            'MARRE':metrics.marre,
            'MASE':metrics.mase,
            'R2':metrics.r2_score,
            'SMAPE':metrics.smape,
        }
        metrics_dic={
            'start_time':self.prediction.time_index[0],
            'end_time':self.prediction.time_index[-1],
            'n':len(self.prediction.time_index),
        }
        
        for metric in metrics_method_dic.keys():
            try:
                if metric=='MASE':
                    value=metrics_method_dic[metric](self.metric_settings['series_pred_gt'],self.prediction,intersect=True,
                                                              insample=self.metric_settings['series_train'], m=96*7)
                    print({metric: value})
                    metrics_dic.update({metric: value})
                else:
                    value=metrics_method_dic[metric](self.metric_settings['series_pred_gt'],self.prediction,intersect=True)
                    print({metric: value})
                    metrics_dic.update({metric: value})
            except:
                print("Fail to calculate metric: {} of model {}".format(metric,self.id))
                
        metrics_df=pd.DataFrame([metrics_dic]).T
        return metrics_df
        metrics_df.to_csv(os.path.join(self.save_path,self.id+'_metrics.csv'))

    