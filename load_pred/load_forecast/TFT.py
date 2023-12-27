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

from darts import TimeSeries

class Model_base():
    # set basic settings:
    # 1. model identification
    # 2. params of model, and whether call optuna
    # 3. prediction settings
    # 2. saving settings
    def __init__(self,id, model_type,
                 call_optuna, optuna_settings, optuna_params_dic,
                 call_one_trial, one_trial_settings, one_trial_params_dic,
                 predict_settings,save_settings):
        # for optuna params_dic should be like:
        #   dic={
        #       'param_key':[tpye, lower_bound, upper_bound, bool_log], #params to search
        #       'param_key':[value] #params not to search
        #   }
        # check if necessary key pairs are passed
        for i in ['folder','save_model','save_prediction','save_metrics']:
            assert i in save_settings.keys()
        
        self.save_settings=save_settings
        self.save_path=os.path.join(save_settings['folder'],id)
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
        
        
        self.predict_settings=predict_settings
        
        self.model_type=model_type
        self.model=None
        self.optuna_study=None
        self.best_params=None
        self.best_model=None
        self.one_trial_model=None
        
        ...
    
    # run one trial on designated params
    def run_one_trial(self):
        now=datetime.datetime.now()
        self.one_trial_model_name='-'.join([str(now.month),str(now.day),str(now.hour),str(now.minute),'one_trial'])
        
        self.one_trial_model=self.model_type(work_dir=self.save_path,
                                             model_name=self.one_trial_model_name, log_tensorboard=True,
                                             save_checkpoints=True,
                                             **self.one_trial_params_dic)
        #self.one_trial_model, optimizer = amp.initialize(self.one_trial_model, optimizer, opt_level='O1')

        self.one_trial_model.fit(**self.one_trial_settings)
        
    # run hyperparams searching
    def run_optuna(self):
        p=self.optuna_params_dic
        
        X_train=self.optuna_setting['X_train']
        y_train=self.optuna_setting['y_train']
        metric_cv=self.optuna_setting['metric_cv']
        n_trials=self.optuna_setting['n_trials']
        stop_threshold=self.optuna_setting['stop_threshold']
        direction=self.optuna_setting['direction']
        
        def objective(trial,p):
            params={}
            for i in p:
                if len(p[i])==1:
                    params.update({i,p[i][0]})
                elif len(p[i])==4:
                    assert isinstance(p[i][1],p[i][0])
                    assert isinstance(p[i][2],p[i][0])
                    if p[i][0]==int:
                        params.update({i,trail.suggest_int(i,p[i][1],p[i][2],p[i][3])})
                    elif p[i][0]==float:
                        params.update({i,trail.suggest_float(i,p[i][1],p[i][2],p[i][3])})
                    elif p[i][0]==bool:
                        params.update({i,trail.suggest_float(i,p[i][1],[p[i][2],p[i][3]])})
                    else:
                        Warning("Invalid param type!")
            self.model=self.model_type(**params)
            
            scores=cross_val_score(self.model, X_train, y_train,
                                   cv=KFold(n_split=5,shuffle=True),
                                   scoring=metric_cv)
            report_cross_validation_scores(trial, scores)
            return scores.mean()
        
        def callback(study, trial):
            for ii, t in enumerate(study.trials):
                if t.value >= threshold:
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
        best_trail=self.optuna_study.best_trial
        best_params=best_trail.params
        #best_params['tree_method']='gpu_hist'
        self.best_params=best_params
        model=self.model_type(**self.best_params)
        model.fit(self.optuna_setting['X_train'],self.optuna_setting['y_train'])
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
    def predict_on_one_trial_model(self):
        ...
        
    def predict_on_best_model(self):
        ...
        
        
# data preperation
bld_pd=pd.read_csv(r'/root/autodl-tmp/data/load_prediction_base/BLD_Sum.csv')
bld_pd.sort_values(by='DateTime')
bld_pd=bld_pd.drop(columns=['RealPower_before_scaling'])
bld=TimeSeries.from_dataframe(bld_pd,time_col="DateTime",freq="15min",fill_missing_dates=True)

transformer = Scaler()

# data split
train_start=pd.Timestamp(2017,1,1,0,0)
#train_end=pd.Timestamp(2017,2,1,0,0)
train_end=pd.Timestamp(2018,12,31,23,45)

pred_start=pd.Timestamp(2019,1,1,0,0)
pred_end=pd.Timestamp(2019,12,31,23,45)

bld_train=bld[train_start:train_end]
bld_pred=bld[pred_start:pred_end]

bld_trian=transformer.fit_transform(bld_train)

from darts.models import TFTModel
from darts.utils.likelihood_models import QuantileRegression
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

model_type=TFTModel
call_optuna=False
optuna_settings=None
optuna_params_dic=None
call_one_trial=True
quantiles = [
    0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,
    0.6,0.7,0.75,0.8,0.85,0.9,0.95,0.99,
]

one_trial_settings={
    'series':bld_train['RealPower'],
    'future_covariates':bld_train.drop_columns(['RealPower', 'wind_speed', 'wind_deg', 'rain_1h',
       'rain_3h', 'snow_1h', 'snow_3h', 'clouds_all', 'weather_main',
       'RealPower_-0d_0h'])
}
my_stopper = EarlyStopping(
    monitor="val_loss",
    patience=2,
    min_delta=1e-2,
    mode='min',
)

one_trial_params_dic={
    'optimizer_kwargs':{
        'lr':1e-4,
    },
    'pl_trainer_kwargs':{"callbacks": [my_stopper]},
    'batch_size':96,
    'dropout':0.2,
    'input_chunk_length':96*7,
    'output_chunk_length':96,
    'hidden_size':128,
    'lstm_layers':6,
    'num_attention_heads':6,
    #'loss_fn':torch.nn.MSELoss(),
    'likelihood':QuantileRegression(
        quantiles=quantiles
    ),
    'n_epochs':100,
    
    'pl_trainer_kwargs':{
      "accelerator": "gpu",
      "devices": [0]
    },
    'random_state':42,
}
predict_settings=None
save_settings={
    'folder':r'/root/autodl-tmp/load_forecast/Temporal_Fusion_Transformer',
    'save_model':True,'save_prediction':True,'save_metrics':True
}

test=Model_base(
    'on_params_in_previous_paper',
    model_type,
    call_optuna,
    optuna_settings,
    optuna_params_dic,
    call_one_trial,
    one_trial_settings,
    one_trial_params_dic,
    predict_settings,
    save_settings
)

test.run_one_trial()
test.save_one_trial_model()