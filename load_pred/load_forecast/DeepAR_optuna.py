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

from scikeras.wrappers import KerasClassifier
from model_base import *


bld_pd=pd.read_csv(r'/root/autodl-tmp/data/load_prediction_base/BLD_Sum.csv')
bld_pd.sort_values(by='DateTime')

bld_target_pd=bld_pd[['RealPower','DateTime']]
bld_past_co_pd=bld_pd[['DateTime','temp','feels_like', 'temp_min', 'temp_max', 'pressure', 'humidity']]
bld_future_co_pd=bld_pd[['DateTime','hour_cos', 'hour_sin','dayofweek_cos', 'dayofweek_sin',
       'month_cos', 'month_sin', 'dayofmonth_cos', 'dayofmonth_sin']]


bld_target=TimeSeries.from_dataframe(bld_target_pd,time_col="DateTime",freq="15min",fill_missing_dates=True)
bld_past_co=TimeSeries.from_dataframe(bld_past_co_pd,time_col="DateTime",freq="15min",fill_missing_dates=True)
bld_future_co=TimeSeries.from_dataframe(bld_future_co_pd,time_col="DateTime",freq="15min",fill_missing_dates=True)

transformer = Scaler()

# data split
train_start=pd.Timestamp(2017,4,1,0,0)
train_end=pd.Timestamp(2018,12,31,23,45)

val_start=pd.Timestamp(2017,1,1,0,0)
val_end=pd.Timestamp(2017,3,31,23,45)

pred_start=pd.Timestamp(2019,1,1,0,0)
pred_end=pd.Timestamp(2019,12,31,23,45)

# define objective function
def objective(trial):
    
    # detect if a GPU is available
    if torch.cuda.is_available():
        num_workers = 4
    else:
        num_workers = 0


    # reproducibility
    torch.manual_seed(42)
    
    my_stopper = EarlyStopping(
        monitor="train_loss",
        patience=5,
        min_delta=1e-1,
        mode='min',
    )

    # build the TCN model
    model = RNNModel(
        optimizer_kwargs={
            'lr':trial.suggest_float("lr", 1e-5, 10),
        },
        pl_trainer_kwargs={
            "callbacks": [my_stopper],
            "gradient_clip_val":0.1,
            "accelerator": "gpu",
            "devices": [0]
            },
        model='LSTM',
        hidden_dim=trial.suggest_int("hidden_dim", 8, 256),
        n_rnn_layers=trial.suggest_int("n_rnn_layers", 1, 8),
        batch_size=trial.suggest_int("batch_size", 8, 96*7),
        dropout=trial.suggest_float("dropout", 0.01, 1),
        input_chunk_length=96*7,
        output_chunk_length=96,
        #'loss_fn':torch.nn.GaussianNLLLoss(reduction='mean'),
        #'likelihood':GaussianLikelihood(),
        # QuantileRegression(
        #    quantiles=quantiles
        #),
        n_epochs=trial.suggest_int("n_epochs", 1, 50),

        random_state=42,
        
    )

    model.fit(
        series=bld_target[train_start:train_end],
        future_covariates=bld_future_co[train_start:train_end],
        val_series=bld_target[val_start:val_end],
        val_future_covariates=bld_future_co[val_start:val_end],
        
        num_loader_workers=4
    )


    # Evaluate how good it is on the validation set, using sMAPE
    preds = model.predict(    
                        n=len(bld_target[val_start:val_end]),
                        series=bld_target[train_start:train_end],
                        future_covariates=bld_future_co,
                        n_jobs=-1,
                        num_samples=1)
    '''    
    scores=cross_val_score(model, bld_target[train_start:train_end],
                            cv=KFold(n_splits=4,shuffle=True),
                            scoring='neg_mean_absolute_error')# bld_future_co[train_start:train_end], 
                            '''
    
    smapes = smape(bld_target, preds, n_jobs=-1, verbose=True,intersect=True)
    smape_val = np.mean(smapes)
    #return scores.mean()
    return smape_val if smape_val != np.nan else float("inf")


# for convenience, print some optimization trials information
def print_callback(study, trial):
    print(f"Current value: {trial.value}, Current params: {trial.params}")
    print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")

study_name = "DeepAR-study-test2-trial120"  # Unique identifier of the study.
storage_name = "sqlite:///{}.db".format(study_name)
# optimize hyperparameters by minimizing the sMAPE on the validation set
study = optuna.create_study(study_name=study_name, storage=storage_name,direction="minimize",load_if_exists=True)

study.optimize(objective, n_trials=120, callbacks=[print_callback])



model_type=RNNModel
call_optuna=False
optuna_settings=None
optuna_params_dic=None
call_one_trial=False
quantiles = [
    0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,
    0.6,0.7,0.75,0.8,0.85,0.9,0.95,0.99,
]

one_trial_settings={
    'series':bld_target[train_start:train_end],
    'future_covariates':bld_future_co[train_start:train_end],
    #'past_covariates':bld_past_co[train_start:train_end],
    'val_series':bld_target[val_start:val_end],
    'val_future_covariates':bld_future_co[val_start:val_end],
    #'val_past_covariates':bld_past_co[val_start:val_end],
    'num_loader_workers':4
}

my_stopper = EarlyStopping(
    monitor="val_loss",
    patience=5,
    min_delta=1e-1,
    mode='min',
)

one_trial_params_dic={
    'optimizer_kwargs':{
        'lr':1e-4,
    },
    'pl_trainer_kwargs':{
        "callbacks": [my_stopper],
        "gradient_clip_val":0.1},
    'model':'LSTM',
    'hidden_dim':128,
    'n_rnn_layers':4,
    'batch_size':64,
    'dropout':0.2,
    'input_chunk_length':96*7,
    'output_chunk_length':96,
    #'loss_fn':torch.nn.GaussianNLLLoss(reduction='mean'),
    #'likelihood':GaussianLikelihood(),
    # QuantileRegression(
    #    quantiles=quantiles
    #),
    'n_epochs':30,
    'pl_trainer_kwargs':{
      "accelerator": "gpu",
      "devices": [0]
    },
    'random_state':42,
}
predict_settings={
    'n':96*365,
    'series':bld_target[:pred_start],
    #'past_covariates':bld_past_co,
    'future_covariates':bld_future_co,
    'n_jobs':-1,
    'num_samples':1
}
metric_settings={
    'series_pred_gt':bld_target,
    'series_train':bld_target[train_start:train_end],
}
save_settings={
    'folder':r'/root/autodl-tmp/load_forecast/DeepAR(Prob_RNN)/optuna',
    'save_model':True,'save_prediction':True,'save_metrics':True
}
test=Model_base(
    'optuna_test2_trial120',
    model_type,
    call_optuna,
    optuna_settings,
    optuna_params_dic,
    call_one_trial,
    one_trial_settings,
    one_trial_params_dic,
    False,
    None,
    False,
    None,
    predict_settings,
    metric_settings,
    save_settings
)
test.optuna_study=study

test.refit_best_trial()
test.save_best_model()

prediction=test.best_model.historical_forecasts(
    series=bld_target[pd.Timestamp(2018,12,24,0,0):pred_end],
    future_covariates=bld_future_co[train_start:pred_end],
    #past_covariates=bld_past_co[train_start:pred_end],
    num_samples=1,
    start=pd.Timestamp(2019,1,1,0,0),
    forecast_horizon=96,
    stride=96,
    retrain=False,
    verbose=True,
    last_points_only=False
)

temp=prediction[0]
for i in range(len(prediction)-1):
    temp=temp.concatenate(prediction[i+1])
    
test.prediction=temp
metrics=test.cal_metrics()
test.prediction.to_csv(os.path.join(test.save_path,test.id+'_optuna_prediction.csv'))
metrics.to_csv(os.path.join(test.save_path,test.id+'_optuna_metrics.csv'))
