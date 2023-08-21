import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.preprocessing import FunctionTransformer

import numpy as np
import json
import os

#import sklearn

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from xgboost import XGBRegressor, plot_importance
import optuna

from optuna.terminator import report_cross_validation_scores
from datetime import timedelta

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

import matplotlib.pyplot as plt

from optuna.visualization import plot_contour
#from optuna.visualization import plot_edf
#from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
#from optuna.visualization import plot_parallel_coordinate
#from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice

from astral.location import Location, LocationInfo
from backports.zoneinfo import ZoneInfo # Python 3.9


def intersection_sum(prefix='BLD',postfix=None,fn_list=None,src_path=None,key=None,save_fn=None,
                     interpolate=False,start_t=None,end_t=None):
    # input csv file must contain "DateTime" as index
    # default interval: 900s
    # key: column on which needs to get intersection 
    # start_t, end_t: common max period, shape like: 'mm-dd-yyyy'
    
    # read all csv files in the list as a list of dataframe
    dataframes=[]
    for i in fn_list:
        fn=src_path+'/'+prefix+'_'+i+'_'+postfix+'.csv'
        df=pd.read_csv(fn)
        if 'DateTime' not in df.columns:
            assert Exception("Invalid input csv file.")
        #turn the datetime row into type(DateTime)
        df['DateTime']=pd.to_datetime(df['DateTime'])
        # slice the original data, left and right not inclusive
        if start_t !=None:
            df=df.loc[df['DateTime']>start_t].copy()
        if end_t !=None: 
            df=df.loc[df['DateTime']<end_t].copy()   
        #if specified to interpolate
        if(interpolate):
            helper=pd.DataFrame({'DateTime':pd.date_range(start=df['DateTime'].min(),
                                                          end=df['DateTime'].max(),freq='900s')})
            df=pd.merge(df,helper,on='DateTime',how='outer').sort_values("DateTime")
            df[key]=df[key].interpolate(method ='linear')
        #print(df.info())
        df['DateTime']=pd.to_datetime(df['DateTime'])
        df=df.set_index('DateTime').copy()
        dataframes.append(df)
      
    #get intersection of datetime of all sub dataframes
    common_datetime=set(dataframes[0].index)
    for df in dataframes[1:]:
        common_datetime=common_datetime.intersection(df.index)
        print('Progress notification: getting intersections')
    #print(common_datetime)
    
    print(dataframes[0].index)
    
    #Create a new empty dataframe   
    combined_df=pd.DataFrame(columns=['DateTime',key])

    #Iterate over the common datetime index
    for dt in common_datetime:
        data_sum = 0
        i=0
        for df in dataframes:
            #row = df[df['DateTime'] == dt]
            #if not row.empty:
                #data_sum += row[key].values[0]
            try:
                value=df.loc[dt,key]
                if value!=None:
                    data_sum+=value
            except:
                print('skipped:'+str(dt))
                pass
            
        data={'DateTime': [dt], key: [data_sum]}
        add_row=pd.DataFrame(data=data)
        # Add the combined row to the new dataframe
        combined_df = pd.concat([combined_df,add_row], ignore_index=True)
    
    combined_df=combined_df.sort_values(by=["DateTime"],ignore_index=True)
    combined_df.to_csv(save_fn,index=False)

class Data_encoder():
    '''
    init requires inputs like:
        weather_setting_={
            'weather_fn':'',
            'keys':['temp','temp_min','temp_max'],
            'shift':[3,3,3] 
        }    
        data_setting_={
            'data_fn':'RobinsonHall',
            'src_path':r'D:/Codes/GIthub_repo/Energy_grid/data/UCSD_raw_data/',
            'load_type':'BLD',
            'split_date':'01-01-2019'
        }
    '''      
    def __init__(self,weather_setting_,data_setting_):
        
        #check if all needed keys are specified
        data_setting_key_list=['load_from_existing_file','data_fn','src_path','load_type',
                               'split_date','save_folder','save_prefix','days_ahead']
        for i in data_setting_key_list:
            if i not in data_setting_.keys():
                raise Exception('Unspecified keys:'+i)
        self.load=None
        self.meta_load=None
        
        #get settings
        self.weather_setting=weather_setting_.copy()
        self.data_setting=data_setting_.copy()
        
        #get weather data, historical weather was shifted here
        self.weather=self.get_weather_data()
        
        #get raw load data, datatime type specified, 'hour_match' generated
        self.load=self.get_load_data().copy()
        
        self.load=self.encode_time_features().copy()
        #get the 
        self.encoded_load=self.encode_load_shift()
        
        self.merged_load=self.get_merged_data()
        self.save_data()
        
    def encode_shift(self,encode,df,unit):
        len(encode['keys'])
        df_encode=df.copy()
        #add a column for further merge
        df_encode['hour_match']=df_encode['DateTime'].dt.strftime("%Y-%m-%d-%H")
        for i in range(len(encode["keys"])):
            for k in range(encode["shift"][i-1]):
                df_encode[encode["keys"][i-1]+'_-'+str(k+1)+unit]=df[encode["keys"][i-1]].shift(k+1)
        return df_encode
        
    def get_weather_data(self):
        #read weather data and transform the datetime
        raw=pd.read_csv(self.weather_setting['weather_fn'])
        raw['DateTime'] = pd.to_datetime(\
            raw['dt_iso'].apply(lambda x: x.replace(' +0000 UTC', '')), format='%Y-%m-%d %H:%M:%S')

        raw=raw.drop_duplicates(subset='DateTime')

        weather_keys=self.weather_setting['keys'].copy()
        weather_keys.append('DateTime')
        weather=raw[weather_keys]
        
        #turn weather data into floats
        for i in self.weather_setting['keys']:
            weather[i]=weather[i].astype(float)

        #replace weather description with intergers
        if 'weather_main' in weather_keys:
            column_dict = {'Clear': 1, 'Clouds':2, "Drizzle":3, "Dust":4,"Fog":5, 
                        "Haze":6, "Mist":7, "Rain":8, "Smoke":9, "Thunderstorm":10}
            weather = weather.replace({"weather_main": column_dict})
            weather['weather_main'].astype(int)
        
        #get the historical data according to the dict  
        weather_setting_inside=self.weather_setting.copy()
        
        weather=self.encode_shift(weather_setting_inside,weather,'h').copy()
        weather=weather.drop(columns=weather_keys)
        
        return weather
    
    def get_load_data(self):
        path=self.data_setting['src_path']+\
            self.data_setting['load_type']+'_'+\
            self.data_setting['data_fn']+'.csv'
        load=pd.read_csv(path)
        load['DateTime']=pd.to_datetime(load["DateTime"])
        load=load.drop_duplicates(subset='DateTime')
        load['hour_match']=load['DateTime'].dt.strftime("%Y-%m-%d-%H")
        self.load=load.copy()
        return load
    

    
    def encode_time_features(self):
        
        if self.load is None:
            raise Exception("Please load raw data first")
        #load original data
        df=self.load
        
        #calculate dict for holidays
        cal = USFederalHolidayCalendar()
        holidays_temp = cal.holidays(start='2014-01-01', end='2021-12-31').to_pydatetime()
        
        holidays=[]
        for i in holidays_temp:
            holidays.append(i.date())

        
        #get origin time features
        df['hour'] = df['DateTime'].dt.hour
        df['quarter'] = df['DateTime'].dt.quarter
        df['month'] = df['DateTime'].dt.month
        #df['year'] = df['DateTime'].dt.year
        #df['dayofyear'] = df['DateTime'].dt.dayofyear
        df['dayofmonth'] = df['DateTime'].dt.day
        
        
        if self.data_setting['load_type'] in ['pv','PV']:
            if 'round_precision' in self.data_setting.keys():
                precision=int(self.data_setting['round_precision'])
            else:
                precision=1
            l = LocationInfo('San Diego', 'California', 'United States')
            df['solar_zenith']=df['DateTime'].apply(lambda x: \
                np.round(Location(l).solar_zenith(x.replace(tzinfo=ZoneInfo('America/Los_Angeles'))),precision))
            df['solar_azimuth']=df['DateTime'].apply(lambda x: \
                np.round(Location(l).solar_azimuth(x.replace(tzinfo=ZoneInfo('America/Los_Angeles'))),precision))
        
        if self.data_setting['load_type'] in ['bld','BLD']:
            df['dayofweek'] = df['DateTime'].dt.dayofweek
            df['is_holiday'] = df['DateTime'].apply(lambda x: x.date() in holidays)
            df['is_holiday'] = df['is_holiday'].astype(int)
        
        self.meta_load=df.copy()
        
        
        #helper function for feature encoding
        #functions only within this scope
        def sin_cos_circle(df,key,period):
            key_cos=key+'_cos'
            df[key_cos]=self.cos_transformer(period).fit_transform(df[key])
            key_sin=key+'_sin'
            df[key_sin]=self.sin_transformer(period).fit_transform(df[key])
            df=df.drop(columns=[key])

        def encode_time_features_targeting(ori,start_date,end_date):
            df=ori.copy()
            df=df.set_index("DateTime").sort_index()
            df=df.loc[df.index>=start_date]
            df_base=df.loc[df.index<end_date].copy()
            
            #calculate the maping encode for each time feature in the following list
            
            if self.data_setting['load_type'] in ['pv','PV']:
                target_encode_columns = ['hour', 'quarter', 'month', 'dayofmonth','solar_zenith','solar_azimuth']
            if self.data_setting['load_type'] in ['bld','BLD']:
                target_encode_columns = ['hour', 'dayofweek', 'quarter', 'month', 'dayofmonth', 'is_holiday']
            target = ['RealPower']
            target_encode_df = df_base[target_encode_columns + target].reset_index().drop(columns = 'DateTime', axis = 1)
            val_maps=list()
            for embed_col in target_encode_columns:
                val_map = target_encode_df.groupby(embed_col)[target].mean().to_dict()[target[0]]
                val_maps.append(val_map)
            
            #apply the mapping to the data
            k=0
            #df=ori.copy()
            for key in target_encode_columns:
                df[key]=df[key].map(val_maps[k]).values
                k=k+1
            return df
               
        #apply the helper function
        dic={
            'hour':24, 'dayofweek':7, 'quarter':4, 
            'month':12, 'dayofmonth':31 #FIXME: dayofmonth vary in months
        }
        if self.data_setting['enable_target_encoding'] is True:
            df=encode_time_features_targeting(ori=self.meta_load,
                                           start_date=self.data_setting['target_encoding_start'],
                                           end_date=self.data_setting['split_date'])
            
        else:        
            for i in dic.keys():
                sin_cos_circle(df,i,dic[i])
            df=df.drop(columns=['hour','dayofweek','quarter','month','dayofmonth'],errors='ignore')
        return df
    
    def encode_load_shift(self):
        if self.data_setting['days_ahead'] is None:
            return self.load
        temp=self.load.copy()
        
        #temp=temp.set_index('DateTime')
        #temp.index=pd.to_datetime(temp.index)
        

        for h in self.data_setting['hours_ahead']:
            for i in self.data_setting['days_ahead']:                      
                new_column='RealPower_-'+str(i)+'d'+'_'+str(h)+'h'
                temp[new_column]=None  
                '''
                helper=self.load[['DateTime']]
                print(helper.info())
                helper['DateTime']=pd.to_datetime(helper['DateTime'])
                
                helper['DateTime_new']=None
                print(helper.info())
                helper['DateTime_new']=helper['DateTime'].map(lambda x:x-timedelta(days=int(i)))
                print(helper.info())
                print(helper.head())
                #helper.set_index['DateTime']
                helper['tag']=None
                valid_list=[]
                for i in helper['DateTime_new']:
                    if i in helper['DateTime']:
                        valid_list.append(i)        
                print(helper.info())
                print(len(valid_list))
                '''
                for k in temp.index:
                    index_new=k-timedelta(days=int(i))-timedelta(hours=int(h))
                    try:
                        temp.at[k,new_column]=temp.at[index_new,'RealPower']
                    except KeyError:
                        print('Skipped:')
                        print(k)
                        pass
        temp.index.name='old'
        temp['DateTime']=temp.index
        self.load=temp.copy()
        #print(temp)
        return temp

    def sin_transformer(self,period):
        return FunctionTransformer(lambda x: np.sin(x/period*2*np.pi))
    
    def cos_transformer(self,period):
        return FunctionTransformer(lambda x: np.cos(x/period*2*np.pi))
    
    def get_merged_data(self):
        return pd.merge(self.encoded_load,self.weather,how='left',on='hour_match')\
            .sort_values(by='DateTime').set_index('DateTime').drop(columns=['hour_match'])
                        
    def save_data(self):
        save_folder=self.data_setting['save_folder']+self.data_setting['save_prefix']+'/'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        save_fn=save_folder+self.data_setting['load_type']+'_'+self.data_setting['data_fn']+'.csv'
        self.merged_load.to_csv(save_fn)
                
def Optuna_core(X_train,y_train,n_trials,metrics,stop_threshold,params_):
    threshold=stop_threshold
    def objective(trial):
        #X, y = load_wine(return_X_y=True)
        
        #the ratio of features used (i.e. columns used); 
        # colsample_bytree. Lower ratios avoid over-fitting.
        #the ratio of the training instances used (i.e. rows used); 
        # subsample. Lower ratios avoid over-fitting.
        #the maximum depth of a tree; 
        # max_depth. Lower values avoid over-fitting.
        #the minimum loss reduction required to make a further split; 
        # gamma. Larger values avoid over-fitting.
        #the learning rate of our GBM (i.e. how much we update our prediction with each successive tree); 
        # eta. Lower values avoid over-fitting.
        #the minimum sum of instance weight needed in a leaf, in certain applications this relates directly to the minimum number of instances needed in a node; 
        # min_child_weight. Larger values avoid over-fitting.
        p=params_
        
        
        params = {
            'max_depth': trial.suggest_int('max_depth', p['max_depth'][0], p['max_depth'][1]),
            'learning_rate': trial.suggest_float('learning_rate', p['learning_rate'][0], p['learning_rate'][1], log=True),
            'n_estimators': trial.suggest_int('n_estimators', p['n_estimators'][0], p['n_estimators'][1]),
            'min_child_weight': trial.suggest_int('min_child_weight', p['min_child_weight'][0], p['min_child_weight'][1]),
            'gamma': trial.suggest_float('gamma', p['gamma'][0], p['gamma'][1], log=True),
            'subsample': trial.suggest_float('subsample', p['subsample'][0], p['subsample'][1], log=True),
            'colsample_bytree': trial.suggest_float('colsample_bytree', p['colsample_bytree'][0], p['colsample_bytree'][1], log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', p['reg_alpha'][0], p['reg_alpha'][1], log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', p['reg_lambda'][0], p['reg_lambda'][1], log=True),
            'eval_metric': 'mlogloss',
            'use_label_encoder': False,
            'tree_method':'gpu_hist'
        }
        
        
        '''
        OLD Version overfit
        params = {
            'max_depth': trial.suggest_int('max_depth', 1, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 5, 500),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.01, 1.0, log=True),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.01, 1.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
            'eval_metric': 'mlogloss',
            'use_label_encoder': False,
            'tree_method':'gpu_hist'
        }
        '''
        # Fit the model
        xgb_model = XGBRegressor(**params)


        scores = cross_val_score(xgb_model, X_train, y_train, 
                                cv=KFold(n_splits=5, shuffle=True),
                                scoring=metrics)
        report_cross_validation_scores(trial, scores)
        return scores.mean()

    def callback(study, trial):
        for ii, t in enumerate(study.trials):
            if t.value >= threshold:
                study.stop()

    study = optuna.create_study(direction="maximize")

    study.optimize(objective, n_trials=n_trials, gc_after_trial=True,
                   callbacks=[callback])
    
    
    
    print('Number of finished trials: {}'.format(len(study.trials)))
    print('Best trial:')
    trial = study.best_trial

    print('  Value: {}'.format(trial.value))
    print('  Params: ')

    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))
        
    return study

class XGBoost_Optuna():
    
    def __init__(self,data_setting_,weather_setting_,model_setting_):
        
        self.X_train=None
        self.y_train=None
        self.X_test=None
        self.y_test=None
        self.data=None
        self.data_setting=dict()
        self.weather_setting=weather_setting_
        self.model_setting=model_setting_
        
        self.optuna_study=None
        
        save_folder=self.model_setting['model_folder']+self.model_setting['save_prefix']+'/'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        self.save_folder=save_folder
        #self.model_params=model_params_
        
        #check if all needed keys are specified
        data_setting_key_list=['load_from_existing_file','data_fn','src_path','load_type',
                               'split_date','save_folder','save_prefix']
        for i in data_setting_key_list:
            if i not in data_setting_.keys():
                raise Exception('Unspecified keys:'+i)
        if data_setting_['load_type'] not in ['BLD','PV']:
            raise Exception("Incorrect load_type, please enter BLD or PV")
        
        #load data
        self.data_setting=data_setting_
        if self.data_setting['load_from_existing_file'] is True:
            self.data=self.load_existing_file()
        else:
            self.data=Data_encoder(weather_setting_=weather_setting_,
                                    data_setting_=data_setting_).merged_load
            
        #get the split of load data
        self.get_data_split()
        
        
    def load_existing_file(self):
        load_path=self.data_setting['save_folder']+self.data_setting['save_prefix']+'/'\
            +self.data_setting['load_type']+'_'+self.data_setting['data_fn']+'.csv'
        load=pd.read_csv(load_path,index_col=0)
        load.index=pd.to_datetime(load.index)
        if 'ReactivePower' in load.columns:
            load=load.drop(columns=['ReactivePower'])
        return load
        
    def get_data_split(self):
        key='RealPower'
        if self.data is None:
            raise Exception("Fail to load existing file")
        if 'ReactivePower' in self.data.index:
            self.data=self.data.drop(columns=['ReactivePower']).copy()
        temp=self.data.loc[self.data.index < '01-01-2020'].copy()
        train = temp.loc[temp.index < self.data_setting['split_date']].copy()
        test = temp.loc[temp.index >= self.data_setting['split_date']].copy()
        self.X_train=train.drop(columns=[key])
        self.y_train=train[key]
        self.X_test=test.drop(columns=[key])
        self.y_test=test[key]    
        
    def optuna_optimizer(self):
        optuna_study=Optuna_core(self.X_train, self.y_train,
                                 self.model_setting['n_trials'],
                                 self.model_setting['metrics'],
                                 self.model_setting['stop_threshold'],
                                 self.model_setting['params'])
        self.optuna_study=optuna_study
        return optuna_study
    
    def optuna_visualization(self):
        if self.optuna_study is None:
            raise Exception('The optuna process failed or the record is missing.')
        
        for i in self.model_setting['visualization_types']:
            save_fn=self.save_folder+self.data_setting['load_type']+'_'+self.data_setting['data_fn']+'_'+i+'.png'
            if i=='optimization_history':
                plot_optimization_history(self.optuna_study).update_layout(width=1400,height=1000,uniformtext_minsize=18)\
                    .write_image(save_fn)
            elif i=='contour':
                plot_contour(self.optuna_study).update_layout(width=2100,height=1500,uniformtext_minsize=18)\
                    .write_image(save_fn)
            elif i=='slice':
                plot_slice(self.optuna_study).update_layout(width=5000,height=1000,uniformtext_minsize=24)\
                    .write_image(save_fn)
                
    def refit_best_trail(self):
        if self.optuna_study is None:
            raise Exception('The optuna process failed or the record is missing.')
        best_trail=self.optuna_study.best_trial
        best_params=best_trail.params
        best_params['tree_method']='gpu_hist'
        self.best_params=best_params
        model=XGBRegressor(**self.best_params)
        model.fit(self.X_train,self.y_train)
        self.best_model=model
        return model
    
    def predict_n_evaluate(self):
        if self.best_model is None:
            raise Exception("Please train best model before predicting.")
        y_pred = self.best_model.predict(self.X_test)
        y_test_ = self.y_test.values
        pd_y_test=pd.DataFrame(self.y_test)
        pd_y_test['RealPower_pred']=y_pred
        save_fn_pred=self.save_folder+self.data_setting['load_type']+'_'+self.data_setting['data_fn']
        pd_y_test.to_csv(save_fn_pred+'_XGBoost_prediction.csv')
        
        def feature_importance_selected(clf_model):

            feature_importance = clf_model.get_booster().get_fscore()
            feature_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            feature_ipt = pd.DataFrame(feature_importance, columns=['feature_name', 'importance'])
            save_fn_1=self.save_folder+self.data_setting['load_type']+'_'+self.data_setting['data_fn']+'_feature_importance'+'.csv'
            feature_ipt.to_csv(save_fn_1, index=False)
            print('feature_importance:', feature_importance)
            save_fn_2=self.save_folder+self.data_setting['load_type']+'_'+self.data_setting['data_fn']+'_feature_importance'+'.png'
            plot_importance(clf_model,max_num_features=20,height=0.6,grid=False)
            plt.savefig(save_fn_2)
            
        def metrics_sklearn(y_valid, y_pred_):
            """模型对验证集和测试集结果的评分"""
            # 准确率
            #accuracy = accuracy_score(y_valid, y_pred_)
            #print('Accuracy:%.2f%%' % (accuracy * 100))

            # error
            mean_squared_error_ = mean_squared_error(y_valid, y_pred_)
            print('mean_squared_error:%.2f' % mean_squared_error_)
            
            mean_absolute_error_ = mean_absolute_error(y_valid, y_pred_)
            print('mean_absolute_error:%.2f' % mean_absolute_error_)

            def mean_absolute_percentage_error(y_true, y_pred): 
                """Calculates MAPE given y_true and y_pred"""
                y_true, y_pred = np.array(y_true), np.array(y_pred)
                return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            
            percentage_error = mean_absolute_percentage_error(y_valid, y_pred_)
            print('percentage_error:%.2f%%' % percentage_error)
        
        # 模型对验证集预测结果评分
        metrics_sklearn(y_test_, y_pred)

        # 模型特征重要性提取、展示和保存
        feature_importance_selected(self.best_model)   
    
    def save_model(self):
        if self.best_model is None:
            raise Exception("Please train best model before predicting.")
        save_fn=self.save_folder+self.data_setting['load_type']+'_'+self.data_setting['data_fn']
        self.best_model.save_model(save_fn+'_XGBoost_Regression_model.model')
       
class Simple_forecast():
    def __init__(self,src_path,ref_start_t,pred_start_t,end_t,method,save_path,**kwargs):
        #prepare the raw data
        self.pred=None 
        self.kwargs=kwargs
        self.save_path=save_path
        alpha=self.kwargs['alpha']
        K=int(self.kwargs['K'])
        
        df=pd.read_csv(src_path).drop_duplicates(subset='DateTime')
        df["DateTime"]=pd.to_datetime(df["DateTime"])
        df=df.sort_values(by="DateTime")
        df=df.set_index("DateTime")
        self.pred_start_t=pred_start_t
        self.end_t=end_t
        data=df[pred_start_t:end_t].copy()
        pred_ref=df[pd.to_datetime(ref_start_t):pd.to_datetime(end_t)+timedelta(minutes=15*(K+5))].copy()
        self.data=data
        
        
        #helper dataframe
        rng = pd.date_range(start=pred_ref.index.min(),\
            end=pred_ref.index.max(), freq = '15min')
        pd_rng=pd.DataFrame(rng)
        pd_rng=pd_rng.rename(columns={0:'DateTime'}).copy()
        
        pred_ref=pd.merge(pred_ref,pd_rng,on='DateTime',how='right').fillna(0)
        pred_ref=pred_ref.set_index('DateTime')
        self.pred_ref=pred_ref
        
        
        
        self.pred=self.prediction_ave_by_prediction_time().copy()
        
        #call the prediction method according to the specified method
        if method not in ['ave_by_prediction_time','ave_by_prediction_value']:
            assert Exception("Invalid method!")
        if method == 'ave_by_prediction_time':
            self.pred=self.prediction_ave_by_prediction_time().copy()
        if method == 'ave_by_prediction_value':
            self.pred=self.prediction_ave_by_prediction_value().copy()
            
        
        ...
        
    def save_prediction(self):
        #assert self.pred!=None
        self.pred.to_csv(self.save_path)
        print("File saved to:"+self.save_path)
        
    def prediction_ave_by_prediction_time(self):
        alpha=self.kwargs['alpha']
        K=int(self.kwargs['K'])
        #helper dataframe
        rng = pd.date_range(start=pd.to_datetime(self.pred_start_t),\
            end=pd.to_datetime(self.end_t)+(K+5)*timedelta(minutes=15), freq = '15min')
        pd_rng=pd.DataFrame(rng)
        pd_rng=pd_rng.rename(columns={0:'DateTime'}).copy()
        
        indexes_pred=pd.date_range(start=self.pred_start_t,end=self.end_t, freq = '15min')
        #get a helper dataframe containing all timesteps in the duration
        #kind of interpolation
        data_n=pd.merge(self.data,pd_rng,on='DateTime',how='right')
        data_n['RealPower_pred'] = data_n.apply(lambda x: [], axis=1)
        data_n['DateTime']=pd.to_datetime(data_n['DateTime'])
        data_n=data_n.set_index('DateTime')
        data_n['RealPower']=data_n['RealPower'].astype(float)
        
        for i in data_n.index:
            if not isinstance(data_n.at[i,'RealPower'],float):
                data_n.at[i,'RealPower']=None    
        
        data_n['historical_ave']=None
        data_n['future_real']=data_n.apply(lambda x: [], axis=1)
        data_n['future_real']=data_n['future_real'].astype(object)
        
        
        
        
        days_list=None
        if self.kwargs['frequent']=='week':
            weeks_list=list(range(1,int(self.kwargs["n"])+1))
            days_list=[i * 7 for i in weeks_list]
           
        if self.kwargs['frequent']=='day':
            days_list=list(range(1,int(self.kwargs["n"])+1))
            
        if indexes_pred.min()-timedelta(days=max(days_list)) < self.pred_ref.index.min():
            print("Alert: insufficient data in ref_duration")
            
        for i in indexes_pred:
            # calculate historical average of all index in prediction duartion   
            history=[]
            for k in days_list:
                index_ahead=i-timedelta(days=int(k))
                if index_ahead in self.pred_ref.index:
                    history.append(self.pred_ref.at[index_ahead,'RealPower'])             
            ave=0
            count=0
            for m in history:
                if isinstance(m,(str,float,int)):
                    ave=ave+float(m)
                    count=count+1
            if count>0:
                ave=ave/count
                data_n.at[i,'historical_ave']=ave
                
            # to get the 
            idx_start=i+timedelta(minutes=15)
            idx_end=i+timedelta(minutes=15*K)
            meta=self.pred_ref[idx_start:idx_end]
            future=meta['RealPower'].values.tolist()
            #
            # print(future)

            data_n.at[i,'future_real']=future

                
        data_n=data_n.reset_index()
        
        last_weights = np.exp(- alpha*(np.arange(K)+1)) if alpha is not None else 0
        if 'ReactivePower' in data_n.columns:
            data_n=data_n.drop(columns=['ReactivePower'])
        
        
        print(data_n.info())
        data_n=data_n.set_index('DateTime')
        
        for i in indexes_pred:
            
            last=self.pred_ref.at[i,'RealPower']
            
            idx_start=i+timedelta(minutes=15)
            idx_end=i+timedelta(minutes=15*K)
            meta=data_n.loc[idx_start:idx_end].fillna(0)
            historical_ave=meta['historical_ave'].values.tolist()
            assert len(historical_ave)==K
            pred=[]
            for k in range(K):
                #print(type(k))
                pred.append(last*last_weights[k]+historical_ave[k]*(1-last_weights[k]))
            
            data_n.at[i,'RealPower_pred']=pred
            
            
            
        '''
        
        # old method
        for n in range(len(data_n)-K-1):
            # "realpower" 1st column
            last=data_n.iat[n,1]
            if not isinstance(last,float):
                last=None
            #last_weights = np.exp(- alpha*(np.arange(K)+1)) if alpha is not None else 0
            for i in range(K):
                #"historical_ave" 3rd column, 'Pred' 2nd column
                share=last_weights[i].copy()
                if not isinstance(share,float):
                    print(last_weights)
                    print(type(last_weights))
                    raise Exception('invalid type of last_weights')
                if not isinstance(last,(float,None)):
                    print(last)
                    print(type(last))
                    raise Exception('invalid type of last')  
                if  data_n.iat[n+i,3] != None:       
                    if last != None:
                        pred_value=share*last+(1-share)*data_n.iat[n+i,3]
                    else:
                        pred_value=data_n.iat[n+i,3]
                    data_n.iat[n+i,2].append(pred_value)
                else:
                    data_n.iat[n+i,2].append(None)
        
        '''
        #print(data_n.head(50))         
        for k in self.kwargs['metrics_steps']:
            data_n['MAE_'+str(k)]=None
            for i in indexes_pred:
                data_n.loc[i,'MAE_'+str(k)]=\
                    mean_absolute_error(data_n.at[i,'RealPower_pred'][:k],data_n.loc[i,'future_real'][:k])
            data_n['RMSE_'+str(k)]=None
            for i in indexes_pred:
                data_n.loc[i,'RMSE_'+str(k)]=\
                    np.sqrt(mean_squared_error(data_n.at[i,'RealPower_pred'][:k],data_n.loc[i,'future_real'][:k]))
            data_n['MAPE_'+str(k)]=None
            for i in indexes_pred:
                data_n.loc[i,'MAPE_'+str(k)]=\
                    mean_absolute_percentage_error(data_n.at[i,'RealPower_pred'][:k],data_n.loc[i,'future_real'][:k])
                
        self.pred=data_n.copy()
        return data_n
             
    
    def prediction_ave_by_prediction_value():
        ...
    
class Prediction_evaluation():
    def __init__(self,abs_path,simple_path,xgb_path,type,name,simple_metrics,scale_coef,group_method,**kwarg):
        #init params
        self.abs_path=abs_path
        self.simple_path = simple_path
        self.xgb_path = xgb_path
        self.type = type
        self.name=name
        self.simple_metrics=simple_metrics
        self.scale_coef=scale_coef
        self.kwarg=kwarg
        self.group_method=group_method
        # init inner vars
        self.metrics_all_dic=None #dict of ave metrics on all data
        self.compare=None
        self.compare_with_metrics=None
        # load data
        self.compare=self.get_comparison_df().copy()
        # calculate metrics for xgb
        self.compare_with_metrics=self.metrics_all().copy()
        #self.plot=self.plot_pred(kwarg)
        ...
        
    def get_comparison_df(self):
        if self.abs_path :
            simple_path=self.simple_path
            xgb_path=self.xgb_path
        else:
            simple_path=self.simple_path+self.type+'_'+self.name+'Simple_pred.csv'
            xgb_path=self.xgb_path+self.type+'_'+self.name+'_XGBoost_prediction.csv'
        try:
            simple=pd.read_csv(simple_path).rename(columns={'RealPower_pred':'Pred_Simple'})
            simple['DateTime']=pd.to_datetime(simple['DateTime'])
            print("Notification: data type of simple predeiction:")
            print(simple['Pred_Simple'].dtype)
            xgb=pd.read_csv(xgb_path).rename(columns={'RealPower_pred':'Pred_XGB'})
            xgb['DateTime']=pd.to_datetime(xgb['DateTime'])
            #print("Notification: data type of XGB predeiction:")
            #print(simple['Pred_XGB'].dtype)
        except:
            print('Fail to load prediction data, check whether the csv file contains columns:"RealPower_pred" ')
            
        # since the loaded xgb prediction may already be scaled down
        # we need to recover it here
        xgb['Pred_XGB']=xgb['Pred_XGB']*self.scale_coef
        df=pd.merge(simple,xgb,on='DateTime',how='left')
        #df=df.T.drop_duplicates().T
        #df.index.name='index'
        #df=df.drop(columns=['DateTime'])
        #df.index.name='DateTime'
        print(df.info())

        #df=df.rename(columns={'RealPower_x':'RealPower'})
        #df=df.drop(columns={"RealPower_y"})
        #df.index = pd.to_datetime(df['DateTime'])
        return df
    
    def metrics_all(self):
        
        def mean_absolute_percentage_error(y_true, y_pred): 
            """Calculates MAPE given y_true and y_pred"""
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        df=self.compare.copy()
        
        # to calculate the average of all values in the columns of Simple metrics
        simple=dict()
        for i in self.simple_metrics:
            #print(i)
            #print(type(i))
            #print(type(df[str(i)]))
            #print(df[str(i)])
            #print(type(df[str(i)].mean()))
            a=df[str(i)].mean()
            simple[str(i)]=a
        #print(df.head(-10))
        df['MSE_XGB']=df[['RealPower','Pred_XGB']].apply(lambda x: np.power(x['RealPower']-x['Pred_XGB'],2)\
            if x['RealPower']!=None and x['Pred_XGB']!= None else None,axis=1)
        
        df['MAE_XGB']=df[['RealPower','Pred_XGB']].apply(lambda x: np.abs(x['RealPower']-x['Pred_XGB'])\
            if x['RealPower']!=None and x['Pred_XGB']!= None else None,axis=1)
        
        if self.type not in ['PV','pv']:
            df['MAPE_XGB']=df[['RealPower','Pred_XGB']].apply(lambda x: (np.abs(x['RealPower']-x['Pred_XGB'])/x['RealPower'])*100\
                if x['RealPower']!=None and x['RealPower']!=0 and x['Pred_XGB']!= None else None,axis=1)
            mape_xgb=df['MAPE_XGB'].mean()
        else:
            mape_xgb='infeasible'
            
        rmse_xgb=np.sqrt(df['MSE_XGB'].mean())
        mae_xgb=df['MAE_XGB'].mean()
        
        df.to_csv('temp.csv')
        
        metric={
            'simple':simple,
            'xgb':{
                'rmse':rmse_xgb,
                'mae':mae_xgb,
                'mape':mape_xgb
            }
        }
        print(metric)
        with open('data.json', 'w') as f:
            json.dump(metric, indent=4, fp=f)
            
        self.metrics_all_dic=metric
        
        return df
    
    def metrics_by_duration(self):
        #if self.compare_with_metrics==None:
        #    raise Exception("Please call metrics_all() first")
        df=self.compare_with_metrics.copy()
        #print(df.info())
        group_method=self.group_method
        assert group_method in ['week','Week','month','Month']
        
        if group_method in ['week','Week']:
            df['week_of_year']=df['DateTime'].dt.isocalendar().week
        elif group_method in ['month','Month']:
            df['month_of_year']=df['DateTime'].dt.isocalendar().month
        
        df_grouped=df.groupby('week_of_year').agg('mean')
        # FIXME: unfinished
        print(df_grouped.head(-10))
        
        return df_grouped
        
    def plot_pred(height,width,df,title,lower,upper,linewidth):
        f, ax = plt.subplots(1)
        f.set_figheight(height)
        f.set_figwidth(width)
        #prev_ylim = ax.get_ylim()
        _ = df[['RealPower','Pred_Simple','Pred_XGB']].plot(ax=ax,
                                                    style=['-','-','-'],
                                                    linewidth=linewidth)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
        ax.set_ylim(lower,upper)
        plot = plt.suptitle(title)
        # ax.set_ylim(prev_ylim) # restore previous ylim
        return plot