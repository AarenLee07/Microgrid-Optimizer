import os
import sys
src_path = sys.path[0].replace("paper_working", "src")
# data_path = sys.path[0].replace("notebooks", "data")
if src_path not in sys.path:
    sys.path.append(src_path)

out_path = sys.path[0].replace("paper_working", "output")

import warnings
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib as mpl
import pandas as pd
import sys
import calendar
import os
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch

######################################################################################

# set default params of plot
rc_={
    "figure.dpi":300,
    "font.size":10,
    "axes.facecolor":"white",
    "savefig.facecolor":"white",
    "text.usetex":True,
}

color_dic_glb={
        'MPC-GT-by_execution':'seagreen',# aquamarine
        'MPC-Prediction-by_execution':'navy',
        'MPC-Heuristic-by_execution':'orange',
        'MPC-Heuristic-minimize_cap':'purple',
        'MPC-Naive-by_execution':'slategray',
        'MSC-GT-by_execution':'gray', #green
        'MPC-Disturbance-by_execution':'purple',
        'MSC-Naive':'grey',
        
        'MPC-DeepAR_optuna-by_execution':'steelblue', 
        'DeepAR_optuna':                 'steelblue', 
        'DeepAR':                        'steelblue', 
        
        'RBC-GT-by_execution':'orange',
        'RBC-GT':             'orange',
        
        'MPC-GT-by_execution':'seagreen', 
        'MPC-GT':             'seagreen', 
        
        'MPC-Simple-by_execution':'yellowgreen',
        'Simple':                 'yellowgreen',
        'Heuristic':              'yellowgreen',
        'ESH':                    'yellowgreen',
        'AMA':                    'yellowgreen',
        
        'MPC-LR_NAIVE-by_execution':'palevioletred', 
        'LR_NAIVE':                 'palevioletred', 
        'LR':                       'palevioletred', 
        
        'MPC-LR_PCo-by_execution':'teal',
        'LR_PCo':                 'teal',
        'LRCo':                 'teal',
        
        'MPC-TFT_optuna-by_execution':'orange', 
        'TFT_optuna':                 'orange', 
        'TFT':                        'orange', 
        
        'MPC-XGB-by_execution':'tomato',
        'XGB':                 'tomato',
        'XGBoost':             'tomato',
        
        
        'U':'steelblue',
        'U-':'orangered',
        'U+':'darkgreen',
        'dc=0':'darkgreen',
        'dc=18$/kWh':'orangered',

        'execution steps=4':'mediumseagreen',
        'execution steps=48':'cornflowerblue',
        'execution steps=96':'mediumvioletred',
        
        'E_1':'black',
        'E_2':'black',
        'E_3':'black',
        'E_4':'black',
        'E_48':'black',
        'E_96':'black'
    }

color_dic_glb_w={
        'MPC-GT-by_execution':'w',# aquamarine
        'MPC-Prediction-by_execution':'w',
        'MPC-Heuristic-by_execution':'w',
        'MPC-Naive-by_execution':'w',
        'MPC-Heuristic-minimize_cap':'w',
        'MSC-GT-by_execution':'w', #green
        'MPC-Disturbance-by_execution':'w',
        'MSC-Naive-by_execution':'w',
        
        'MPC-DeepAR_optuna-by_execution':'w', 
        'RBC-GT-by_execution':'w',
        'MPC-GT-by_execution':'w', 
        'MPC-Simple-by_execution':'w',
        'MPC-LR_NAIVE-by_execution':'w', 
        'MPC-LR_PCo-by_execution':'w',
        'MPC-TFT_optuna-by_execution':'w', 
        'MPC-XGB-by_execution':'w'
    }

marker_dic_glb_w={
        'MPC-GT-by_execution':'_',
        'MPC-Prediction-by_execution':'^',
        'MPC-Heuristic-by_execution':'x',
        'MPC-Naive-by_execution':'v',
        'MPC-Heuristic-minimize_cap':'o',
        'MPC-Disturbance-by_execution':'+',
        'MSC-GT-by_execution':'_',
        'MSC-Naive-by_execution':'_',
        
        'MPC-DeepAR_optuna-by_execution':'v', 
        'DeepAR_optuna':                 'v', 
        'RBC-GT-by_execution':'_',
        'RBC-GT':             '_',
        'MPC-GT-by_execution':'_', 
        'MPC-GT':             '_', 
        'MPC-Simple-by_execution':'o',
        'Simple':                 'o',
        'MPC-LR_NAIVE-by_execution':'purple', 
        'LR_NAIVE':                 'purple', 
        'MPC-LR_PCo-by_execution':'X',
        'LR_PCo':                 'X',
        'MPC-TFT_optuna-by_execution':'*', 
        'TFT_optuna':                 '*', 
        'MPC-XGB-by_execution':'P',
        'XGB':                 'P',
    }

marker_dic_glb={
        'MPC-GT-by_execution':'_',
        'MPC-Prediction-by_execution':'_',
        'MPC-Heuristic-by_execution':'x',
        'MPC-Naive-by_execution':'_',
        'MPC-Disturbance-by_execution':'_',
        'MPC-Heuristic-minimize_cap':'o',
        'MSC-GT-by_execution':'_',
        'MSC-Naive-by_execution':'_',
        
        'MPC-DeepAR_optuna-by_execution':'^', 
        'RBC-GT-by_execution':'_',
        'MPC-GT-by_execution':'_', 
        'MPC-Simple-by_execution':'x',
        'MPC-LR_NAIVE-by_execution':'o', 
        'MPC-LR_PCo-by_execution':'+',
        'MPC-TFT_optuna-by_execution':'v', 
        'MPC-XGB-by_execution':'s'
    }
marker_s_dict={
        'MPC-GT-by_execution':1,
        'MPC-Prediction-by_execution':1,
        'MPC-Heuristic-by_execution':0.3,
        'MPC-Naive-by_execution':1,
        'MPC-Disturbance-by_execution':1,
        'MPC-Heuristic-minimize_cap':0.3,
        'MSC-GT-by_execution':1,
        'MSC-Naive-by_execution':1,
        
        'MPC-DeepAR_optuna-by_execution':0.3, 
        'RBC-GT-by_execution':1,
        'MPC-GT-by_execution':1, 
        'MPC-Simple-by_execution':0.3,
        'MPC-LR_NAIVE-by_execution':0.3, 
        'MPC-LR_PCo-by_execution':0.3,
        'MPC-TFT_optuna-by_execution':0.3, 
        'MPC-XGB-by_execution':0.3
    }
legend_dict={
        'MPC-GT-by_execution':'MPC-GT',
        'MPC-Prediction-by_execution':'MPC-Prediction',
        'MPC-Heuristic-by_execution':'track real',
        'MPC-Naive-by_execution':'MPC-Naive',
        'MPC-Disturbance-by_execution':'MPC-Arti-Noise',
        'MPC-Heuristic-minimize_cap':'track necessary (ours)',
        'MSC-GT-by_execution':'RBC',
        'MSC-Naive-by_execution':'MSC-Naive',
        
        'MPC-DeepAR_optuna-by_execution':'DeepAR', 
        'RBC-GT-by_execution':'RBC',
        'MPC-GT-by_execution':'MPC-GT', 
        'MPC-Simple-by_execution':'AMA',
        'MPC-LR_NAIVE-by_execution':'LR', 
        'MPC-LR_PCo-by_execution':'LRCo',
        'MPC-TFT_optuna-by_execution':'TFT', 
        'MPC-XGB-by_execution':'XGBoost'
        
    }
legend_fs=10
label_fs=12
ticklabel_fs=9
title_fs=14

######################################################################################

def get_merged_df(file_folder=None,log_fn=None,id_exe_unne=None, id_sol_nece=None, id_mini=None):
    log_df=pd.read_excel(log_fn,sheet_name="Sheet1")
    fn_by_exe=log_df[log_df.id==id_exe_unne]["save_fn"].values[0]
    fn_by_sol=log_df[log_df.id==id_sol_nece]["save_fn"].values[0]
    fn_mini=log_df[log_df.id==id_mini]["save_fn"].values[0]
    fn_by_exe=os.path.join(file_folder,fn_by_exe)
    fn_by_sol=os.path.join(file_folder,fn_by_sol)
    fn_mini=os.path.join(file_folder,fn_mini)
    df_by_exe=pd.read_excel(fn_by_exe, sheet_name="op_log",index_col=0)["latest_p_grid_max"].interpolate()
    df_by_sol=pd.read_excel(fn_by_sol, sheet_name="op_log",index_col=0)["latest_p_grid_max"].interpolate()
    df_mini=pd.read_excel(fn_mini, sheet_name="op_log",index_col=0)["latest_p_grid_max"].interpolate()
    df_error=pd.read_excel(fn_by_exe, sheet_name="op_log",index_col=0)["net_load_error"].interpolate()
    
    # Merge the Series into a DataFrame
    df_merged = pd.merge(df_by_exe, df_by_sol, left_index=True, right_index=True)
    df_merged = pd.merge(df_merged, df_mini, left_index=True, right_index=True)
    df_merged = pd.merge(df_merged, df_error, left_index=True, right_index=True)

    # Rename the columns
    df_merged.columns = ['actual_p_max', 'necessary', 'minimized', 'net_load_error']
    df_merged['unnecessary']=df_merged["necessary"]-df_merged["actual_p_max"]
    df_merged['unnecessary_minimized']=df_merged["necessary"]-df_merged["minimized"]
    
    df_merged['latest_max_neg_net_load_error']=0
    mini=0
    for i in df_merged.index:
        if df_merged.loc[i]["net_load_error"]<mini:
            mini=df_merged.loc[i]["net_load_error"]
        df_merged.loc[i,"latest_max_neg_net_load_error"]=mini
        
    
    print(type(df_merged))
    return df_merged

def plot_track_p_max(df_merged,figsize,line_keys=['actual_p_max','necessary','unnecessary'],linewidth=1,month='May',bbox_to_anchor=(1.04, -0.015, 1, 1),
                     plot_error_bar=False,error_bar_width=0.01, ylimit_main=[-150,250],ylimit_sub=[-150,250],plot_error_line=True,
                     inside_start_day=5,inside_days=1,legend_loc="lower right",track_real=False,shadow=False,all_positive=False,
                     xlabel_l=None,xlabel_r=None,ylabel=None,line_clr_dic=None, label_dic=None,
                     save_fn=None,ax=None,axins=None):
        if line_clr_dic==None:
            line_clr_dic={
                    'actual_p_max':'teal',
                    'necessary':'green',
                    'unnecessary':'coral',
                    'unnecessary_minimized':'purple',
                    "latest_max_neg_net_load_error":'red',
                    'accumated_neg':'indianred',     
            }
        if label_dic==None:
            label_dic={
                    'actual_p_max':'Peak demand till t',
                    'necessary':'green',
                    'unnecessary':'discr. w/GT',
                    'unnecessary_minimized':'purple',
                    "latest_max_neg_net_load_error":'Peak error- till t',
                    'accumated_neg':'Normalized acc. error(-) '
                    
            }
        axin_color='steelblue'
        
        if ax==None:
            group_plot=False
            fig, ax = plt.subplots(figsize=figsize)
        else:
            ax=ax
            group_plot=True

        def plot_ax_data(ax,draw_legend=True):
            for key in line_keys:
                
                if key=='unnecessary' and shadow==False:
                    if all_positive == False:
                        ax.plot(df_merged.index, df_merged[key],color=line_clr_dic[key],linewidth=linewidth,label=label_dic[key])
                    else:
                        ax.plot(df_merged.index, df_merged['necessary'],color=line_clr_dic[key],linewidth=linewidth,label=label_dic[key])
                
                elif key=='unnecessary' and shadow==True:
                    if all_positive == False:
                        ax.fill_between(df_merged.index,0,df_merged[key],facecolor='orange',alpha=0.3,label=label_dic[key])
                    else:
                        ax.fill_between(df_merged.index,df_merged['actual_p_max'],df_merged['necessary'],facecolor='orange',alpha=0.3,label=label_dic[key])
                    
                else :
                    ax.plot(df_merged.index, df_merged[key],color=line_clr_dic[key],linewidth=linewidth,label=label_dic[key])
            if plot_error_bar:
                ax2 = ax.twinx()
                flag_red,flag_gray=0,0
                for b in df_merged.index:
                    if df_merged.loc[b,'increasing']==1:
                        color='red'    
                    else: 
                        color='gray'
                    if (flag_red==1)&(color=='red') or (flag_gray==1)&(color=='gray'):
                        ax2.bar(b, df_merged.loc[b,"net_load_error"],width=error_bar_width,color=color,alpha=0.6)
                    else:
                        print('label added')
                        ax2.bar(b, df_merged.loc[b,"net_load_error"],width=error_bar_width,color=color,alpha=0.6,label=label_dic[color])
                        if color=='red':
                            flag_red=1 
                        if color=='gray':
                            flag_gray=1
                        
                ax2.set_ylim(ylimit_main[0]/5,ylimit_main[1]/5)
                #ax2.set_yticklabels(ax2.get_yticklabels(),rotation=0,fontsize=ticklabel_fs)
                ax2.set_yticklabels([])
                
                ax2.legend(fontsize=legend_fs,loc='lower right')
                
                if draw_legend==False:
                    ax2.legend('', frameon=False)
                    ax2.set_yticklabels([])
                    #ax2.set_yticklabels([])
                
            #ax.yaxis.set_major_formatter(lambda x, pos: f'{abs(x):g}')
            ax.margins(x=0.01)
            ax.hlines(y=0,xmin=df_merged.index[0],xmax=df_merged.index[-1],linestyles='--',colors='gray',linewidth=0.6,alpha=0.6)
            ax.tick_params(axis='both', direction='in')
            ax.set_xticklabels(ax.get_xticklabels(),rotation=0,fontsize=ticklabel_fs)
            ax.tick_params(axis='both',which='major',labelsize=ticklabel_fs)
            if draw_legend==False:
                ax.legend('', frameon=False)

        plot_ax_data(ax=ax)
        
        '''
        ax.text(x=0.55,y=0.76,s='$necessary$',ha='center',va='center',transform=ax.transAxes,color='darkgreen')
        if group_plot==False:
            ax.text(x=0.55,y=0.28,s='$unnecessary$',ha='center',va='center',transform=ax.transAxes,color='orange')
        else:
            ax.text(x=0.55,y=0.28,s='$unnecessary$ $(track-real)$',ha='center',va='center',transform=ax.transAxes,color='orange',fontsize=label_fs-2)
        if track_real==True:
            ax.text(x=0.55,y=0.5,s='$unnecessary$',ha='center',va='center',transform=ax.transAxes,color='purple')
        '''
            

        
        
        # set the subplot:
        if axins==None:
            axins = inset_axes(ax, width="60%", height="100%", loc='lower left',
                            bbox_to_anchor=bbox_to_anchor, 
                            bbox_transform=ax.transAxes)
        else:
            axins=axins

        # plot original data
        plot_ax_data(ax=axins,draw_legend=False)

        # set x range
        xlim0 = df_merged.index[inside_start_day*96]
        xlim1 = df_merged.index[inside_start_day*96+inside_days*96]

        # set y range
        axins.set_xlim(xlim0, xlim1)
        axins.set_ylim(ylimit_main)
        axins.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
        axins.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        #axins.xaxis.set_minor_formatter(mdates.DateFormatter('%H'))
        axins.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
        #axins.set_xticklabels(axins.get_xticklabels(),rotation=0)
        axins.set_yticklabels([])
        axins.set_xlabel(xlabel_r,fontsize=label_fs,loc='center')
        
        for loc in ['top','bottom','left','right']:
                axins.spines[loc].set_color(axin_color)  
        axins.tick_params(axis='both',which='major',labelsize=ticklabel_fs)
        axins.tick_params(direction='out',axis='y')
        axins.tick_params(direction='in',axis='x')
        # draw frame in the ori plot
        tx0 = xlim0
        tx1 = xlim1
        ty0 = ylimit_sub[0]
        ty1 = ylimit_sub[1]
        sx = [tx0,tx1,tx1,tx0,tx0]
        sy = [ty0,ty0,ty1,ty1,ty0]
        ax.plot(sx,sy,axin_color,linewidth=0.8)

        # draw connect lines
        xy = (xlim1,ty1)
        xy2 = (xlim0,ylimit_main[1])
        con = ConnectionPatch(xyA=xy2,xyB=xy,coordsA="data",coordsB="data",color=axin_color,
                axesA=axins,axesB=ax,linestyle="--",linewidth=0.5)
        axins.add_artist(con)

        xy = (xlim1,ty0)
        xy2 = (xlim0,ylimit_main[0])
        con = ConnectionPatch(xyA=xy2,xyB=xy,coordsA="data",coordsB="data",color=axin_color,
                axesA=axins,axesB=ax,linestyle="--",linewidth=0.5)
        axins.add_artist(con)
        axins.legend().remove()
        ax.tick_params(direction='out',axis='y')
        
        ax.set_ylim(ylimit_main)
        ax.set_xlabel(xlabel_l,fontsize=label_fs,loc='center')
        ax.set_ylabel(ylabel,fontsize=label_fs)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=24*7))
        ax.xaxis.set_minor_locator(mdates.HourLocator(interval=24))
        ax.tick_params(axis='x', rotation=0)

        ax.set_xticklabels(ax.get_xticklabels(),rotation=0,fontsize=ticklabel_fs)
        ax.legend(frameon=False,loc=legend_loc,fontsize=legend_fs)
        ax.tick_params(axis='both',which='major',labelsize=ticklabel_fs)
        
        # adjust layout and save fig
        if ax==None:
            #plt.tight_layout()
            #plt.savefig(save_fn,bbox_inches='tight')
            plt.show()
        
        return ax,axins
        
        
        
def cal_relative(fn):
    df=pd.read_excel(fn,sheet_name="Sheet1",index_col=0)
    lower_bound=df[(df.strategy=="optimal")&(df.pred_model=="GT")]["OPEX"].values[0]
    upper_bound=df[(df.strategy=="MSC")&(df.pred_model=="GT")]["OPEX"].values[0]
    diff=upper_bound-lower_bound
    df["relative_OPEX"]=(df["OPEX"]-lower_bound)/diff*100
    df["disturbance_MAPE"]=df["disturbance_scale"]*100
    return df

def get_df_for_plot(method,fn_list):
    dfs=[]
    necessaey_cols=["method","strategy","B_kWh","pred_model",
                    "month_of_year","p_grid_max","price_dc",
                    "disturbance_rule","disturbance_MAPE","p_grid_max_method",
                    "OPEX","tou_cost","demand_charge","relative_OPEX"]
    for fn in fn_list:
        df=pd.DataFrame(cal_relative(fn),columns=necessaey_cols)
        dfs.append(df)
        
    for i in range(len(dfs)):
        df=dfs[i]
        rule_list=df["disturbance_rule"].unique()
        rule_list=rule_list[pd.notnull(rule_list)]
        for rule in rule_list:
            a=df[(df.disturbance_rule==rule)&(df.disturbance_MAPE==0.1)].copy()
            for k in ["OPEX","tou_cost","demand_charge","relative_OPEX"]:
                a[k]=[df[(df.strategy=="optimal")&(df.pred_model=="GT")][k].values[0]]*len(a)
            a["disturbance_MAPE"]=[0]*len(a)
            df=df._append(a).copy()
        dfs[i]=df.copy()
        
    df_concated=pd.concat(dfs).drop_duplicates().reset_index(drop=True)
    df_concated["disturbance_MAPE"] = df_concated["disturbance_MAPE"].apply(lambda x: round(x, 1))
    df_by_method=df_concated.groupby(by='p_grid_max_method')

    df_by_exe=df_by_method.get_group(method).groupby(by="disturbance_rule")
    df_uniform=df_by_exe.get_group("uniform")
    df_uniform_neg=df_by_exe.get_group("uniform_neg")
    df_uniform_pos=df_by_exe.get_group("uniform_pos")

    mape=sorted(df_uniform["disturbance_MAPE"].unique())
    y_uniform=list()
    y_uniform_neg=list()
    y_uniform_pos=list()
    y_uniform_mean=list()
    y_uniform_neg_mean=list()
    y_uniform_pos_mean=list()
    for m in mape:
        u=np.array(df_uniform[df_uniform.disturbance_MAPE==m]["relative_OPEX"].values)
        n=np.array(df_uniform_neg[df_uniform_neg.disturbance_MAPE==m]["relative_OPEX"].values)
        p=np.array(df_uniform_pos[df_uniform_pos.disturbance_MAPE==m]["relative_OPEX"].values)
        y_uniform.append(u)
        y_uniform_pos.append(p)
        y_uniform_neg.append(n)
        y_uniform_mean.append(np.nanmean(u))
        y_uniform_pos_mean.append(np.nanmean(p))
        y_uniform_neg_mean.append(np.nanmean(n))
    y_dic={
        'U':y_uniform,
        'U-':y_uniform_neg,
        'U+':y_uniform_pos
    }
    mean_dic={
        'U':y_uniform_mean,
        'U-':y_uniform_neg_mean,
        'U+':y_uniform_pos_mean
    }
    return mape,y_dic,mean_dic

def cluster_box_plot(figsize,
                     mape,y_dic,mean_dic,ylimit=(-10,180),plot_noise=True,show_yaxis=True,
                     y_dic_self_define=None,self_define_position=True,markersize_raw=None,
                     mean_dic_old=None,ax=None,vol=True,x_label='Default_label',
                     plot_line_new=True, plot_line_old=False, save_fn=None):
    if vol ==True:
        for dic in [y_dic,mean_dic,mean_dic_old]:
            if dic!=None:
                for i in dic:
                    for k in range(len(dic[i])):
                        dic[i][k]=-dic[i][k]+100
        if y_dic_self_define!=None:
            for key in y_dic_self_define.keys():
                for i in range(len(y_dic_self_define[key][0])):
                    y_dic_self_define[key][0][i][3]=-y_dic_self_define[key][0][i][3]+100
        #ylimit=(-100,120)
    
    if plot_line_old==True:
        assert mean_dic_old is not None
    if ax!=None:
        ax=ax
    else:
        fig, ax=plt.subplots(figsize=figsize)
    positions1=np.array(range(1,len(mape)+1))-figsize[0]/32
    positions2=np.array(range(1,len(mape)+1))
    positions3=np.array(range(1,len(mape)+1))+figsize[0]/32
    
    # deprecated
    '''color_dic={
        'U':'steelblue',
        'U-':'orangered',
        'U+':'darkgreen',
        'dc=0':'darkgreen',
        'dc=18$/kWh':'orangered',
        "XGBoost":'salmon',
        "LR-PCo":'crimson',
        "TFT":'blueviolet',
        "LR":'chocolate',
        #"TFT_NAIVE",
        "DeepAR":'teal',
        "Heuristic":'dodgerblue',
        #'RF_NAIVE',
        'execution steps=4':'mediumseagreen',
        'execution steps=48':'cornflowerblue',
        'execution steps=96':'slateblue',
    }'''
    alpha_dic={
        'U':0.15,
        'U-':0.15,
        'U+':0.15,
        'dc=0':0.4,
        'dc=18$/kWh':0.4,
        "XGBoost":0.6,
        "LR-PCo":0.5,
        "LRCo":0.5,
        "TFT":0.5,
        "LR":0.6,
        #"TFT_NAIVE",
        "DeepAR":0.7,
        "ESH":0.7,
        "AMA":0.7,
        "Heuristic":0.7,
        #'RF_NAIVE',
        'execution steps=4':0.7,
        'execution steps=48':0.7,
        'execution steps=96':0.7,
    }
    width_dic={
        'U':0.5,
        'U-':0.5,
        'U+':0.5,
        'dc=0':0,
        'dc=18$/kWh':0,
        "XGBoost":1,
        "LR-PCo":1,
        "LRCo":1,
        "TFT":1,
        "LR":1,
        "Heuristic":1,
        #"TFT_NAIVE",
        "DeepAR":1,
        "ESH":1,
        "AMA":1,
        #'RF_NAIVE',
        'execution steps=4':0.7,
        'execution steps=48':0.7,
        'execution steps=96':0.7,
    }
    pos_dic={
        'U':positions1,
        'U-':positions2,
        'U+':positions3,
        'dc=0':positions1,
        'dc=18$/kWh':positions3,
        'execution steps=4':positions1,
        'execution steps=48':positions2,
        'execution steps=96':positions3,

    }
    used_keys=[]

    boxs=[]
    if self_define_position==False: 
        for key in y_dic.keys():
            used_keys.append(key)
            bplot=ax.boxplot(y_dic[key],positions=pos_dic[key],patch_artist=True,showmeans=True,widths=figsize[0]/40,        
                    boxprops={"facecolor": color_dic_glb[key],
                            "edgecolor": "w",
                            "linewidth": 0,
                            'alpha':0.4},
                    medianprops={"color": color_dic_glb[key], "linewidth": 0},
                    meanprops={'marker':'+',
                            'markerfacecolor':color_dic_glb[key],
                            'markeredgecolor':color_dic_glb[key],
                            'markersize':4.1*figsize[0]/8},
                    sym="",showfliers=True, showcaps=False,
                    whiskerprops={'color': 'w', 'linewidth': 1, 'linestyle': '--', 'alpha':0},
                    )
            if plot_line_new==True:
                lplot=ax.plot(pos_dic[key],mean_dic[key],linestyle='-',color=color_dic_glb[key],alpha=0.2,linewidth=1)
            if plot_line_old==True:
                lplot=ax.plot(pos_dic[key],mean_dic_old[key],linestyle='--',marker='+',color=color_dic_glb[key],
                            alpha=0.5,linewidth=0.8,markersize=0*figsize[0]/8)
            ax.set_xticks([])
            ax.set_xticklabels([])
            boxs.append(bplot['boxes'][0])
            
        for i in range(len(positions2)):
            if i%2==0:
                diff=positions2[1]-positions2[0]
                l=positions2[i]-diff/2
                r=positions2[i]+diff/2
                ax.fill_betweenx(ylimit, l,r, facecolor='dimgray', alpha=0.05)
            
    if self_define_position==True:  
        assert y_dic_self_define!=None
        labels=[]
        positions=[]
        minor=[]
        
        f_count=0
        for key in y_dic_self_define.keys():
            labels.append(str('%.3g' % key))
            positions.append(y_dic_self_define[key][2])
            for i in range(len(y_dic_self_define[key][0])):
                values=np.array(y_dic_self_define[key][0][i][3])
                position=y_dic_self_define[key][0][i][0]
                c_key=y_dic_self_define[key][0][i][2]
                width=y_dic_self_define[key][0][i][1]
                
                if (plot_noise==False):
                    if c_key in ['U','U+','U-']:
                        continue
                    position-=width*2
                    width*=2
                if markersize_raw==None:
                    markersize=width*figsize[0]*50/len(y_dic_self_define.keys())
                else:
                    markersize=markersize_raw
                marker='+'
                if c_key in ['XGBoost','LRCo','LR-PCo','TFT','LR','DeepAR','ESH','AMA']:
                    width=width/5
                    markersize=markersize/1.5
                    marker='o'
                else:
                    width=y_dic_self_define[key][0][i][1]
                used_keys.append(c_key)
                #print(values,position,c_key,width)
                bplot=ax.boxplot([values],positions=[position],patch_artist=True,showmeans=True,
                                widths=[width],        
                        boxprops={"facecolor": color_dic_glb[c_key],
                                "edgecolor": color_dic_glb[c_key],
                                "linewidth": width_dic[c_key],
                                'alpha':alpha_dic[c_key]},
                        medianprops={"color": color_dic_glb[c_key], "linewidth": 0},
                        meanprops={'marker':marker,
                                'markerfacecolor':color_dic_glb[c_key],
                                'markeredgecolor':color_dic_glb[c_key],
                                'markersize':markersize},
                        sym="",showfliers=True, showcaps=False,
                        whiskerprops={'color': 'w', 'linewidth': 1, 'linestyle': '--', 'alpha':0},
                        )
                
            if f_count%2==0 or f_count>12:
                ax.fill_betweenx(ylimit, y_dic_self_define[key][1],y_dic_self_define[key][3], 
                                 facecolor='dimgray', alpha=0.05)
            f_count+=1
            '''    
            if plot_line_new==True:
                lplot=ax.plot(pos_dic[key],mean_dic[key],linestyle='-',color=color_dic[key],alpha=0.2,linewidth=0.5)
            if plot_line_old==True:
                lplot=ax.plot(pos_dic[key],mean_dic_old[key],linestyle='--',marker='+',color=color_dic[key],
                            alpha=0.5,linewidth=0.8,markersize=0*figsize[0]/8)'''
        
        #ax.set_xticks([])
        #ax.set_xticklabels([])
        ax.set_xticks(ticks=positions,minor=minor,labels=labels,fontsize=ticklabel_fs)
        boxs.append(bplot['boxes'][0])
        
    if vol ==True:
        ax.axhline(y=100,dashes=(5, 8),xmin=0,xmax=1,color='gray',linewidth=0.7,alpha=0.4, zorder=1)
        ax.axhline(y=0,dashes=(5, 8),xmin=0,xmax=1,color='gray',linewidth=0.7,alpha=0.4, zorder=2)

    if self_define_position==False:
        labels=[]
        positions=[]
        minor=[]

        for i in range(len(mape)):
            try:
                labels.append(str('%.3g' % mape[i]))
            except:
                labels.append(mape[i])
            positions.append(positions2[i])
            minor.append(False)
        ax.set_xticks(ticks=positions,minor=minor,labels=labels,fontsize=ticklabel_fs)
    
    
    ax.tick_params(direction='in', axis="x", which='both')

    ax.set_ylim(ylimit)
    # Set major locator
    #ax.yaxis.set_major_locator(plt.MultipleLocator(20))

    # Set minor locator
    #ax.yaxis.set_minor_locator(plt.MultipleLocator(10))

    # Set y-tick labels font size
    ax.tick_params(axis='y', labelsize=9)
    ax.tick_params(axis='x', labelsize=9)
    #ax.set_xlabel('MAPE (%)',loc='left',fontsize=label_fs)
    #ax.set_xlabel('Given ground truth (steps)',loc='left',fontsize=label_fs)
    


    legend_elements=[]
    used_keys=list(set(used_keys))
    for key in color_dic_glb:
        if key in used_keys:
            if plot_noise==True:
                if key not in ['XGBoost','LRCo','LR-PCo','TFT','LR','DeepAR','ESH','Heuristic','AMA']:
                    legend_elements.append(Patch(facecolor=color_dic_glb[key], edgecolor=color_dic_glb[key],
                                        label=key,alpha=alpha_dic[key],linewidth=width_dic[key]))
            else:
                legend_elements.append(Patch(facecolor=color_dic_glb[key], edgecolor=color_dic_glb[key],
                                        label=key,alpha=alpha_dic[key],linewidth=width_dic[key]))
    #if plot_line_new==True:
    #legend_elements.append(Line2D([0], [0], marker='+', color='gray', label='Mean',linewidth=0,
    #        markerfacecolor='gray', markersize=8))
    
        #legend_elements.append(Line2D([0], [0], marker='+', color='gray', label='Median',linewidth=0,
                               # markerfacecolor='gray', markersize=0))
    if plot_line_old==True:
        legend_elements.append(Line2D([0], [0], marker='+', color='gray', label='track real',linewidth=0.6,linestyle='--',
                            markerfacecolor='gray', markersize=0))
    if vol==True:
        ax.legend(handles=legend_elements,loc='lower left',fontsize=legend_fs) 
        ax.set_ylabel("Vol* (%)", fontsize=label_fs, labelpad=0)
    else:
        ax.set_ylabel("Relative regret (%)", fontsize=label_fs)
        ax.legend(handles=legend_elements,loc='upper left',fontsize=legend_fs) 
    #ax.set_xlabel("MAPE (%)", loc='center',fontsize=label_fs)
    ax.set_xlabel(x_label,loc='center',fontsize=label_fs)
    
    if show_yaxis==False:
        ax.set_ylabel(None)
        ax.set_yticklabels([])
    if save_fn is not None:
        plt.savefig(save_fn)
        
    return ax


# libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib as mpl
import pandas as pd
import sys
import calendar
from matplotlib.lines import Line2D

'''
src_path = sys.path[0].replace("figures\visualization\script", "src")
#replace notebook as scripts
data_path = sys.path[0].replace("figures\visualization\script", "data")
if src_path not in sys.path:
    sys.path.append(src_path)

out_path = sys.path[0].replace("figures\visualization\script", "output")
'''



def hex_to_RGB(hex_str):
    """ #FFFFFF -> [255,255,255]"""
    #Pass 16 to the integer function for change of base
    return [int(hex_str[i:i+2], 16) for i in range(1,6,2)]

def get_color_gradient(c1, c2, n):
    """
    Given two hex colors, returns a color gradient
    with n colors.
    """
    assert n > 1
    c1_rgb = np.array(hex_to_RGB(c1))/255
    c2_rgb = np.array(hex_to_RGB(c2))/255
    mix_pcts = [x/(n-1) for x in range(n)]
    rgb_colors = [((1-mix)*c1_rgb + (mix*c2_rgb)) for mix in mix_pcts]
    return ["#" + "".join([format(int(round(val*255)), "02x") for val in item]) for item in rgb_colors]

def gradient_image(ax, direction=0.3, cmap_range=(0, 1), **kwargs):
    """
    Draw a gradient image based on a colormap.

    Parameters
    ----------
    ax : Axes
        The axes to draw on.
    direction : float
        The direction of the gradient. This is a number in
        range 0 (=vertical) to 1 (=horizontal).
    cmap_range : float, float
        The fraction (cmin, cmax) of the colormap that should be
        used for the gradient, where the complete colormap is (0, 1).
    **kwargs
        Other parameters are passed on to `.Axes.imshow()`.
        In particular, *cmap*, *extent*, and *transform* may be useful.
    """
    phi = direction * np.pi / 2
    v = np.array([np.cos(phi), np.sin(phi)])
    X = np.array([[v @ [1, 0], v @ [1, 1]],
                  [v @ [0, 0], v @ [0, 1]]])
    a, b = cmap_range
    X = a + (b - a) / X.max() * X
    im = ax.imshow(X, interpolation='bicubic', clim=(0, 1),
                   aspect='auto', **kwargs)
    return im

# process the original data
def pre_process(df,key,duration_key):
    df.set_index('id')
    
    df=df[['strategy','pred_method',duration_key,key,'days','p_grid_max_method']]
    df=df.replace('optimal',"MPC")
    df=df.replace("Simple","Heuristic")
    #df=df.replace("Simple-our-method","Heuristic-our-method")
    df['label']=df['strategy']+'-'+df['pred_method']+'-'+df['p_grid_max_method']
    data=dict()
    
    for i in df[duration_key].unique():
        try:
            for label in df['label'].unique():
                data[label]=df[((df[duration_key]==i)) & (df['label']==label)][key].values[0]
            
            upper=df[((df[duration_key]==i)) & (df['label']=='MSC-GT-by_execution')][key].values[0]
            lower=df[((df[duration_key]==i)) & (df['label']=='MPC-GT-by_execution')][key].values[0]
            '''
            #assert upper>lower
            prediction=df[((df['week_of_year']==i)) & (df['label']=='MPC-Prediction')][key].values[0]
            heuristic=df[((df['week_of_year']==i)) & (df['label']=='MPC-Heuristic')][key].values[0]
            naive=df[((df['week_of_year']==i)) & (df['label']=='MPC-Naive')][key].values[0]
            '''
        except:
            pass
        
        upper_id=df[((df[duration_key]==i)) & (df['label']=='MSC-GT-by_execution')][key].index
        lower_id=df[((df[duration_key]==i)) & (df['label']=='MPC-GT-by_execution')][key].index
        
        for label in df["label"].unique():
            data[label+'_id']=df[((df[duration_key]==i)) & (df['label']==label)][key].index
        '''
        prediction_id=df[((df['week_of_year']==i)) & (df['label']=='MPC-Prediction')][key].index
        heuristic_id=df[((df['week_of_year']==i)) & (df['label']=='MPC-Heuristic')][key].index
        naive_id=df[((df['week_of_year']==i)) & (df['label']=='MPC-Naive')][key].index 
        '''
        
        new_key='relative_'+key
        df.loc[lower_id,new_key]=0
        df.loc[upper_id,new_key]=100
        
        for label in df["label"].unique():
            df.loc[data[label+'_id'],new_key]=(data[label]-lower)/(upper-lower)*100
        '''    
        df.loc[prediction_id,new_key]=(prediction-lower)/(upper-lower)*100
        df.loc[heuristic_id,new_key]=(heuristic-lower)/(upper-lower)*100
        df.loc[naive_id,new_key]=(naive-lower)/(upper-lower)*100
        '''
    #df_new=df.drop(df[df['label']=='MPC-Naive'].index)
    print(df)
    df=df.reindex()
    invalid_duration_list=df[(df[new_key]>100)|(df[new_key]<0)|(df[key].isna())].index
    invalid_duration_list=df.iloc[invalid_duration_list][duration_key].unique()
    df['is_valid']=df[duration_key].apply(lambda x: True if x not in invalid_duration_list else False)
    if key=='OPEX':
        df['OPEX']=df['OPEX']*df['days']/1000
    return df

def plot_origin(df,key,relative,save_fn,fontsize,ylimit,duration_key):
    if relative:
        new_key='relative_'+key
    else:
        new_key=key
    color_dict=color_dic_glb
    fig,ax=plt.subplots(figsize=(20,5))

    scatter_x=np.array(df[duration_key])
    scatter_y=np.array(df[new_key])
    group=np.array(df['label'])
    #for i in range(len(df)):  
    for g in np.unique(group):
        i = np.where(group == g)
        ax.scatter(scatter_x[i], scatter_y[i], label=g,c=color_dict[g],\
            marker='_',s=250)
    ax.set_ylim(ylimit)
    #ax.xaxis.set_major_formatter
    ax.legend(fontsize=legend_fs)
    plt.grid(axis = 'x')
    plt.tight_layout()

    if save_fn is not None:
        plt.savefig(save_fn)
        
    plt.show()
        
def plot_valid(df,relative,limit,figsize,save_fn,key,fontsize,duration_key):
    
    if relative:
        new_key='relative_'+key
    else:
        new_key=key

    df_valid=df.drop(df[df.is_valid==False].index)

    x_coor=np.array([df_valid[df_valid.label=='MPC-Prediction-by_execution'][duration_key],\
        df_valid[df_valid.label=='MPC-Prediction-by_execution'][duration_key]])

    y_coor=np.array([df_valid[df_valid.label=='MPC-Heuristic-by_execution'][new_key],\
        df_valid[df_valid.label=='MPC-Prediction-by_execution'][new_key]])

    color_dict=color_dic_glb
    maker_dict=marker_dic_glb

    fig,ax=plt.subplots(figsize=figsize)

    scatter_x=np.array(df_valid[duration_key])
    scatter_y=np.array(df_valid[new_key])
    is_valid=np.array(df_valid['is_valid'])
    group=np.array(df_valid['label'])
    label_x=[]
    for i in scatter_x: 
        label_x.append(duration_key+str(i))

    ax.set_xticks(ticks=scatter_x)
    ax.set_xticklabels(labels=label_x,fontsize=fontsize,rotation=0)

    for g in np.unique(group):
        i = np.where(group == g)
        ax.scatter(scatter_x[i], scatter_y[i], label=g,c=color_dict[g],\
            marker=maker_dict[g],s=250)
    ax.set_ylim(limit)
    ax.legend(loc='upper left',bbox_to_anchor=(1.02,1),fontsize=legend_fs)

    for i in range(len(x_coor[0])):
        ax.bar(x=x_coor[0][i],height=np.abs(y_coor[0][i]-y_coor[1][i]), \
            bottom=min(y_coor[0][i],y_coor[1][i]),color='lightsteelblue',width=0.6,alpha=0.3)
    ax.set_xlabel(duration_key+" 2019",fontsize=fontsize*1.5)
    ax.set_ylabel("Relative "+key+" (Percentage)",fontsize=fontsize*1.5)


    ax.set_title("Relative "+key+" of Different Models",fontsize=fontsize*1.5)
    ax.text(s="Notes: 1. "+key+" under MPC-GT is marked as lower bound while MSC_GT marked as upper bound \n  \
                    \t   2. Data sets which don't meet our expection that MPC-GT is upper bound and MSC-GT is lower bound were removed.",
            fontsize=fontsize*1.2,
            x=0,
            y=-40
            )
    #ax.grid(True)
    plt.grid(axis = 'x',linestyle='--',alpha=0.1)
    plt.grid(axis = 'y',linestyle='--',alpha=0.8)

    plt.tight_layout()
    
    if save_fn is not None:
        plt.savefig(save_fn)
        
    plt.show()
    
def plot_origin_valid_bar(df,relative,limit,figsize,key,save_fn,fontsize,bbox_to_anchor,
                          notes_y=-40,duration_key='week_of_year',show_notes=False):
    if relative:
        new_key='relative_'+key
    else:
        new_key=key
    df_valid=df.drop(df[df.is_valid==False].index)

    x_coor=np.array([df_valid[df_valid.label=='MPC-Prediction'][duration_key],\
        df_valid[df_valid.label=='MPC-Prediction'][duration_key]])

    y_coor=np.array([df_valid[df_valid.label=='MPC-Heuristic'][new_key],\
        df_valid[df_valid.label=='MPC-Prediction'][new_key]])
    
    color_dict=color_dic_glb
    maker_dict=marker_dic_glb

    fig,ax=plt.subplots(figsize=figsize)
    
    plt.grid(axis = 'x',linestyle='--',alpha=0.1)
    plt.grid(axis = 'y',linestyle='--',alpha=0.8)
    
    scatter_x=np.array(df[duration_key])
    scatter_y=np.array(df[new_key])

    group=np.array(df['label'])
    label_x=[]
    for i in scatter_x: 
        if duration_key=='month_of_year':
            label_x.append(calendar.month_abbr[i])
        else:
            label_x.append(duration_key+str(i))
    ax.set_xticks(ticks=scatter_x)
    ax.set_xticklabels(labels=label_x,fontsize=fontsize,rotation=0)

    for g in np.unique(group):
        i = np.where(group == g)
        ax.scatter(scatter_x[i], scatter_y[i], label=g,c=color_dict[g],\
            marker=maker_dict[g],s=250)
    ax.set_ylim(limit)
    
    def min_help(a,b):
        for i in range(len(a)):
            if a[i]>b[i]:
                a[i] = b[i]
        return a
    
    if relative:
        #for i in range(len(x_coor[0])):
        ax.bar(x=x_coor[0],height=np.abs(y_coor[0]-y_coor[1]), \
            bottom=min_help(y_coor[0],y_coor[1]),
            color='lightsteelblue',width=0.6,alpha=0.3,label="Prediction<Heuristic")
        
    if not relative:
        x_coor_MPC_GT=np.array([df[df.label=='MPC-GT'][duration_key],\
            df[df.label=='MPC-GT'][duration_key]])
        y_coor_MPC_GT=np.array([[0]*len(df[df.label=='MPC-GT'][new_key]),\
            df[df.label=='MPC-GT'][new_key]])
        #for i in range(len(x_coor_MPC_GT[0])):
        ax.bar(x=x_coor_MPC_GT[0],height=np.abs(y_coor_MPC_GT[0]-y_coor_MPC_GT[1]), \
            bottom=y_coor_MPC_GT[0],color='seagreen',
            width=0.6,alpha=0.1,label="MPC_GT")
            
    ax.legend(loc='upper left',bbox_to_anchor=bbox_to_anchor,fontsize=legend_fs)
        
    
    ax.set_xlabel(duration_key+" 2019",fontsize=fontsize*1.5)
    ax.set_ylabel("Relative "*relative+key+" (Percentage)"*relative+"(US dollar/day)"*(not relative),fontsize=fontsize*1.5)

    ax.set_title("Relative "*relative+key+" of Different Models",fontsize=fontsize*1.5)
    if relative*show_notes:
        ax.text(s="Notes: 1. "+key+" under MPC-GT is marked as lower bound while MSC_GT marked as upper bound.",
                fontsize=fontsize*1.2, x=0, y=notes_y
                )
    
    plt.tight_layout()
    
    if save_fn is not None:
        plt.savefig(save_fn)
    plt.show()
    
    return ax
  
def mplot_origin_valid_bar(params):
    
    
    def min_help(a,b):
        for i in range(len(a)):
            if a[i]>b[i]:
                a[i] = b[i]
        return a
    if params['group_plot']==None:
        fig, axs = plt.subplots(nrows=1, ncols=params["n_subplots"],
                                figsize=params["figsize"],
                                sharey=params["sharey"])
    else:
        axs=params['group_plot']
    fontsize=params["fontsize"]
    relative=params["relative"]
    marker_s=params["marker_s"]
    
    for i in range(params["n_subplots"]):
        i=str(i)
        
        key=params["subplots"][i]["key"]
        df=params["subplots"][i]["df"]
        if params["labels_not_show"] != None:
            df=df.drop(df[df["label"].isin(params["labels_not_show"])].index)
        limit=params["subplots"][i]["limit"]  
        duration_key=params["subplots"][i]["duration_key"]
        subtitle=params["subplots"][i]["subtitle"]
        is_gradient=params["subplots"][i]["gradient"]
        plot_arrow=params["subplots"][i]["plot_arrow"]
        if plot_arrow==True:
            arrow_end=params["subplots"][i]["arrow_end"]
            arrow_start=params["subplots"][i]["arrow_start"]
        i=int(i)
        
        if is_gradient:
            gd_params=params["subplots"][str(i)]["gradient_params"]
            # cal the gradient display range
            extent_l=(gd_params["cmap_range"][0]*100-limit[0])/(limit[1]-limit[0])
            extent_h=(gd_params["cmap_range"][1]*100-limit[0])/(limit[1]-limit[0])
            extent=(0.01,0.99,extent_l,extent_h)
            gradient_image(axs[i], direction=0, extent=extent, transform=axs[i].transAxes,
               cmap=plt.colormaps[gd_params["cmap_name"]], cmap_range=gd_params["cmap_range"], alpha=gd_params["alpha"])
            color_dict=color_dic_glb_w
            maker_dict=marker_dic_glb_w

        else:
            color_dict=color_dic_glb
            maker_dict=marker_dic_glb
            
        if relative:
            new_key='relative_'+key
            try:
                arrow_start='relative_'+arrow_start
                arrow_end='relative_'+arrow_end
            except:
                ...
        else:
            new_key=key
        df_valid=df#.drop(df[df.is_valid==False].index)

        #axs[i].grid(axis = 'x',linestyle='--',alpha=0.1)
        #axs[i].grid(axis = 'y',linestyle='--',alpha=0.8)
        
        scatter_x=np.array(df[duration_key])
        scatter_y=np.array(df[new_key])
        
        if plot_arrow==True:
            arrawx=np.array(df_valid[df_valid.label==arrow_start][duration_key])
            arrowy_s=np.array(df_valid[df_valid.label==arrow_start][new_key])
            arrowy_e=np.array(df_valid[df_valid.label==arrow_end][new_key])
            print(range(len(arrawx)))
            for k in range(len(arrawx)):
                axs[i].arrow(arrawx[k], arrowy_s[k], 0, (arrowy_e[k]-arrowy_s[k])*0.99,
                            width=0.005*limit[1]/550,color='dimgray',alpha=0.4,head_width=0.2,head_length=6*limit[1]/550,
                            length_includes_head=True)

        group=np.array(df['label'])
        if params["show_line"]:
            for label in group:
                x=np.array(df_valid[df_valid.label==label][duration_key])
                y=np.array(df_valid[df_valid.label==label][new_key])
                axs[i].plot(x, y, color='white', linestyle='dashed',
                    linewidth=0.3, markersize=0,alpha=0.3)
        
        if params['group_plot']==None:
            label_x=[]
            for k in scatter_x: 
                if duration_key=='month_of_year':
                    label_x.append(calendar.month_abbr[k])
                else:
                    label_x.append(duration_key+str(k))
            axs[i].set_xticks(ticks=scatter_x)
            axs[i].set_xticklabels(labels=label_x,fontsize=ticklabel_fs,rotation=0)
        else:
            axs[i].set_xticks(ticks=scatter_x)
            axs[i].set_xticklabels(labels=scatter_x,fontsize=ticklabel_fs,rotation=0)
        
        if is_gradient:
            for g in np.unique(group):
                maker_dict=marker_dic_glb_w
                m = np.where(group == g)
                axs[i].scatter(scatter_x[m], scatter_y[m], label=legend_dict[g],c=color_dict[g],\
                    marker=maker_dict[g],s=marker_s_dict[g]*marker_s)
        else:
            for g in np.unique(group):
                m = np.where(group == g)
                axs[i].scatter(scatter_x[m], scatter_y[m], label=legend_dict[g],c=color_dict[g],\
                    marker=maker_dict[g],s=marker_s_dict[g]*marker_s)
            
        axs[i].set_ylim(limit)
        
        if relative :
            axs[i].axhline(y=100,linewidth=1,color='green',linestyle='--')
            axs[i].axhline(y=0,linewidth=1,color='orange',linestyle='--')
            if ('MPC-Prediction' in params["labels_not_show"]) or\
                ('MPC-Heuristic'in params["labels_not_show"]):
                    pass
            else:
                ...
                '''
                x_coor=np.array([df_valid[df_valid.label=='MPC-Prediction-by_execution'][duration_key],\
                    df_valid[df_valid.label=='MPC-Prediction-by_execution'][duration_key]])
                y_coor=np.array([df_valid[df_valid.label=='MPC-Heuristic-by_execution'][new_key],\
                    df_valid[df_valid.label=='MPC-Prediction-by_execution'][new_key]])
                axs[i].bar(x=x_coor[0],height=np.abs(y_coor[0]-y_coor[1]), \
                    bottom=min_help(y_coor[0],y_coor[1]),
                    color='lightsteelblue',width=0.6,alpha=0.3,label="Prediction<Heuristic")'''
            
        if not relative:
            x_coor_MPC_GT=np.array([df[df.label=='MPC-GT-by_execution'][duration_key],\
                df[df.label=='MPC-GT-by_execution'][duration_key]])
            y_coor_MPC_GT=np.array([[0]*len(df[df.label=='MPC-GT-by_execution'][new_key]),\
                df[df.label=='MPC-GT-by_execution'][new_key]])
            axs[i].bar(x=x_coor_MPC_GT[0],height=np.abs(y_coor_MPC_GT[0]-y_coor_MPC_GT[1]), \
                bottom=y_coor_MPC_GT[0],color='seagreen',
                width=0.6,alpha=0.1,label="MPC_GT")
            
        if duration_key=='month_of_year':
            axs[i].set_xlabel("Month of year (2019)",fontsize=label_fs,loc='center')
        else:
            axs[i].set_xlabel(duration_key,fontsize=label_fs,loc='center')
        axs[i].set_title(subtitle,fontsize=title_fs)
        if key=='grid_max':
            axs[i].set_ylabel("Peak demand (kW)",fontsize=label_fs)
        if key=='OPEX':
            #axs[i].set_ylabel("OPEX (k\$)",fontsize=label_fs)
            axs[i].set_ylabel("VoI* (%)",fontsize=label_fs)

        #if (i==0)&(params["save_fn"]==None):
        #    axs[i].set_ylabel("Relative "*relative+key+" (Percentage)"*relative+"(\$/day)"*(not relative),fontsize=label_fs)
        #else:
        #    axs[i].set_ylabel("Relative "*relative+key+" (Percentage)"*relative+"(\$/day)"*(not relative),fontsize=label_fs)
        axs[i].tick_params(axis='both',which='major',labelsize=ticklabel_fs)
        axs[i].tick_params(axis='x',direction="in")
        
        

        
    if relative*params["show_notes"]:
        plt.text(s="Notes: 1. "+key+" under MPC-GT is marked as lower bound while MSC_GT marked as upper bound.",
                fontsize=fontsize*1.2, x=0, y=-40
                )
    plt.suptitle(params["suptitle"],fontsize=fontsize*1.5)
    if params['group_plot']==None:
        plt.tight_layout()
        leg = plt.legend(loc='upper right', bbox_to_anchor=params["bbox_to_anchor"], fontsize=legend_fs)
        if is_gradient:
            for handle in leg.legendHandles:
                handle.set_color('black')
    else:
        leg = axs[0].legend(loc='lower left', bbox_to_anchor=params["bbox_to_anchor"], fontsize=legend_fs)
    
    if params["save_fn"] is not None:
        plt.savefig(params["save_fn"])
    plt.show()
    
    return 
          
def pre_process_dc_line(df):
    df=df.rename(columns={"grid_max":"mpc_grid_max", "OPEX":"mpc_opex","tou_cost":"mpc_tou_cost"})
    df['label']=df['start']+" to "+df["end"]
    df_gruoped=df.groupby("label")
    plt.tight_layout()
    return df_gruoped

def plot_dc_line_group(df,plot_key_list_x,plot_key_list_y,x_key,suptitle,base_size):
    #plt.subplots()
    x_i_p=len(plot_key_list_x)
    y_i_p=len(plot_key_list_y)
    plt.subplots(y_i_p,x_i_p,sharex="col",sharey='row',figsize=base_size,)
    n=1
    y_i=1
    for s_y in plot_key_list_y:
        
        x_i=1
        for s_x in plot_key_list_x:
            x=df.get_group(s_x)[x_key]
            x_label=[]
            for l in x:
                x_label.append(str(l))
            y_mpc=df.get_group(s_x)['mpc_'+s_y]
            y_rbc=df.get_group(s_x)['rbc_'+s_y]
            ax=plt.subplot(y_i_p,x_i_p, n)
            ax.set_xticks(x)
            ax.set_xticklabels(x_label,rotation=0)
            ax.grid(axis='x',color='lightgray',linestyle="--")
            plt.scatter(x,y_mpc,color="seagreen",label="MPC",marker='x')
            plt.scatter(x,y_rbc,color="orangered",label="RBC",marker='.')
            if n==1:
                plt.legend()
            plt.plot(x,y_mpc,color="seagreen",alpha=0.7)
            plt.plot(x,y_rbc,color="orangered",alpha=0.7)

            if y_i==y_i_p:
                ax.set_xlabel(x_key,fontsize=15)
            if x_i==1:
                ax.set_ylabel(s_y,fontsize=15)
            
            if y_i==1:
                plt.title(str(s_x))
            x_i+=1
            n+=1
        y_i+=1
    
    plt.suptitle(suptitle,fontsize=18)
    plt.tight_layout()
    
    #plt.figsize([base_size[0]*len(plot_key_list_x),base_size[1]*len(plot_key_list_y)])
    plt.show()

def plot_box(df,relative,limit,figsize,key,save_fn,fontsize):
    
    labels=df['label'].unique()
    to_delete=np.array(['MPC-GT','MSC-GT'],dtype=object)
    labels_set=set(labels)
    to_delete_set=set(to_delete)
    labels=labels_set-to_delete_set
    
    if relative:
        key="relative_"+key

    boxes=[]
    fig,axs=plt.subplots(nrows=2,ncols=1,figsize=figsize)
    for i in labels:
        boxes.append(df[df.label==i][key])

    legend_elements = [Line2D([0], [0], color='orange', lw=1, label='median'),
                       Line2D([], [], color='black', marker='o', linestyle='None',
                          markersize=5, markerfacecolor='w',label='outliers')
                   ]    
   

    axs[0].boxplot(boxes, labels = labels, vert=False,showmeans=False,
               meanline=True)
    axs[0].set_xlim(limit)
    axs[0].set_title("Relative OPEX (percentage) of different models")
    axs[0].legend(handles=legend_elements, loc='upper right',fontsize=legend_fs)
    
    vp=axs[1].violinplot(boxes, vert=False,showmeans=False,
                      showmedians=True)
    vp['cmedians'].set_color('orange')

    for pc in vp['bodies']:
        pc.set_color("w")
        pc.set_edgecolor('black')
    axs[1].set_yticks([y + 1 for y in range(len(boxes))],
                  labels=labels)
    axs[1].set_xlim(limit)

    if save_fn is not None:
        plt.savefig(save_fn)
    plt.show()
    
def mplot_disturbance_type(params):
    fig, axs = plt.subplots(nrows=1, ncols=params["n_subplots"],
        figsize=params["figsize"],
        sharey=params["sharey"])
    fontsize=params["fontsize"]
    relative=params["relative"]
    compare_key=params["compare_key"]
    df=params["df"]
    for i in range(params["n_subplots"]):
        i=str(i)
        key=params["subplots"][i]["key"]
        
        limit=params["subplots"][i]["limit"]  
        method_key=params["subplots"][i]["method"]
        x_key=params["subplots"][i]["x_key"]
        subtitle=params["subplots"][i]["subtitle"]
        i=int(i)
        
        x_labels=np.array((df[df[compare_key]==method_key][x_key])*100)
           
        x_coor=np.linspace(0, len(df[df[compare_key]==method_key][x_key])-1, len(df[df[compare_key]==method_key][x_key]))
        y_coor_tou=np.array(df[df[compare_key]==method_key]["tou_cost"])
        y_coor_dc=np.array(df[df[compare_key]==method_key]["demand_charge"])

        axs[i].bar(x=x_coor,height=y_coor_tou,
                bottom=[0]*len(x_coor),
                color='lightblue',width=0.5,
                alpha=0.5,label="tou_cost")
        axs[i].bar(x=x_coor,height=y_coor_dc,
                bottom=y_coor_tou,
                color='cadetblue',width=0.5,
                alpha=0.5,label="demand_charge")
    
        axs[i].set_xticks(ticks=x_coor)
        axs[i].set_xticklabels(labels=np.round(x_labels,1),fontsize=fontsize,rotation=0)

        for label, color in zip(axs[i].get_xticklabels(), ["darkred"]*3+["black"]*13+["darkred"]*5):
            label.set_color(color)

        axs[i].set_ylim(limit)
        axs[i].set_title(subtitle,fontsize=fontsize*1.2)
        axs[i].axhline(y=df.loc[(df.pred_model=='GT')&(df.method=='MPC')]["OPEX"].values[0],
                       color='seagreen', linestyle='-', linewidth=1, label="MPC-GT-OPEX")
        axs[i].axhline(y=df.loc[(df.pred_model=='GT')&(df.method=='MPC')]["tou_cost"].values[0],
                       color='seagreen', linestyle='--', linewidth=1, label="MPC-GT-tou")
        axs[i].axhline(y=df.loc[(df.pred_model=='GT')&(df.method=='RBC')]["OPEX"].values[0],
                       color='orangered', linestyle='-', linewidth=1, label="MSC-GT-OPEX")
        axs[i].axhline(y=df.loc[(df.pred_model=='GT')&(df.method=='RBC')]["tou_cost"].values[0],
                       color='orangered', linestyle='--', linewidth=1, label="MSC-GT-tou")
        
        if i==0:
            axs[i].set_xlabel("MAPE(\%)",fontsize=fontsize*1.5,loc="left")
            axs[i].set_ylabel("Relative "*relative+key+" (Percentage)"*relative+"(US dollar/day)"*(not relative),fontsize=fontsize*1.5)    
    
    leg = plt.legend(loc='upper right', bbox_to_anchor=params["bbox_to_anchor"], fontsize=legend_fs)
    plt.suptitle(params["suptitle"],fontsize=fontsize*1.5)
    plt.tight_layout()
    
    if params["save_fn"] is not None:
        plt.savefig(params["save_fn"])
    plt.show()
    
    return


def cal_relative_12mon(fn,drop_base,group_keys=['month_of_year']):
    df=pd.read_excel(fn,sheet_name="Sheet1",index_col=0)
    df_grouped=df.groupby(group_keys)
    df_to_concat=[]
    for i in df_grouped.groups.keys():
        df=df_grouped.get_group(i)
        lower_bound=df[(df.strategy=="optimal")&(df.pred_model=="GT")]["OPEX"].values[0]
        upper_bound=df[(df.strategy=="MSC")&(df.pred_model=="GT")]["OPEX"].values[0]
        diff=upper_bound-lower_bound
        df["relative_OPEX"]=(df["OPEX"]-lower_bound)/diff*100
        if drop_base:
            df=df.drop(df[(df.pred_model=="GT")].index)
        df_to_concat.append(df)
    df=pd.concat(df_to_concat)
    return df
def get_df(fn,drop_base):
    concat_k96_dc=cal_relative_12mon(fn,drop_base)
    #concat_k96_dc['pred_K']=concat_k96_dc['exe_K']-concat_k96_dc['concat_K']
    pred_K=concat_k96_dc['concat_K'].unique()
    relative_dic_exeK96_dc=[]
    mean=[]
    for i in pred_K:
        values=concat_k96_dc[concat_k96_dc.concat_K==i]['relative_OPEX'].unique()
        relative_dic_exeK96_dc.append(values)
        mean.append(values.mean())
    return relative_dic_exeK96_dc,pred_K,mean




def get_df_for_plot_new(method,fn_list,x_quantiles=None):
    dfs=[]
    necessaey_cols=["method","strategy","B_kWh","pred_model",
                    "month_of_year","p_grid_max","price_dc",
                    "disturbance_rule","disturbance_MAPE","p_grid_max_method",
                    "OPEX","tou_cost","demand_charge","relative_OPEX"]
    for fn in fn_list:
        df=pd.DataFrame(cal_relative(fn),columns=necessaey_cols)
        dfs.append(df)
        
    for i in range(len(dfs)):
        df=dfs[i]
        rule_list=df["disturbance_rule"].unique()
        rule_list=rule_list[pd.notnull(rule_list)]
        for rule in rule_list:
            a=df[(df.disturbance_rule==rule)&(df.disturbance_MAPE==0.1)].copy()
            for k in ["OPEX","tou_cost","demand_charge","relative_OPEX"]:
                a[k]=[df[(df.strategy=="optimal")&(df.pred_model=="GT")][k].values[0]]*len(a)
            a["disturbance_MAPE"]=[0]*len(a)
            df=df._append(a).copy()
        dfs[i]=df.copy()
        
    df_concated=pd.concat(dfs).drop_duplicates().reset_index(drop=True)
    df_concated["disturbance_MAPE"] = df_concated["disturbance_MAPE"].apply(lambda x: round(x, 1))
    df_by_method=df_concated.groupby(by='p_grid_max_method')

    df_by_exe=df_by_method.get_group(method).groupby(by="disturbance_rule")
    df_uniform=df_by_exe.get_group("uniform")
    df_uniform_neg=df_by_exe.get_group("uniform_neg")
    df_uniform_pos=df_by_exe.get_group("uniform_pos")

    mape=sorted(df_uniform["disturbance_MAPE"].unique())
    
    y_uniform=list()
    y_uniform_neg=list()
    y_uniform_pos=list()
    y_uniform_mean=list()
    y_uniform_neg_mean=list()
    y_uniform_pos_mean=list()
    y_dic_all={}
    for m in mape:
        p=np.array(df_uniform_pos[df_uniform_pos.disturbance_MAPE==m]["relative_OPEX"].values)
        u=np.array(df_uniform[df_uniform.disturbance_MAPE==m]["relative_OPEX"].values)
        n=np.array(df_uniform_neg[df_uniform_neg.disturbance_MAPE==m]["relative_OPEX"].values)
        
        if x_quantiles!=None:
            m=m*x_quantiles*2
        else:
            x_quantiles=0.5
            
        if m in y_dic_all.keys():
            ...
        else:
            if m==0*x_quantiles*2:
                y_dic_all.update({m:[list(),0.5*x_quantiles*2,1*x_quantiles*2,1.5*x_quantiles*2]})
            elif m==0.1*x_quantiles*2:
                y_dic_all.update({m:[list(),1.5*x_quantiles*2,2*x_quantiles*2,2.5*x_quantiles*2]})
            elif m==0.5*x_quantiles*2:
                y_dic_all.update({m:[list(),2.5*x_quantiles*2,3*x_quantiles*2,3.5*x_quantiles*2]})
            else:
                y_dic_all.update({m:[list(),m+2.5*x_quantiles*2,m+3*x_quantiles*2,m+3.5*x_quantiles*2]})
        y_dic_all[m][0].append([m,0.1,'U+',p, np.nanmean(p)])
        y_dic_all[m][0].append([m,0.1,'U',u, np.nanmean(u)])
        y_dic_all[m][0].append([m,0.1,'U-',n, np.nanmean(n)])
        
            
        y_uniform.append(u)
        y_uniform_pos.append(p)
        y_uniform_neg.append(n)
        y_uniform_mean.append(np.nanmean(u))
        y_uniform_pos_mean.append(np.nanmean(p))
        y_uniform_neg_mean.append(np.nanmean(n))
    y_dic={
        'U':y_uniform,
        'U-':y_uniform_neg,
        'U+':y_uniform_pos
    }
    mean_dic={
        'U':y_uniform_mean,
        'U-':y_uniform_neg_mean,
        'U+':y_uniform_pos_mean
    }
    return mape,y_dic,mean_dic,y_dic,y_dic_all

def recal_position(dic,inter_group_r=0.9,inter_bar_r=0.7,base=1):
    key_list=np.array(list(dic.keys()))
    for k in range(len(key_list)):

        key=key_list[k]

        N=len(dic[key][0])
        start=dic[key][1]

        inner_gap=(base*inter_group_r)/(N+1)
        outter_gap=inner_gap+base*(1-inter_group_r)/2
        width=inner_gap*inter_bar_r
        
        for i in range(N):
            if i==0:
                dic[key][0][i][0]=start+outter_gap
            else:
                dic[key][0][i][0]=start+outter_gap+inner_gap*i
            dic[key][0][i][1]=width
    return dic
