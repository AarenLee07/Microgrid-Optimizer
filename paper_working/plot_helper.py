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

legend_fs=10
label_fs=12
ticklabel_fs=9
title_fs=14

rc_={
    "figure.dpi":600,
    "font.size":10,
    "axes.facecolor":"white",
    "savefig.facecolor":"white",
    "text.usetex":True,
    "legend.frameon":False
}

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
    df_error=pd.read_excel(fn_by_exe, sheet_name="op_log",index_col=0)["load_bld_error"].interpolate()
    
    # Merge the Series into a DataFrame
    df_merged = pd.merge(df_by_exe, df_by_sol, left_index=True, right_index=True)
    df_merged = pd.merge(df_merged, df_mini, left_index=True, right_index=True)
    df_merged = pd.merge(df_merged, df_error, left_index=True, right_index=True)

    # Rename the columns
    df_merged.columns = ['actual_p_max', 'necessary', 'minimized', 'load_pred_error']
    df_merged['unnecessary']=df_merged["necessary"]-df_merged["actual_p_max"]
    df_merged['unnecessary_minimized']=df_merged["necessary"]-df_merged["minimized"]
    print(type(df_merged))
    return df_merged

def plot_track_p_max(df_merged,figsize,line_keys=['actual_p_max','necessary','unnecessary'],linewidth=1,
                     plot_error_bar=False, ylimit_main=[-150,250],ylimit_sub=[-150,250],
                     inside_start_day=5,inside_days=1,legend_loc="lower right",track_real=False,shadow=False,
                     save_fn=None,ax=None,axins=None):
        line_clr_dic={
                'actual_p_max':'teal',
                'necessary':'green',
                'unnecessary':'coral',
                'unnecessary_minimized':'purple'
        }
        axin_color='steelblue'
        
        if ax==None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            ax=ax

        def plot_ax_data(ax):
            for key in line_keys:
                if key=='unnecessary' and shadow==False:
                    ax.plot(df_merged.index, df_merged[key],label=key,color=line_clr_dic[key],linewidth=linewidth)
                elif key=='unnecessary' and shadow==True:
                    ax.fill_between(df_merged.index,0,df_merged[key],facecolor='orange',alpha=0.3)
                else :
                    ax.plot(df_merged.index, df_merged[key],label=key,color=line_clr_dic[key],linewidth=linewidth)
            if plot_error_bar:
                ax.bar(df_merged.index, df_merged["load_pred_error"])
            ax.yaxis.set_major_formatter(lambda x, pos: f'{abs(x):g}')
            ax.margins(x=0.01)
            ax.hlines(y=0,xmin=df_merged.index[0],xmax=df_merged.index[-1],linestyles='--',colors='gray',linewidth=0.6,alpha=0.6)
            ax.tick_params(axis='both', direction='in')
            ax.set_xticklabels(ax.get_xticklabels(),rotation=0,fontsize=ticklabel_fs)
            ax.tick_params(axis='both',which='major',labelsize=ticklabel_fs)

        plot_ax_data(ax=ax)
        ax.text(x=0.7,y=0.76,s='$necessary$',ha='center',va='center',transform=ax.transAxes,color='darkgreen')
        ax.text(x=0.7,y=0.28,s='$track\_real$',ha='center',va='center',transform=ax.transAxes,color='peru')
        if track_real==True:
            ax.text(x=0.7,y=0.5,s='$unnecessary$',ha='center',va='center',transform=ax.transAxes,color='purple')
        ax.set_ylim(ylimit_main)
        ax.set_xlabel("Day of month (May-2019)",fontsize=label_fs,loc='center')
        ax.set_ylabel("Peak demand till $t$ (kW)",fontsize=label_fs)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=24*7))
        ax.xaxis.set_minor_locator(mdates.HourLocator(interval=24))
        ax.tick_params(axis='x', rotation=0)

        ax.set_xticklabels(ax.get_xticklabels(),rotation=0,fontsize=ticklabel_fs)
        #ax.legend(frameon=False,loc=legend_loc,fontsize=legend_fs)
        ax.tick_params(axis='both',which='major',labelsize=ticklabel_fs)
        
        
        # set the subplot:
        if axins==None:
            axins = inset_axes(ax, width="60%", height="100%", loc='lower left',
                            bbox_to_anchor=(1.05, -0.031, 1, 1), 
                            bbox_transform=ax.transAxes)
        else:
            axins=axins

        # plot original data
        plot_ax_data(ax=axins)

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
        axins.set_xlabel("Hour of day (1-May)",fontsize=label_fs,loc='center')
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
        ax.tick_params(direction='out',axis='y')
        # adjust layout and save fig
        if ax==None:
            plt.tight_layout()
            plt.savefig(save_fn,bbox_inches='tight')
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
        y_uniform_mean.append(np.mean(u))
        y_uniform_pos_mean.append(np.mean(p))
        y_uniform_neg_mean.append(np.mean(n))
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
                     mape,y_dic,mean_dic,ylimit=(-10,180),
                     mean_dic_old=None,ax=None,
                     plot_line_new=True, plot_line_old=False, save_fn=None):
    if plot_line_old==True:
        assert mean_dic_old is not None
    if ax!=None:
        ax=ax
    else:
        fig, ax=plt.subplots(figsize=figsize)
    positions1=np.array(range(1,len(mape)+1))-figsize[0]/32
    positions2=np.array(range(1,len(mape)+1))
    positions3=np.array(range(1,len(mape)+1))+figsize[0]/32
    color_dic={
        'U':'steelblue',
        'U-':'orangered',
        'U+':'darkgreen'
    }
    pos_dic={
        'U':positions1,
        'U-':positions2,
        'U+':positions3
    }

    boxs=[]
    for key in color_dic.keys():
        bplot=ax.boxplot(y_dic[key],positions=pos_dic[key],patch_artist=True,showmeans=True,widths=figsize[0]/40,        
                boxprops={"facecolor": color_dic[key],
                        "edgecolor": "w",
                        "linewidth": 0,
                        'alpha':0.4},
                medianprops={"color": color_dic[key], "linewidth": 0},
                meanprops={'marker':'+',
                        'markerfacecolor':color_dic[key],
                        'markeredgecolor':color_dic[key],
                        'markersize':4.1*figsize[0]/8},
                sym="",showfliers=True, showcaps=False,
                whiskerprops={'color': 'w', 'linewidth': 1, 'linestyle': '--', 'alpha':0},
                )
        if plot_line_new==True:
            lplot=ax.plot(pos_dic[key],mean_dic[key],linestyle='-',color=color_dic[key],alpha=0.2,linewidth=0.5)
        if plot_line_old==True:
            lplot=ax.plot(pos_dic[key],mean_dic_old[key],linestyle='--',marker='+',color=color_dic[key],
                          alpha=0.5,linewidth=0.8,markersize=0*figsize[0]/8)
        ax.set_xticks([])
        ax.set_xticklabels([])
        boxs.append(bplot['boxes'][0])


    labels=[]
    positions=[]
    minor=[]
    for i in range(len(mape)):
        #labels.append("")
        #minor.append(True)
        #positions.append(positions1[i])
        labels.append(str('%.3g' % mape[i]))
        positions.append(positions2[i])
        minor.append(False)
        #labels.append("")
        #positions.append(positions3[i])
        #minor.append(True)
    ax.set_xticks(ticks=positions,minor=minor,labels=labels,fontsize=ticklabel_fs)
    ax.tick_params(direction='in', axis="x", which='both')

    ax.set_ylim(ylimit)
    # Set major locator
    ax.yaxis.set_major_locator(plt.MultipleLocator(20))

    # Set minor locator
    ax.yaxis.set_minor_locator(plt.MultipleLocator(10))

    # Set y-tick labels font size
    ax.tick_params(axis='y', labelsize=9)
    ax.tick_params(axis='x', labelsize=9)
    ax.set_xlabel('MAPE(%)',loc='left',fontsize=label_fs)
    ax.set_ylabel("Relative regret (%)", fontsize=label_fs)
    for i in range(len(positions2)):
        if i%2==0:
            diff=positions2[1]-positions2[0]
            l=positions2[i]-diff/2
            r=positions2[i]+diff/2
            ax.fill_betweenx(ylimit, l,r, facecolor='dimgray', alpha=0.05)

    legend_elements=[]
    for key in color_dic.keys():
        legend_elements.append(Patch(facecolor=color_dic[key], edgecolor='w',label=key,alpha=0.4))
    #if plot_line_new==True:
    legend_elements.append(Line2D([0], [0], marker='+', color='gray', label='Mean',linewidth=0,
                                markerfacecolor='gray', markersize=8))
    
        #legend_elements.append(Line2D([0], [0], marker='+', color='gray', label='Median',linewidth=0,
                               # markerfacecolor='gray', markersize=0))
    if plot_line_old==True:
        legend_elements.append(Line2D([0], [0], marker='+', color='gray', label='track real',linewidth=0.6,linestyle='--',
                            markerfacecolor='gray', markersize=0))
    
    ax.legend(handles=legend_elements,loc='upper left',fontsize=legend_fs) 
    ax.set_xlabel("MAPE", loc='center',fontsize=label_fs)
    
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

# set default params of plot
rc_={
    "figure.dpi":300,
    "font.size":10,
    "axes.facecolor":"white",
    "savefig.facecolor":"white",
    "text.usetex":True,
}

color_dic_glb={
        'MPC-GT':'seagreen',# aquamarine
        'MPC-Prediction':'navy',
        'MPC-Heuristic':'peru',
        'MPC-Heuristic-our-method':'purple',
        'MPC-Naive':'slategray',
        'MSC-GT':'gray', #green
        'MPC-Disturbance':'purple',
        'MSC-Naive':'grey'
    }
color_dic_glb_w={
        'MPC-GT':'w',# aquamarine
        'MPC-Prediction':'w',
        'MPC-Heuristic':'w',
        'MPC-Naive':'w',
        'MPC-Heuristic-our-method':'w',
        'MSC-GT':'w', #green
        'MPC-Disturbance':'w',
        'MSC-Naive':'w'
    }
marker_dic_glb_w={
        'MPC-GT':'_',
        'MPC-Prediction':'^',
        'MPC-Heuristic':'x',
        'MPC-Naive':'v',
        'MPC-Heuristic-our-method':'o',
        'MPC-Disturbance':'+',
        'MSC-GT':'_',
        'MSC-Naive':'_'
    }

marker_dic_glb={
        'MPC-GT':'_',
        'MPC-Prediction':'_',
        'MPC-Heuristic':'x',
        'MPC-Naive':'_',
        'MPC-Disturbance':'_',
        'MPC-Heuristic-our-method':'o',
        'MSC-GT':'_',
        'MSC-Naive':'_'
    }
marker_s_dict={
        'MPC-GT':1,
        'MPC-Prediction':1,
        'MPC-Heuristic':0.3,
        'MPC-Naive':1,
        'MPC-Disturbance':1,
        'MPC-Heuristic-our-method':0.3,
        'MSC-GT':1,
        'MSC-Naive':1
    }
legend_dict={
        'MPC-GT':'MPC-GT',
        'MPC-Prediction':'MPC-Prediction',
        'MPC-Heuristic':'track real',
        'MPC-Naive':'MPC-Naive',
        'MPC-Disturbance':'MPC-Arti-Noise',
        'MPC-Heuristic-our-method':'track necessary(ours)',
        'MSC-GT':'RBC',
        'MSC-Naive':'MSC-Naive'
    }

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
    
    df=df[['strategy','pred_model',duration_key,key,'days']]
    df=df.replace('optimal',"MPC")
    df=df.replace("Simple","Heuristic")
    df=df.replace("Simple-our-method","Heuristic-our-method")
    df['label']=df['strategy']+'-'+df['pred_model']
    data=dict()
    
    for i in df[duration_key].unique():
        try:
            for label in df['label'].unique():
                data[label]=df[((df[duration_key]==i)) & (df['label']==label)][key].values[0]
            
            upper=df[((df[duration_key]==i)) & (df['label']=='MSC-GT')][key].values[0]
            lower=df[((df[duration_key]==i)) & (df['label']=='MPC-GT')][key].values[0]
            '''
            #assert upper>lower
            prediction=df[((df['week_of_year']==i)) & (df['label']=='MPC-Prediction')][key].values[0]
            heuristic=df[((df['week_of_year']==i)) & (df['label']=='MPC-Heuristic')][key].values[0]
            naive=df[((df['week_of_year']==i)) & (df['label']=='MPC-Naive')][key].values[0]
            '''
        except:
            pass
        
        upper_id=df[((df[duration_key]==i)) & (df['label']=='MSC-GT')][key].index
        lower_id=df[((df[duration_key]==i)) & (df['label']=='MPC-GT')][key].index
        
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

    x_coor=np.array([df_valid[df_valid.label=='MPC-Prediction'][duration_key],\
        df_valid[df_valid.label=='MPC-Prediction'][duration_key]])

    y_coor=np.array([df_valid[df_valid.label=='MPC-Heuristic'][new_key],\
        df_valid[df_valid.label=='MPC-Prediction'][new_key]])

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
            arrow_start='relative_'+arrow_start
            arrow_end='relative_'+arrow_end
        else:
            new_key=key
        df_valid=df#.drop(df[df.is_valid==False].index)

        axs[i].grid(axis = 'x',linestyle='--',alpha=0.1)
        axs[i].grid(axis = 'y',linestyle='--',alpha=0.8)
        
        scatter_x=np.array(df[duration_key])
        scatter_y=np.array(df[new_key])
        
        if plot_arrow==True:
            arrawx=np.array(df_valid[df_valid.label==arrow_start][duration_key])
            arrowy_s=np.array(df_valid[df_valid.label==arrow_start][new_key])
            arrowy_e=np.array(df_valid[df_valid.label==arrow_end][new_key])
            print(range(len(arrawx)))
            for k in range(len(arrawx)):
                axs[i].arrow(arrawx[k], arrowy_s[k], 0, (arrowy_e[k]-arrowy_s[k])*0.95,
                            width=0.005*limit[1]/550,color='dimgray',alpha=0.4,head_width=0.4,head_length=3*limit[1]/550,
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
            if ('MPC-Prediction' in params["labels_not_show"]) or\
                ('MPC-Heuristic'in params["labels_not_show"]):
                    pass
            else:
                x_coor=np.array([df_valid[df_valid.label=='MPC-Prediction'][duration_key],\
                    df_valid[df_valid.label=='MPC-Prediction'][duration_key]])
                y_coor=np.array([df_valid[df_valid.label=='MPC-Heuristic'][new_key],\
                    df_valid[df_valid.label=='MPC-Prediction'][new_key]])
                axs[i].bar(x=x_coor[0],height=np.abs(y_coor[0]-y_coor[1]), \
                    bottom=min_help(y_coor[0],y_coor[1]),
                    color='lightsteelblue',width=0.6,alpha=0.3,label="Prediction<Heuristic")
            
        if not relative:
            x_coor_MPC_GT=np.array([df[df.label=='MPC-GT'][duration_key],\
                df[df.label=='MPC-GT'][duration_key]])
            y_coor_MPC_GT=np.array([[0]*len(df[df.label=='MPC-GT'][new_key]),\
                df[df.label=='MPC-GT'][new_key]])
            axs[i].bar(x=x_coor_MPC_GT[0],height=np.abs(y_coor_MPC_GT[0]-y_coor_MPC_GT[1]), \
                bottom=y_coor_MPC_GT[0],color='seagreen',
                width=0.6,alpha=0.1,label="MPC_GT")
            
        if duration_key=='month_of_year':
            axs[i].set_xlabel("Month of year(2019)",fontsize=label_fs,loc='center')
        else:
            axs[i].set_xlabel(duration_key,fontsize=label_fs,loc='center')
        axs[i].set_title(subtitle,fontsize=title_fs)
        if key=='grid_max':
            axs[i].set_ylabel("peak demand (kW)",fontsize=label_fs)
        if key=='OPEX':
            axs[i].set_ylabel("OPEX (k\$)",fontsize=label_fs)

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
        leg = axs[1].legend(loc='upper right', bbox_to_anchor=params["bbox_to_anchor"], fontsize=legend_fs)
    
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