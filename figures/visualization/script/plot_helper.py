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
        'MPC-Heuristic':'steelblue',
        'MPC-Naive':'slategray',
        'MSC-GT':'orangered', #green
        'MPC-Disturbance':'purple',
        'MSC-Naive':'grey'
    }
color_dic_glb_w={
        'MPC-GT':'w',# aquamarine
        'MPC-Prediction':'w',
        'MPC-Heuristic':'w',
        'MPC-Naive':'w',
        'MSC-GT':'w', #green
        'MPC-Disturbance':'w',
        'MSC-Naive':'w'
    }
marker_dic_glb_w={
        'MPC-GT':'_',
        'MPC-Prediction':'^',
        'MPC-Heuristic':'x',
        'MPC-Naive':'v',
        'MPC-Disturbance':'+',
        'MSC-GT':'_',
        'MSC-Naive':'_'
    }

marker_dic_glb={
        'MPC-GT':'_',
        'MPC-Prediction':'_',
        'MPC-Heuristic':'_',
        'MPC-Naive':'_',
        'MPC-Disturbance':'_',
        'MSC-GT':'_',
        'MSC-Naive':'_'
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
    
    df=df[['strategy','pred_model',duration_key,key]]
    df=df.replace('optimal',"MPC")
    df=df.replace("Simple","Heuristic")
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
    ax.legend(fontsize=fontsize)
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
    ax.set_xticklabels(labels=label_x,fontsize=fontsize,rotation=45)

    for g in np.unique(group):
        i = np.where(group == g)
        ax.scatter(scatter_x[i], scatter_y[i], label=g,c=color_dict[g],\
            marker=maker_dict[g],s=250)
    ax.set_ylim(limit)
    ax.legend(loc='upper left',bbox_to_anchor=(1.02,1),fontsize=fontsize)

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
    
def plot_origin_valid_bar(df,relative,limit,figsize,key,save_fn,fontsize,
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
    ax.set_xticklabels(labels=label_x,fontsize=fontsize,rotation=45)

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
            
    ax.legend(loc='upper left',bbox_to_anchor=(1.02,1),fontsize=fontsize)
        
    
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
        
    fig, axs = plt.subplots(nrows=1, ncols=params["n_subplots"],
                            figsize=params["figsize"],
                            sharey=params["sharey"])
    fontsize=params["fontsize"]
    relative=params["relative"]
    
    for i in range(params["n_subplots"]):
        i=str(i)
        key=params["subplots"][i]["key"]
        df=params["subplots"][i]["df"]
        limit=params["subplots"][i]["limit"]  
        duration_key=params["subplots"][i]["duration_key"]
        subtitle=params["subplots"][i]["subtitle"]
        is_gradient=params["subplots"][i]["gradient"]
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
        else:
            new_key=key
        df_valid=df.drop(df[df.is_valid==False].index)

        x_coor=np.array([df_valid[df_valid.label=='MPC-Prediction'][duration_key],\
            df_valid[df_valid.label=='MPC-Prediction'][duration_key]])
        y_coor=np.array([df_valid[df_valid.label=='MPC-Heuristic'][new_key],\
            df_valid[df_valid.label=='MPC-Prediction'][new_key]])
        
        axs[i].grid(axis = 'x',linestyle='--',alpha=0.1)
        axs[i].grid(axis = 'y',linestyle='--',alpha=0.8)
        
        scatter_x=np.array(df[duration_key])
        scatter_y=np.array(df[new_key])

        group=np.array(df['label'])
        if params["show_line"]:
            for label in group:
                x=np.array(df_valid[df_valid.label==label][duration_key])
                y=np.array(df_valid[df_valid.label==label][new_key])
                axs[i].plot(x, y, color='white', linestyle='dashed',
                    linewidth=0.3, markersize=0,alpha=0.3)
        
        label_x=[]
        for k in scatter_x: 
            if duration_key=='month_of_year':
                label_x.append(calendar.month_abbr[k])
            else:
                label_x.append(duration_key+str(k))
        axs[i].set_xticks(ticks=scatter_x)
        axs[i].set_xticklabels(labels=label_x,fontsize=fontsize,rotation=45)
        
        if is_gradient:
            for g in np.unique(group):
                maker_dict=marker_dic_glb_w
                m = np.where(group == g)
                axs[i].scatter(scatter_x[m], scatter_y[m], label=g,c=color_dict[g],\
                    marker=maker_dict[g],s=params["marker_s"]*0.12)
        else:
            for g in np.unique(group):
                m = np.where(group == g)
                axs[i].scatter(scatter_x[m], scatter_y[m], label=g,c=color_dict[g],\
                    marker=maker_dict[g],s=params["marker_s"])
            
        axs[i].set_ylim(limit)
        
        if relative:
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
            
        
        axs[i].set_xlabel(duration_key+" 2019",fontsize=fontsize*1.5)
        axs[i].set_title(subtitle,fontsize=fontsize*1.2)
        if i==0:
            axs[i].set_ylabel("Relative "*relative+key+" (Percentage)"*relative+"(US dollar/day)"*(not relative),fontsize=fontsize*1.5)
    
    leg = plt.legend(loc='upper right', bbox_to_anchor=(1.5, 1), fontsize=fontsize)
    if is_gradient:
        for handle in leg.legendHandles:
            handle.set_color('black')
        
    if relative*params["show_notes"]:
        plt.text(s="Notes: 1. "+key+" under MPC-GT is marked as lower bound while MSC_GT marked as upper bound.",
                fontsize=fontsize*1.2, x=0, y=-40
                )
    plt.suptitle(params["suptitle"],fontsize=fontsize*1.5)
    plt.tight_layout()
    
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
            ax.set_xticklabels(x_label,rotation=45)
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
    axs[0].legend(handles=legend_elements, loc='upper right')
    
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