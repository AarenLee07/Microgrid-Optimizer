# libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib as mpl
import pandas as pd
import sys

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


# process the original data
def pre_process(df,key):
    df.set_index('id')
    df=df[['strategy','pred_model','week_of_year',key]]
    df=df.replace('optimal',"MPC")
    df=df.replace("Simple","Heuristic")
    df['label']=df['strategy']+'-'+df['pred_model']

    for i in df['week_of_year'].unique():
        try:
            upper=df[((df['week_of_year']==i)) & (df['label']=='MSC-GT')][key].values[0]
            lower=df[((df['week_of_year']==i)) & (df['label']=='MPC-GT')][key].values[0]
            #assert upper>lower
            prediction=df[((df['week_of_year']==i)) & (df['label']=='MPC-Prediction')][key].values[0]
            heuristic=df[((df['week_of_year']==i)) & (df['label']=='MPC-Heuristic')][key].values[0]
            naive=df[((df['week_of_year']==i)) & (df['label']=='MPC-Naive')][key].values[0]
        except:
            pass
        
        upper_id=df[((df['week_of_year']==i)) & (df['label']=='MSC-GT')][key].index
        lower_id=df[((df['week_of_year']==i)) & (df['label']=='MPC-GT')][key].index
        prediction_id=df[((df['week_of_year']==i)) & (df['label']=='MPC-Prediction')][key].index
        heuristic_id=df[((df['week_of_year']==i)) & (df['label']=='MPC-Heuristic')][key].index
        naive_id=df[((df['week_of_year']==i)) & (df['label']=='MPC-Naive')][key].index 
        
        new_key='relative_'+key
        df.loc[lower_id,new_key]=0
        df.loc[upper_id,new_key]=100
        df.loc[prediction_id,new_key]=(prediction-lower)/(upper-lower)*100
        df.loc[heuristic_id,new_key]=(heuristic-lower)/(upper-lower)*100
        df.loc[naive_id,new_key]=(naive-lower)/(upper-lower)*100
    #df_new=df.drop(df[df['label']=='MPC-Naive'].index)
    print(df)
    df=df.reindex()
    invalid_week_list=df[(df[new_key]>100)|(df[new_key]<0)|(df[key].isna())].index
    invalid_week_list=df.iloc[invalid_week_list]['week_of_year'].unique()
    df['is_valid']=df['week_of_year'].apply(lambda x: True if x not in invalid_week_list else False)
    return df

def plot_origin(df,key,relative,save_fn,fontsize,ylimit):
    if relative:
        new_key='relative_'+key
    else:
        new_key=key
    color_dict={
        'MPC-GT':'seagreen',
        'MPC-Prediction':'navy',
        'MPC-Heuristic':'steelblue',
        'MPC-Naive':'slategray',
        'MSC-GT':'orangered',
        'MSC-Naive':'grey'
    }
    fig,ax=plt.subplots(figsize=(20,5))

    scatter_x=np.array(df['week_of_year'])
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
        
def plot_valid(df,relative,limit,figsize,save_fn,key,fontsize):
    
    if relative:
        new_key='relative_'+key
    else:
        new_key=key

    df_valid=df.drop(df[df.is_valid==False].index)

    x_coor=np.array([df_valid[df_valid.label=='MPC-Prediction']['week_of_year'],\
        df_valid[df_valid.label=='MPC-Prediction']['week_of_year']])

    y_coor=np.array([df_valid[df_valid.label=='MPC-Heuristic'][new_key],\
        df_valid[df_valid.label=='MPC-Prediction'][new_key]])

    color_dict={
        'MPC-GT':'seagreen',
        'MPC-Prediction':'navy',
        'MPC-Heuristic':'steelblue',
        'MPC-Naive':'gainsboro',
        'MSC-GT':'orangered',
        'MSC-Naive':'grey'
    }
    maker_dict={
        'MPC-GT':'_',
        'MPC-Prediction':'_',
        'MPC-Heuristic':'_',
        'MPC-Naive':'_',
        'MSC-GT':'_',
        'MSC-Naive':'_'
    }

    fig,ax=plt.subplots(figsize=figsize)

    scatter_x=np.array(df_valid['week_of_year'])
    scatter_y=np.array(df_valid[new_key])
    is_valid=np.array(df_valid['is_valid'])
    group=np.array(df_valid['label'])
    label_x=[]
    for i in scatter_x: 
        label_x.append('Week '+str(i))

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
    ax.set_xlabel("Weeks in the year of 2019",fontsize=fontsize*1.5)
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
    
def plot_origin_valid_bar(df,relative,limit,figsize,key,save_fn,fontsize,notes_y=-40):
    if relative:
        new_key='relative_'+key
    else:
        new_key=key
    df_valid=df.drop(df[df.is_valid==False].index)

    x_coor=np.array([df_valid[df_valid.label=='MPC-Prediction']['week_of_year'],\
        df_valid[df_valid.label=='MPC-Prediction']['week_of_year']])

    y_coor=np.array([df_valid[df_valid.label=='MPC-Heuristic'][new_key],\
        df_valid[df_valid.label=='MPC-Prediction'][new_key]])
    
    color_dict={
        'MPC-GT':'seagreen',
        'MPC-Prediction':'navy',
        'MPC-Heuristic':'steelblue',
        'MPC-Naive':'gainsboro',
        'MSC-GT':'orangered',
        'MSC-Naive':'grey'
    }
    
    maker_dict={
        'MPC-GT':'_',
        'MPC-Prediction':'_',
        'MPC-Heuristic':'_',
        'MPC-Naive':'_',
        'MSC-GT':'_',
        'MSC-Naive':'_'
    }
    
    

    fig,ax=plt.subplots(figsize=figsize)
    
    plt.grid(axis = 'x',linestyle='--',alpha=0.1)
    plt.grid(axis = 'y',linestyle='--',alpha=0.8)
    
    scatter_x=np.array(df['week_of_year'])
    scatter_y=np.array(df[new_key])

    group=np.array(df['label'])
    label_x=[]
    for i in scatter_x: 
        label_x.append('Week '+str(i))
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
        x_coor_MPC_GT=np.array([df[df.label=='MPC-GT']['week_of_year'],\
            df[df.label=='MPC-GT']['week_of_year']])
        y_coor_MPC_GT=np.array([[0]*len(df[df.label=='MPC-GT'][new_key]),\
            df[df.label=='MPC-GT'][new_key]])
        #for i in range(len(x_coor_MPC_GT[0])):
        ax.bar(x=x_coor_MPC_GT[0],height=np.abs(y_coor_MPC_GT[0]-y_coor_MPC_GT[1]), \
            bottom=y_coor_MPC_GT[0],color='seagreen',
            width=0.6,alpha=0.1,label="MPC_GT")
            
    ax.legend(loc='upper left',bbox_to_anchor=(1.02,1),fontsize=fontsize)
        
    
    ax.set_xlabel("Weeks in the year of 2019",fontsize=fontsize*1.5)
    ax.set_ylabel("Relative "*relative+key+" (Percentage)"*relative+"(US dollar/day)"*(not relative),fontsize=fontsize*1.5)

    ax.set_title("Relative "*relative+key+" of Different Models",fontsize=fontsize*1.5)
    if relative:
        ax.text(s="Notes: 1. "+key+" under MPC-GT is marked as lower bound while MSC_GT marked as upper bound.",
                fontsize=fontsize*1.2, x=0, y=notes_y
                )
    
    plt.tight_layout()
    
    if save_fn is not None:
        plt.savefig(save_fn)
    plt.show()

        
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
    fig,ax=plt.subplots(figsize=figsize)
    for i in labels:
    
        boxes.append(df[df.label==i][key])
    
    #box_1, box_2, box_3, box_4 = data['收入_Jay'], data['收入_JJ'], data['收入_Jolin'], data['收入_Hannah']
    
    #plt.figure(figsize=figsize)#设置画布的尺寸
    plt.title('Examples of boxplot',fontsize=fontsize)#标题，并设定字号大小
   
    #vert=False:水平箱线图；showmeans=True：显示均值
    ax.boxplot(boxes, labels = labels, vert=False,showmeans=True )
    ax.set_xlim(limit)
    if save_fn is not None:
        plt.savefig(save_fn)
    plt.show()