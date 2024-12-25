#!/usr/bin/env python
# coding: utf-8

# # Visualize global AUC.

# In[1]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


# In[]]


parser = argparse.ArgumentParser()

## Perturbation type.
parser.add_argument(
    '--pert_type', 
    type=str, 
    default='trt_sh.cgs', 
    help='Perturbation type of protein signatures, e.g., trt_sh.cgs or trt_oe'
) # knockdown signatures
args = parser.parse_args()


# ### Setting.

# In[44]:


# ========================================================
# Setting
# ========================================================

# ===== Proteins ===== #

# ----- Perturbation type ----- #
pert_type = args.pert_type
# pert_type = 'trt_sh.cgs'
# pert_type = 'trt_oe'


# Cell selcection method.
# cell_select = "Each" # "All"：平均化データのみ使用, "Each"：各細胞種を使用
cell_select = "All"


# Cell line.
c = "All"


# ===== Parameters ===== #

# ---- fold list ----- #
fold_list = [1, 2, 3, 4, 5]
    
    
# ----- SNP profiling method ------ #
snp_type_list = [ 
    "SNPprofileEQsum", "SNPprofilePvalCodingMean", "fuma_gwas_P"
]


# ------ Neural network ------ #
nn_type_list = [ 
    'TargetFeature_VAE_DiseaseFeature_VAE',
    'TargetFeature_VQ_VAE_Parallel_DiseaseFeature_VQ_VAE_Parallel'
]

# ------ Pi model ------ #
pi_type_list = [ 
    'TargetFeature_VAE_DiseaseFeature_VAE',
    'TargetFeature_VQ_VAE_Parallel_DiseaseFeature_VQ_VAE_Parallel'
]


# ### Color list.

# In[3]:


# ========================================================
# Color リストを作成
# ========================================================

# ----- Color list ----- #
tab20b_list = sns.color_palette("tab20b")
tab20_list = sns.color_palette("tab20c")
blue_list = sns.color_palette("Blues_r")
set2_list = sns.color_palette("Set2")
ylorbr_list = sns.color_palette("YlOrBr")
reds_list = sns.color_palette( "Reds" )
rdpu_list = sns.color_palette( "RdPu" )
puple_list = sns.color_palette( "Purples" )
color_list = [blue_list[i] for i in [0,1,2,3,4,5] ] + [tab20_list[i] for i in [4,5,6] ]

# カラーdict.
color_dict = dict(zip( ['knn', 'rbf'],  [ tab20_list[0], tab20_list[5]] ) )

color_dict


# In[48]:


# ========================================================
# カラーdict
# ========================================================

color_dict = {}
orphan_color_dict = {}
feature_color_dict = {}
vae_color_dict = {}

# ===== Pimodel ====== #
color_dict['Pi_All_TargetFeature_VQ_VAE_Parallel_DiseaseFeature_VQ_VAE_Parallel'] = tab20_list[5]
color_dict['Pi_All_TargetFeature_VAE_DiseaseFeature_VAE'] = tab20_list[7]

# ===== Supervised NN ===== #
color_dict['NN_All_TargetFeature_VQ_VAE_Parallel_DiseaseFeature_VQ_VAE_Parallel'] = tab20_list[1]
color_dict['NN_All_TargetFeature_VAE_DiseaseFeature_VAE'] = tab20_list[2]

# ===== Baselin methods. ===== #
color_dict['multitask'] = tab20_list[16]
color_dict['SNPprofileEQsum'] = tab20_list[17]
color_dict['fuma_gwas_P'] = tab20_list[18] # Magma
color_dict['SNPprofilePvalCodingMean'] = tab20_list[19]


# In[5]:


sns.color_palette("YlOrBr")


# In[6]:


sns.color_palette("tab20c",  20)


# In[7]:


sns.color_palette("Blues_r")


# In[8]:


sns.color_palette("Purples")


# In[9]:


sns.color_palette( "RdPu" )


# In[10]:


sns.color_palette("Reds")


# In[11]:


# =========================================
# Gold standard data.
# =========================================

i_f = f"../../semisupervised/data/{pert_type}/LabelData/label.txt"
gold_df = pd.read_csv(i_f, sep = '\t', encoding= 'shift-jis' )
    
dis_list = list(set(gold_df['disease_id']))# Disease list.
pro_list = list(set(gold_df['gene'])) # target list.

gold_df = gold_df.rename(columns = {'disease_id':'disease'})[['disease', 'disease_name', 'disease_degree', 'gene']]
gold_df = gold_df.drop_duplicates()
gold_df = gold_df.drop('gene', axis = 1).drop_duplicates() # 'gene'の列を除く

gold_df.head()


# In[12]:


# =========================================
# 階級幅を設定（次数6以降は大きい幅で）
# =========================================

common_gold_df = gold_df.copy()
max_deg = np.max(   common_gold_df['disease_degree']   ) # 次数の最大値


# ----- 次数とその疾患数 ----- #
deg_gold_df = common_gold_df.copy()
deg_gold_df['count'] = 1
deg_gold_df = deg_gold_df.groupby(by = ['disease_degree'])['count'].sum(numeric_only = True).reset_index()\
                                .sort_values(by = 'disease_degree', ignore_index = True) # 次数ごとに和



# ----- 評価用マトリックス作成 ----- #
large_deg_gold_df = deg_gold_df.iloc[5:].reset_index(drop = True) # 小さい次数5種を除いた次数とその数
total = large_deg_gold_df['count'].sum() # 小さい次数5種を除いた疾患数
quot = total // 5 # 商

init_state_df = np.array(   [0] * (1 + len( large_deg_gold_df ))*5  ) # 各foldの疾患数
init_state_df = init_state_df.reshape( [5, 1 + len( large_deg_gold_df )] ) # fold x 上の指標

# Dataframe.
init_state_df = pd.DataFrame(init_state_df, 
                        index=[1, 2, 3, 4, 5], 
                        columns=["num"]+ list(deg_gold_df.iloc[5:]['disease_degree']) )



# ----- fold番号を割り当てる ----- #
state_df = init_state_df.copy()
res_list = []

for i in range(len( large_deg_gold_df )):
    
    deg, count = tuple( large_deg_gold_df.iloc[i].values ) # disease degree, counts

    if i == 0:
        tmp_state = 1
        tmp_num = count

    elif tmp_num < quot:
        tmp_state = tmp_state
        tmp_num += count
        
    elif tmp_state == 5:
        tmp_state = 5
        tmp_num += count
        
    else:
        tmp_state += 1
        tmp_num = count

    state_df.at[tmp_state, "num"] += count
    state_df.at[tmp_state, deg] +=1
    res_list.append(tmp_state)

    
    
# ----- クラス分けの結果をまとめる ----- #
large_deg_gold_df['clust_id'] = res_list # クラスタidを付与
tmp_df = large_deg_gold_df.groupby(by = 'clust_id')['disease_degree'].max(numeric_only = True)\
                        .reset_index().rename(columns = {'disease_degree':'max_degree'}) # clust idごとに最大の次数を抽出
large_deg_gold_df = pd.merge( large_deg_gold_df, tmp_df, on = 'clust_id', how = 'left') # 最大次数の情報を紐付け


# ----- 階級幅のdataframe ----- #
bin_df = deg_gold_df.iloc[:5].copy() # 小さい次数5種類
bin_df['max_degree'] = [s for s in bin_df['disease_degree']] # 最大次数を紐付け
bin_df = bin_df.drop( 'count', axis=1) # countの列を削除
bin_df = pd.concat( [ bin_df, large_deg_gold_df[['disease_degree', 'max_degree']] ], axis=0 ) # 小さい次数と大きい次数をマージ
bin_dict = dict(zip(bin_df['disease_degree'], bin_df['max_degree'])) # dict

    
print("numは特に注意して確認")
state_df


# ### ===== Global AUC ===== #

# In[46]:


# =========================================
# Local AUCを読み込む
# =========================================

for v_type in ['auc']:

    lauc_df = pd.DataFrame()
    
        
    # ===== Multi-task ===== #
    
    # ----- AUC ----- #
    i_f = f'../data/{pert_type}/global_auc/multitask.txt'
    tmp_auc_df = pd.read_csv(i_f, sep = '\t')
    tmp_auc_df = tmp_auc_df.groupby( by = ['cell', 'similarity_type', 'l1', 'l2'])['auc'].mean().reset_index()
    tmp_auc_df = tmp_auc_df[tmp_auc_df['cell'] == "All"] # cellを選択
    
    # ----- AUC/AUPR/BED AUC ---- #
    i_f = f'../data/{pert_type}/global_{v_type}/multitask.txt'
    tmp_df = pd.read_csv(i_f, sep = '\t')
    if v_type == 'bedauc':
        tmp_df = tmp_df.rename(columns = {'auc': 'bedauc'})
    tmp_df = tmp_df.groupby( by = ['cell', 'similarity_type', 'l1', 'l2'])[v_type].mean().reset_index()
        
    tmp_df = tmp_df[tmp_df['cell'] == "All"] # cellを選択
    if v_type == "auc":
        tmp_df = pd.merge( tmp_df, tmp_auc_df, on = ['cell', 'similarity_type', 'l1', 'l2', 'auc'], how = 'left' ) # concat
    else:
        tmp_df = pd.merge( tmp_df, tmp_auc_df, on = ['cell', 'similarity_type', 'l1', 'l2'], how = 'left' ) # concat
    
    
    # ----- 各関係タイプ、細胞タイプにおけるBED AUC最大の行を選択 ----- #
    tmp_df = tmp_df.reset_index(drop=True)
    tmp_df = tmp_df.loc[ tmp_df.groupby(by = ['similarity_type'])['auc'].idxmax() ]
    if v_type == 'auc':
        tmp_df = tmp_df.drop( ['cell', 'similarity_type', 'l1', 'l2'], axis=1) # 列を除く
    else:
        tmp_df = tmp_df.drop( ['cell', 'similarity_type', 'l1', 'l2', 'auc'], axis=1) # 列を除く
    tmp_df['ticks'] = 'multitask' # ticks
    lauc_df = pd.concat( [lauc_df, tmp_df], axis=0) # concat
    
    
    
    
    # ====== SNP profiling method ====== #
    for snp_type in snp_type_list:
    
        i_f = f"../data/{pert_type}/global_auc/{snp_type}.txt"
        tmp_df = pd.read_csv( i_f, sep = '\t' )
        tmp_df['ticks'] = snp_type # SNP profiling methodの種類
        tmp_df = tmp_df[[v_type, 'ticks']] # 列を選択
        lauc_df = pd.concat( [lauc_df, tmp_df], axis = 0 ) # concat
        
        
        
    # ====== Neural network ===== #
    for nn_type in nn_type_list:
            
        if nn_type == 'TargetFeature_VAE_DiseaseFeature_VAE':
            i_f = f'../data/{pert_type}/global_auc/nn_{nn_type}.txt'
            
        elif nn_type == 'TargetFeature_VQ_VAE_Parallel_DiseaseFeature_VQ_VAE_Parallel':
            i_f = f'../data/{pert_type}/global_auc/nn_{nn_type}.txt'
    
        else:
            i_f = "None"
    
        tmp_df = pd.read_csv( i_f, sep = '\t' )
    
        # ----- AUC最大のパラメータを選択 ----- #
        tmp_df = tmp_df.dropna( subset=['auc'] ).reset_index(drop = True) # aucがnanの行を除く
        tmp_df = pd.DataFrame(tmp_df.loc[ tmp_df['auc'].idxmax() ]).T # aucが最大のパラメータを選択
        tmp_df = tmp_df[[v_type]] # 列を選択
        tmp_df['pairwise_type'] = nn_type # model type
        tmp_df['ticks'] = [ f'NN_All_{p}' 
                          for p in tmp_df['pairwise_type'] ] # ticks
        tmp_df = tmp_df.drop( ['pairwise_type'], axis = 1)
    
        lauc_df = pd.concat( [ lauc_df, tmp_df], axis = 0) # concat
        
        
    # ====== Pi model ===== #
    for nn_type in pi_type_list:
        if nn_type == 'TargetFeature_VAE_DiseaseFeature_VAE':
            i_f = f'../data/{pert_type}/global_auc/pi_{nn_type}.txt'
            
        elif nn_type == 'TargetFeature_VQ_VAE_Parallel_DiseaseFeature_VQ_VAE_Parallel':
            i_f = f'../data/{pert_type}/global_auc/pi_{nn_type}.txt'
    
        else:
            i_f = "None"
    
            
        tmp_df = pd.read_csv( i_f, sep = '\t' )
    
        # ----- AUC最大のパラメータを選択 ----- #
        tmp_df = tmp_df.dropna( subset=['auc'] ).reset_index(drop = True) # aucがnanの行を除く
        tmp_df = pd.DataFrame(tmp_df.loc[ tmp_df['auc'].idxmax() ]).T # aucが最大のパラメータを選択
        tmp_df = tmp_df[[v_type]] # 列を選択
        tmp_df['pairwise_type'] = nn_type # model type
        tmp_df['ticks'] = [ f'Pi_All_{p}' 
                          for p in tmp_df['pairwise_type'] ] # ticks
        tmp_df = tmp_df.drop( ['pairwise_type'], axis = 1)
    
        lauc_df = pd.concat( [ lauc_df, tmp_df], axis = 0) # concat
    
    
    
    
    # ==============================================================
    # Global AUC  (multiview vs. singleview vs. multitask)
    # ==============================================================
    
    # ^^^^^ データを選択 ^^^^^ #
    tick_list =  list(color_dict.keys())
    tick_list.reverse()
    tmp_df = lauc_df[ lauc_df['ticks'].isin(tick_list) ]# 比較する条件を選択
    max_value = np.max( tmp_df[v_type] ) # max of the number of unlabeled data
        
    
    # ^^^^^ GDA_Allについて、次数ごとにboxplotを作成 ^^^^^ #
    fig = plt.figure(figsize=[10,10])
    ax = fig.add_subplot(1, 1, 1)
    sns.barplot(  
        data= tmp_df, x='ticks', y=v_type, 
        palette= color_dict, edgecolor = 'k',
        ax=ax, 
        order= tick_list,
        linewidth=4
    )
    
    
    # ----- Setting ----- #
    plt.xlabel('Method')
    plt.ylabel('Global {}'.format(v_type.upper()))
    plt.xticks(rotation = 45, ha = 'right')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper left',bbox_to_anchor=(1.05,1))
    plt.tick_params(labelsize=28, width = 4, length = 10) # メモリの太さ
    if v_type == 'auc':
        plt.ylim( [0.4, max_value+0.01] )
    
    # ----- 枠線 ----- #
    spines = 3 # 枠線の太さ
    ax.spines["left"].set_linewidth(spines) # 枠線を太くする
    ax.spines["bottom"].set_linewidth(spines) # 枠線を太くする
    plt.gca().spines['right'].set_visible(False) # 枠線を消す
    plt.gca().spines['top'].set_visible(False) # 枠線を消す
    
    # ----- Ssve the data ----- #
    o_dir = f"../data/{pert_type}/plot_paper/{v_type}"
    os.makedirs(o_dir, exist_ok=True)
    o_f = f'{o_dir}/global.png'
    plt.savefig(o_f, bbox_inches = 'tight')
    
    
    # ----- 予測精度の値を保存 ------ #
    o_f = f'{o_dir}/global.txt'
    tmp_df.to_csv( o_f, sep = '\t', index = None )

    

lauc_df.head()


# # In[47]:


# tmp_df = lauc_df.copy()
# tmp_df['count'] = 1
# tmp_df.groupby( by = 'ticks')['count'].sum()


# In[ ]:





# In[ ]:





# In[ ]:




