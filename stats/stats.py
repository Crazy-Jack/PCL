import pandas as pd
import os
import argparse
import numpy as np

def entropy(x):
    '''
    H(x)
    '''
    unique, count = np.unique(x, return_counts=True, axis=0)
    prob = count/len(x)
    H = np.sum((-1)*prob*np.log2(prob))

    return H

def joint_entropy(x, y):
    '''
    H(x,y)
    '''
    combine = np.c_[x, y]
    return entropy(combine)

def conditional_entropy(x, y):
    '''
    H(x|y)
    '''
    return joint_entropy(x, y) - entropy(y)

def mutual_information(x, y):
    '''
    I(x;y)
    '''
    return entropy(x) - conditional_entropy(x, y)

def get_MI_and_H(df, gran_lvl):
    '''
    input:
        - df: pandas dataframe
        - gran_lvl: latent class granularity level

    output:
        - MI: mutual information of Y and T
        - H_Y_T: conditional entropy of Y given T
        - H_T_Y: conditional entropy of T given Y
    '''

    T = df['class'].tolist()
    Y = df[gran_lvl].tolist()

    H_Y_T = conditional_entropy(Y, T)
    H_T_Y = conditional_entropy(T, Y)
    MI = mutual_information(Y, T)

    return MI, H_Y_T, H_T_Y

def add_stat_to_df(df, args):
    '''
    given a df, compute the Mutual information and conditional entropy for all latent labels existed in df.

    input:
        - df: pandas dataframe

    output:
        - df: pandas dataframe with H and MI
    
    '''
    valid_indices = [index for index, row in df.iterrows() if type(index) == int and index >=0]
    df_sample_only = df.loc[valid_indices]

    df_sample_only['instance_id'] = list(range(len(df_sample_only)))
    gran_lvl_list = [col for col in df_sample_only.columns if 'label_gran' in col or 'class' in col or 'instance_id' in col]

    H_Y_T_ = {}
    H_T_Y_ = {}
    MI_ = {}
    H_y_ = {}
    for gran_lvl in gran_lvl_list:
        MI_[gran_lvl], H_Y_T_[gran_lvl], H_T_Y_[gran_lvl] = get_MI_and_H(df_sample_only, gran_lvl)
        H_y_[gran_lvl] = entropy(df_sample_only[gran_lvl].tolist())
        
    H_Y_T_ = pd.Series(H_Y_T_, name='H(Y|T)')
    H_T_Y_ = pd.Series(H_T_Y_, name='H(T|Y)')
    MI_ = pd.Series(MI_, name='I(Y;T)')

    H_y_ = pd.Series(H_y_, name='H(Y)')
    df_stats = pd.concat([H_Y_T_, H_T_Y_, MI_, H_y_], axis=1)
#     df_stats['I(Y;T)/H(Y|T)'] = df_stats.apply(lambda row: row['I(Y;T)']/row['H(Y|T)'], axis=1)
#     df_stats['H(T|Y)/H(Y|T)'] = df_stats.apply(lambda row: row['H(T|Y)']/row['H(Y|T)'], axis=1)
    df_stats['log(I(Y;T)/log(H(Y|T)))'] = df_stats.apply(lambda row: np.log(row['I(Y;T)']/np.log(row['H(Y|T)'])), axis=1)
    

    return df_stats



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data Processing for computing H and MI')
    parser.add_argument("--data_path", type=str, default='', help="dataset path")
    parser.add_argument("--meta_data_filename", type=str, default='meta_data_train.csv', help='name of mata data file')
    parser.add_argument("--customized_name", type=str, default='', help='customized name')
    parser.add_argument("--save_path", type=str, default='', help="save path")

    args = parser.parse_args()
    if not args.save_path:
        args.save_path = args.data_path

    df = pd.read_csv(os.path.join(args.data_path, args.meta_data_filename), index_col=0)
    df_stats = add_stat_to_df(df, args)
    df_stats.to_csv(os.path.join(args.save_path, args.meta_data_filename.split('.')[0]+'_stats_' + args.customized_name + '.csv'))
