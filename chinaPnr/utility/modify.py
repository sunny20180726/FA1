#!/usr/bin/env python
# -*- coding:utf-8 -*-

# Index
# ----------------------------------------
# makeup_miss_for_num        按照指定的方法对数据集的数字类型数据进行填充
# makeup_miss_for_1_num     按照指定的方法对数据集的指定列数字类型数据进行填充
# makeup_miss_for_str     按照指定的方法对数据集的字符串类型数据进行填充
# makeup_miss_for_1_str   按照指定的方法对数据集的指定列字符串类型数据进行填充
# density_encoder        对类别变量进行密度编码 包括Nan

import random
import numpy as np
import pandas as pd


def makeup_num_miss(p_df, p_var_list, p_method):
    """
    按照指定的方法对数据集的数字类型数据进行填充
    :param p_df:数据集
    :param p_var_list:数字型变量名称list
    :param p_method:填充方法 'MEAN' 'RANDOM' 'PERC50'
    :return:
    """

    # df=dataAll
    # var_list=var_list
    # method='MEAN'
    # col= 'age1'

    if p_method.upper() not in ['MEAN', 'RANDOM', 'PERC50']:
        print('Please specify the correct treatment method for missing continuous variable:Mean or Random or ' +
              'PERC50 !')
        return

    for col in set(p_var_list):
        makeup_num_miss_for_1(p_df, col, p_method)

    print("function makeup_miss_for_num finished!...................")



def makeup_num_miss_for_1(p_df, p_var, p_method):
    """
    按照指定的方法对数据集的指定列数字类型数据进行填充
    :param p_df:数据集
    :param p_var:数字型变量名称
    :param p_method:填充方法 'MEAN' 'RANDOM' 'PERC50'
    :return:
    """

    # df=dataAll
    # var_list=var_list
    # method='MEAN'
    # col= 'age1'

    if p_method.upper() not in ['MEAN', 'RANDOM', 'PERC50']:
        print('Please specify the correct treatment method for missing continuous variable:Mean or Random or ' +
              'PERC50!')
        return

    valid_df = p_df.loc[~np.isnan(p_df[p_var])][[p_var]]
    desc_state = valid_df[p_var].describe()
    value_mu = desc_state['mean']
    value_perc50 = desc_state['50%']

    # makeup missing
    if p_method.upper() == 'PERC50':
        p_df[p_var].fillna(value_perc50, inplace=True)
    elif p_method.upper() == 'MEAN':
        p_df[p_var].fillna(value_mu, inplace=True)
    elif p_method.upper() == 'RANDOM':
        # 得到列索引
        index_col = list(p_df.columns).index(p_var)
        for i in range(p_df.shape[0]):
            if np.isnan(p_df.iloc[i][p_var]):
                p_df.iloc[i, index_col] = random.sample(set(valid_df[p_var]), 1)[0]

    print("function makeup_miss_for_1_num("+p_var+") finished!...................")


def makeup_str_miss(p_df, p_str_var_list, p_method):
    """
    按照指定的方法对数据集的字符串类型数据进行填充
    :param p_df: 数据集
    :param p_str_var_list: 字符类型变量名称list
    :param p_method: 填充方法 'MODE' 'RANDOM'
    :return:
    """

    if p_method.upper() not in ['MODE', 'RANDOM']:
        print('Please specify the correct treatment method for missing continuous variable:MODE or Random!')
        return

    for var in p_str_var_list:
        makeup_str_miss_for_1(p_df, var, p_method)

    print("function makeup_miss_for_str finished!...................")


def makeup_str_miss_for_1(p_df, p_var, p_method):
    """
    按照指定的方法对数据集的指定列字符串类型数据进行填充
    :param p_df: 数据集
    :param p_var: 字符类型变量名称list
    :param p_method: 填充方法 'MODE' 'RANDOM'
    :return:
    """
    # p_df=pd1
    # p_var="var1"
    # p_method='MODE'
    # i = 3

    if p_method.upper() not in ['MODE', 'RANDOM']:
        print('Please specify the correct treatment method for missing continuous variable:MODE or Random!')
        return p_df

    valid_df = p_df.loc[p_df[p_var] == p_df[p_var]][[p_var]]

    if valid_df.shape[0] != p_df.shape[0]:
        index_var = list(p_df.columns).index(p_var)
        if p_method.upper() == "MODE":
            dict_var_freq = {}
            num_recd = valid_df.shape[0]
            for v in set(valid_df[p_var]):
                dict_var_freq[v] = valid_df.loc[valid_df[p_var] == v].shape[0]*1.0/num_recd
            mode_val = max(dict_var_freq.items(), key=lambda x: x[1])[0]
            p_df[p_var].fillna(mode_val, inplace=True)
        elif p_method.upper() == "RANDOM":
            list_dict = list(set(valid_df[p_var]))
            for i in range(p_df.shape[0]):
                if p_df.loc[i][p_var] != p_df.loc[i][p_var]:
                    p_df.iloc[i, index_var] = random.choice(list_dict)

    print("function makeup_miss_for_1_str("+p_var+") finished!...................")


def density_encoder_for_1(p_df, p_var, p_target):
    """
    对类别变量进行浓度编码 包括Nan
    :param p_df: 数据集
    :param p_var: 要分析的类别型变量的变量名
    :param p_target: 响应变量名
    :return: 返回每个类别对应的响应率
    """
    # df = data_all
    # col = 'marital'
    # target = col_target

    dict_encoder = {}
    for v in set(p_df[p_var]):
        if v == v:
            sub_df = p_df[p_df[p_var] == v]
        else:
            xlist = list(p_df[p_var])
            nan_ind = [i for i in range(len(xlist)) if xlist[i] != xlist[i]]
            sub_df = p_df.loc[nan_ind]
        dict_encoder[v] = sum(sub_df[p_target]) * 1.0 / sub_df.shape[0]
    new_col = [dict_encoder[i] for i in p_df[p_var]]
    print("function density_encoder_for_1(" + p_var + ") finished!...................")
    return new_col


def density_encoder(p_df, p_str_list, p_target):
    """
    对数据集所有字符型特征进行浓度编码
    :param p_df: 数据集
    :param p_str_list: 字符型变量名称list
    :param p_target: 目标变量
    :return:
    """
    for var in p_str_list:
        p_df[var] = density_encoder_for_1(p_df, var, p_target)
    print("function density_encoder finished!...................")


def standard_max_min(p_df, p_var, p_target):
    """
    最大最小值归一化
    :param p_df:
    :param p_var:
    :param p_target:
    :return:
    """
    return


def standard_std(p_df, p_var, p_target):
    """
    标准差归一化
    :param p_df:
    :param p_var:
    :param p_target:
    :return:
    """
    return


def bin_best_ks(p_df, p_var, p_target):
    """
    ks分箱法 不推荐
    :param p_df:
    :param p_var:
    :param p_target:
    :return:
    """
    return





