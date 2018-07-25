#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import datetime
import collections

import chinaPnr.utility.explore as u_explore
import chinaPnr.utility.modify as u_modify
import chinaPnr.utility.others as u_others
import chinaPnr.utility.sample as u_sample
import chinaPnr.utility.model as u_model
import chinaPnr.utility.assess as u_assess
# import io
# import sys
# import numbers
# import numpy as np
# from matplotlib import pyplot

import numbers



if __name__ == '__main__':
    # # ##########################################################
    # # #################原始数据处理              #################
    # # ##########################################################
    # # 根目录
    # path_root = os.getcwd()
    # # 路径
    # path_explore_result = path_root+'\\Result\\Explore'
    # u_others.create_path(path_explore_result)
    # # ID的字段名
    # col_id = 'CUST_ID'
    # # 目标字段名
    # col_target = 'CHURN_CUST_IND'
    #
    # # 合并数据
    # data_bank = pd.read_csv(path_root + '\\bankChurn.csv')
    # data_external = pd.read_csv(path_root + '\\ExternalData.csv')
    # data_all = pd.merge(data_bank, data_external, on=col_id)
    # data_all.head(5)
    # # #########################################################
    # # ###              数据探索                     #############
    # # #########################################################
    # # 得到类别型和数字型变量名列表并保存
    # string_var_list, number_var_list = u_explore.get_list_for_number_str_col(p_df=data_all, p_col_id=col_id,
    #                                                                          p_col_target=col_target)
    # u_others.list2txt(path_explore_result, "var_string_list.txt", string_var_list)
    # u_others.list2txt(path_explore_result, "var_number_list.txt", number_var_list)
    # # data_all[string_var_list]
    # # data_all[number_var_list]
    # # todo 调用小程序手动调整
    # # todo 如果重新跑数据 或者调整字段则 用txt2list()重新加载即可
    # # string_var_list = txt2list(path_explore_result+"\\var_string_list.txt")
    # # number_var_list = txt2list(path_explore_result+"\\var_number_list.txt")
    #
    # # 分别进行数字型变量和字符串变量的探索
    # u_explore.num_var_perf(p_df=data_all, p_var_list=number_var_list, p_target_var=col_target,
    #                        p_path=path_explore_result)
    # u_explore.str_var_pref(p_df=data_all, p_var_list=string_var_list, p_target_var=col_target,
    #                        p_path=path_explore_result)
    #
    # # 选择15个数字变量 看相关性
    # # corr_cols = random.sample(number_var_list, 15)
    # # sample_df = data_all[corr_cols]
    # # scatter_matrix(sample_df, alpha=0.2, figsize=(14, 8), diagonal='kde')
    # # plt.show()
    #
    # # 缺失值填充
    # u_modify.makeup_num_miss(p_df=data_all, p_var_list=number_var_list, p_method="MEAN")
    # u_modify.makeup_str_miss(p_df=data_all, p_str_var_list=string_var_list, p_method="MODE")
    #
    # # 浓度编码
    # u_modify.density_encoder(data_all, string_var_list, col_target)
    #
    # # 卡方分箱
    # cutoff_points = u_model.chi_merge_max_interval(data_all, "day", col_target, 5)
    # var_cutoff = {}
    # var_cutoff["day"] = cutoff_points
    # data_all["day"] = data_all["day"].map(lambda x: u_model.assign_bin(x, cutoff_points))


    def ChangeContent(x):
        y = x.upper()
        if y == '_MOBILEPHONE':
            y = '_PHONE'
        return y
    # ##########################################################
    # #################原始数据处理              #################
    # ##########################################################
    # 根目录
    path_root = os.getcwd()
    # 路径
    path_explore_result = path_root+'\\results\\explore'
    u_others.create_path(path_explore_result)
    # # ID的字段名
    col_id = 'Idx'
    # # 目标字段名
    col_target = 'target'

    allData = pd.read_csv('allData_0.csv',header = 0,encoding = 'gbk')
    allFeatures = list(allData.columns)
    allFeatures.remove('target')
    allFeatures.remove('Idx')
    allFeatures.remove('ListingInfo')

# numerical_var = []
# for col in allFeatures:
#     if len(set(allData[col])) == 1:
#         print('delete {} from the dataset because it is a constant'.format(col))
#         del allData[col]
#         allFeatures.remove(col)
#     else:
#         #uniq_vals = list(set(allData[col]))
#         #if np.nan in uniq_vals:
#             #uniq_vals.remove(np.nan)
#         uniq_valid_vals = [i for i in allData[col] if i == i]
#         uniq_valid_vals = list(set(uniq_valid_vals))
#         if len(uniq_valid_vals) >= 10 and isinstance(uniq_valid_vals[0], numbers.Real):
#             numerical_var.append(col)
#
# categorical_var = [i for i in allFeatures if i not in numerical_var]


# 去掉取值完全一样的数据
allFeatures = list(allData.columns)
for col in allFeatures:
    if len(set(allData[col])) == 1:
        print('delete {} from the dataset because it is a constant'.format(col))
        del allData[col]
        allFeatures.remove(col)
# 得到类别型和数字型变量名列表并保存
string_var_list, number_var_list = u_explore.get_list_for_number_str_col(p_df=allData, p_col_id=col_id,
                                                                         p_col_target=col_target)
u_others.list2txt(path_explore_result, "var_string_list.txt", string_var_list)
u_others.list2txt(path_explore_result, "var_number_list.txt", number_var_list)
# data_all[string_var_list]
# data_all[number_var_list]
# todo 调用小程序手动调整
# todo 如果重新跑数据 或者调整字段则 用txt2list()重新加载即可
# string_var_list = txt2list(path_explore_result+"\\var_string_list.txt")
# number_var_list = txt2list(path_explore_result+"\\var_number_list.txt")

'''
For each categorical variable, if the missing value occupies more than 50%, we remove it.
Otherwise we will use missing as a special status
'''
missing_pcnt_threshould_1 = 0.5
for col in string_var_list:
    missingRate = u_explore.missing_categorial_for_1(allData,col)
    print('{0} has missing rate as {1}'.format(col,missingRate))
    if missingRate > missing_pcnt_threshould_1:
        string_var_list.remove(col)
        del allData[col]
    if 0 < missingRate < missing_pcnt_threshould_1:
        # In this way we convert NaN to NAN, which is a string instead of np.nan
        allData[col] = allData[col].map(lambda x: str(x).upper())
u_explore.missing_categorial(allData,string_var_list,path_explore_result+'\\missing_categorial.csv')

allData_bk = allData.copy()
'''
For continuous variable, if the missing value is more than 30%, we remove it.
Otherwise we use random sampling method to make up the missing
'''
missing_pcnt_threshould_2 = 0.3
deleted_var = []
for col in number_var_list:
    missingRate = u_explore.missing_continuous_for_1(allData, col)
    print('{0} has missing rate as {1}'.format(col, missingRate))
    if missingRate > missing_pcnt_threshould_2:
        deleted_var.append(col)
        del allData[col]
        print('we delete variable {} because of its high missing rate'.format(col))
for var in deleted_var:
    number_var_list.remove(var)
u_explore.missing_continuous(allData,number_var_list,path_explore_result+'\\missing_num.csv')


####################################
# Step 3: 缺失值填补#
####################################
u_modify.makeup_num_miss(allData,number_var_list,"PERC50")
u_explore.missing_continuous(allData,number_var_list,path_explore_result+'\\missing_num02.csv')

allData.to_csv('allData_1.csv', header=True,encoding='gbk', columns = allData.columns, index=False)
####################################
# Step 3: Group variables into bins#
####################################
trainData = pd.read_csv('allData_1.csv',header = 0, encoding='gbk')
for col in string_var_list:
    #for Chinese character, upper() is not valid
    if col not in ['UserInfo_7','UserInfo_9','UserInfo_19','UserInfo_22','UserInfo_23','UserInfo_24','Education_Info3','Education_Info7','Education_Info8']:
        trainData[col] = trainData[col].map(lambda x: str(x).upper())




