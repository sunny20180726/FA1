#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import random
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import datetime
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

    data1 = pd.read_csv('PPD_LogInfo_3_1_Training_Set.csv', header=0)
    data2 = pd.read_csv('PPD_Training_Master_GBK_3_1_Training_Set.csv', header=0, encoding='gbk')
    data3 = pd.read_csv('PPD_Userupdate_Info_3_1_Training_Set.csv', header=0)
    data2.head(5)
    data2.columns
    data2['city_match'] = data2.apply(lambda x: int(x.UserInfo_2 == x.UserInfo_4 == x.UserInfo_8 == x.UserInfo_20),
                                      axis=1)
    del data2['UserInfo_2']
    del data2['UserInfo_4']
    del data2['UserInfo_8']
    del data2['UserInfo_20']
    data1['logInfo'] = data1['LogInfo3'].map(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d'))
    data1['Listinginfo'] = data1['Listinginfo1'].map(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d'))
    data1['ListingGap'] = data1[['logInfo','Listinginfo']].apply(lambda x: (x[1]-x[0]).days,axis = 1)

    timeWindows = u_explore.time_window_selection(data1, 'ListingGap', range(30, 361, 30))

