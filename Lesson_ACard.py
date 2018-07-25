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



if __name__ == '__main__':
    # ##########################################################
    # #################原始数据处理              #################
    # ##########################################################
    # 根目录
    path_root = os.getcwd()
    # 路径
    path_explore_result = path_root+'\\results\\explore'
    u_others.create_path(path_explore_result)
    # ID的字段名
    col_id = 'Idx'
    # 目标字段名
    col_target = 'target'
    # 排除字段名
    drop_var = ['ListingInfo']

    allData = pd.read_csv('allData_0.csv', header=0, encoding='gbk')
    # 时间窗口
    # allData = pd.read_csv('allData_00.csv', header=0, encoding='gbk')
    # timeWindows = u_explore.time_window_selection(p_df=allData, p_days_col="ListingGap",
    #                                               p_time_windows=range(30, 361, 30),
    #                                               p_save_file=path_explore_result+'\\timeWindows.png')
    # 得到类别型和数字型变量名列表并保存
    string_var_list, number_var_list, all_var_list = u_explore.get_list_for_number_str_col(p_df=allData,
                                                                                           p_col_id=col_id,
                                                                                           p_col_target=col_target,
                                                                                           p_drop_col=drop_var)
    u_others.list2txt(path_explore_result, "var_string_list.csv", string_var_list)
    u_others.list2txt(path_explore_result, "var_number_list.csv", number_var_list)
    u_others.list2txt(path_explore_result, "all_var_list.csv", all_var_list)
    # todo 调用小程序手动调整
    # todo 如果重新跑数据 或者调整字段则 用txt2list()重新加载即可
    # string_var_list = txt2list(path_explore_result+"\\var_string_list.csv")
    # number_var_list = txt2list(path_explore_result+"\\var_number_list.csv")
    # all_var_list = txt2list(path_explore_result+"\\all_var_list.csv")
    ####################################
    # Step 3: 删除取值完全一样的数据；删除缺失过多的数据#
    ####################################
    '''
    去掉取值完全一样的数据
    '''
    for col in all_var_list:
        if len(set(allData[col])) == 1:
            print('delete {} from the dataset because it is a constant'.format(col))
            del allData[col]
            all_var_list.remove(col)
    string_var_list, number_var_list, all_var_list = u_explore.get_list_for_number_str_col(p_df=allData,
                                                                                           p_col_id=col_id,
                                                                                           p_col_target=col_target,
                                                                                           p_drop_col=drop_var)
    '''
    去掉缺失值超过阈值的变量 连续变量0.3 字符变量0.5
    '''
    u_modify.drop_num_missing_over_pcnt(p_df=allData,  p_num_var_list=number_var_list, p_threshould=0.3)
    u_modify.drop_str_missing_over_pcnt(p_df=allData,  p_str_var_list=string_var_list, p_threshould=0.5)
    string_var_list, number_var_list, all_var_list = u_explore.get_list_for_number_str_col(p_df=allData,
                                                                                           p_col_id=col_id,
                                                                                           p_col_target=col_target,
                                                                                           p_drop_col=drop_var)

    u_explore.missing_categorial(allData, string_var_list, path_explore_result+'\\missing_categorial.csv')
    u_explore.missing_continuous(allData, number_var_list, path_explore_result+'\\missing_num.csv')

    string_var_list, number_var_list, all_var_list = u_explore.get_list_for_number_str_col(p_df=allData,
                                                                                           p_col_id=col_id,
                                                                                           p_col_target=col_target,
                                                                                           p_drop_col=drop_var)
    u_others.list2txt(path_explore_result, "var_string_list.csv", string_var_list)
    u_others.list2txt(path_explore_result, "var_number_list.csv", number_var_list)
    u_others.list2txt(path_explore_result, "all_var_list.csv", all_var_list)

    allData_bk = allData.copy()
    ####################################
    # Step 3: 缺失值填补#
    ####################################
    u_modify.makeup_num_miss(allData,number_var_list,"PERC50")
    u_explore.missing_continuous(allData,number_var_list,path_explore_result+'\\missing_num02.csv')
    u_modify.makeup_str_miss(allData,string_var_list,"MODE")
    u_explore.missing_categorial(allData, string_var_list, path_explore_result+'\\missing_categorial02.csv')

    allData.to_csv('allData_1.csv', header=True,encoding='gbk', columns=allData.columns, index=False)
    ####################################
    # Step 3: 变量分组#
    ####################################
    trainData = pd.read_csv('allData_1.csv', header=0, encoding='gbk')
    string_var_list, number_var_list, all_var_list = u_explore.get_list_for_number_str_col(p_df=trainData,
                                                                                           p_col_id=col_id,
                                                                                           p_col_target=col_target,
                                                                                           p_drop_col=drop_var)
    for col in string_var_list:
        trainData[col] = trainData[col].map(lambda x: str(x).upper())

    deleted_features = []  # delete the categorical features in one of its single bin occupies more than 90%
    encoded_features = {}
    merged_features = {}
    var_iv_list = {}  # save the IV values for binned features
    var_woe_list = {}
    u_modify.woe_iv_for_string(p_df=trainData, p_str_var_list=string_var_list, p_target=col_target,
                               p_deleted_var_list=deleted_features, p_encoded_var_list=encoded_features,
                               p_merged_var_list=merged_features, p_var_iv_list=var_iv_list,
                               p_var_woe_list=var_woe_list)
    string_var_list, number_var_list, all_var_list = u_explore.get_list_for_number_str_col(p_df=trainData,
                                                                                           p_col_id=col_id,
                                                                                           p_col_target=col_target,
                                                                                           p_drop_col=drop_var)

    var_cutoff_list = {}
    u_modify.woe_iv_for_num(p_df=trainData, p_str_num_list=number_var_list, p_target=col_target,
                            p_deleted_var_list=deleted_features, p_var_iv_list=var_iv_list,
                            p_var_woe_list=var_woe_list, p_var_cutoff_list=var_cutoff_list)



















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