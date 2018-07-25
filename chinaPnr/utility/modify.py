#!/usr/bin/env python
# -*- coding:utf-8 -*-

# Index
# ----------------------------------------
# makeup_miss_for_num        按照指定的方法对数据集的数字类型数据进行填充
# makeup_miss_for_1_num     按照指定的方法对数据集的指定列数字类型数据进行填充
# makeup_miss_for_str     按照指定的方法对数据集的字符串类型数据进行填充
# makeup_miss_for_1_str   按照指定的方法对数据集的指定列字符串类型数据进行填充
# density_encoder        对类别变量进行密度编码 包括Nan
# drop_str_missing_over_pcnt   删除缺失率超过阈值的字符型变量 默认阈值0.5
# drop_num_missing_over_pcnt  删除缺失率超过阈值的连续型变量 默认阈值0.3
# bad_rate_encoding 计算类别型变量bad_rate 并排序输出

import random
import numpy as np
import pandas as pd
import chinaPnr.utility.explore as u_explore


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


def drop_str_missing_over_pcnt(p_df, p_str_var_list, p_threshould=0.5):
    """
    删除缺失率超过阈值的字符型变量 默认阈值0.5
    :param p_df:
    :param p_str_var_list:
    :param p_threshould:
    :return:
    """
    for col in p_str_var_list:
        missing_rate = u_explore.missing_categorial_for_1(p_df, col)
        print('{0} has missing rate as {1}'.format(col, missing_rate))
        if missing_rate > p_threshould:
            p_str_var_list.remove(col)
            del p_df[col]
        if 0 < missing_rate < p_threshould:
            # In this way we convert NaN to NAN, which is a string instead of np.nan
            p_df[col] = p_df[col].map(lambda x: str(x).upper())
    print("function drop_str_missing_over_pcnt finished!...................")


def drop_num_missing_over_pcnt(p_df, p_num_var_list, p_threshould=0.3):
    """
    删除缺失率超过阈值的连续型变量 默认阈值0.3
    :param p_df:
    :param p_num_var_list:
    :param p_threshould:
    :return:
    """
    # p_df=allData
    # p_num_var_list = number_var_list
    # p_threshould=0.3
    for col in p_num_var_list:
        missing_rate = u_explore.missing_continuous_for_1(p_df, col)
        print('{0} has missing rate as {1}'.format(col, missing_rate))
        if missing_rate > p_threshould:
            del p_df[col]
            print('we delete variable {} because of its high missing rate'.format(col))
    print("function drop_num_missing_over_pcnt finished!...................")


def bad_rate_encoding(p_df, p_col, p_target):
    """
    计算类别型变量bad_rate 并排序输出
    :param p_df:
    :param p_col:
    :param p_target:
    :return:
    """
    total = p_df.groupby([p_col])[p_target].count()
    total = pd.DataFrame({"total": total})
    bad = p_df.groupby([p_col])[p_target].sum()
    bad = pd.DataFrame({"bad": bad})
    regroup = total.merge(bad, left_index=True, right_index=True, how="left")
    regroup.reset_index(level=0, inplace=True)
    regroup["bad_rate"] = regroup.apply(lambda x: x.bad*1.0/x.total, axis=1)
    br_dict = regroup[[p_col, "bad_rate"]].set_index([p_col]).to_dict(orient="index")
    bad_rate_enconding = p_df[p_col].map(lambda x: br_dict[x]["bad_rate"])
    return {"encoding": bad_rate_enconding, "br_rate": br_dict}


def max_badrate_for_string_1(p_df, p_col):
    """
    计算某个类别型变量最大类别占比
    :param p_df:
    :param p_col:
    :return:
    """
    n = p_df.shape[0]
    total = p_df.groupby([p_col])[p_col].count()
    pcnt = total*1.0/n
    return max(pcnt)


def merge_bad0(p_df, p_col, p_target):
    """
    合并没有bad样本的组 将其合并到bad_rate最小且不为0的一组
    :param p_df:
    :param p_col:
    :param p_target:
    :return:
    """
    total = p_df.groupby([p_col])[p_target].count()
    total = pd.DataFrame({'total': total})
    bad = p_df.groupby([p_col])[p_target].sum()
    bad = pd.DataFrame({'bad': bad})
    regroup = total.merge(bad, left_index=True, right_index=True, how='left')
    regroup.reset_index(level=0, inplace=True)
    regroup['bad_rate'] = regroup.apply(lambda x: x.bad*1.0/x.total, axis=1)
    regroup = regroup.sort_values(by='bad_rate')
    col_regroup = [[i] for i in regroup[p_col]]
    for i in range(regroup.shape[0]):
        col_regroup[1] = col_regroup[0] + col_regroup[1]
        col_regroup.pop(0)
        if regroup['bad_rate'][i+1] > 0:
            break
    new_group = {}
    for i in range(len(col_regroup)):
        for g2 in col_regroup[i]:
            new_group[g2] = 'Bin '+str(i)
    return new_group


def calc_woe_iv(p_df, p_col, p_target):
    """
    计算WOE
    :param p_df:
    :param p_col:
    :param p_target:
    :return:
    """
    total = p_df.groupby([p_col])[p_target].count()
    total = pd.DataFrame({'total': total})
    bad = p_df.groupby([p_col])[p_target].sum()
    bad = pd.DataFrame({'bad': bad})
    regroup = total.merge(bad, left_index=True, right_index=True, how='left')
    regroup.reset_index(level=0, inplace=True)
    n = sum(regroup['total'])
    b = sum(regroup['bad'])
    regroup['good'] = regroup['total'] - regroup['bad']
    g = n - b
    regroup['bad_pcnt'] = regroup['bad'].map(lambda x: x*1.0/b)
    regroup['good_pcnt'] = regroup['good'].map(lambda x: x * 1.0 / g)
    regroup['WOE'] = regroup.apply(lambda x: np.log(x.good_pcnt*1.0/x.bad_pcnt),axis = 1)
    woe_dict = regroup[[p_col, 'WOE']].set_index(p_col).to_dict(orient='index')
    for k, v in woe_dict.items():
        woe_dict[k] = v['WOE']
    iv = regroup.apply(lambda x: (x.good_pcnt-x.bad_pcnt)*np.log(x.good_pcnt*1.0/x.bad_pcnt),axis = 1)
    iv = sum(iv)
    return {"WOE": woe_dict, 'IV': iv}


def woe_iv_for_string(p_df, p_str_var_list, p_target, p_deleted_var_list, p_encoded_var_list, p_merged_var_list,
                      p_var_iv_list, p_var_woe_list):
    """
    处理类别型变量，并计算WOE、IV：
    1、如果类别个数大于5 则进行浓度编码 转化为连续性变量 参与后续连续型变量进行WOE、IV的计算
    2、如果类别个数小于等于5 则：
    2.1、如果某一类别占全部类别的90%或以上直接删除该变量 进行WOE、IV的计算
    2.2、如果某一类别bad_rate为0 则和最小的bad_rate进行合并 进行WOE、IV的计算
    :param p_df:
    :param p_str_var_list:
    :param p_deleted_var_list: 删除变量的list
    :param p_encoded_var_list: 浓度编码的list
    :param p_merged_var_list: 进行合并的list
    :param p_var_iv_list: IV列表
    :param p_var_woe_list: WOE列表
    :return:
    """
    # for col in p_str_var_list:
    #     p_df[col] = p_df[col].map(lambda x: str(x).upper())

    for var in p_str_var_list:
        print('we are processing {}'.format(var))
        if len(set(p_df[var])) > 5:
            print('{} is encoded with bad rate'.format(var))
            col0 = str(var) + '_encoding'
            # (1), 计算不良率并对原始值进行编码 删除并登记原变量
            encoding_result = bad_rate_encoding(p_df, var, p_target)
            p_df[col0], br_encoding = encoding_result['encoding'], encoding_result['br_rate']
            del p_df[var]
            p_deleted_var_list.append(var)
            # (2), 保存编码结果，包括新的列名和不良率
            p_encoded_var_list[var] = [col0, br_encoding]
        else:
            max_pcnt = max_badrate_for_string_1(p_df, var)
            if max_pcnt >= 0.9:
                print('{} is deleted because of large percentage of single bin'.format(var))
                del p_df[var]
                p_deleted_var_list.append(var)
                continue
            bad_bin = p_df.groupby([var])[p_target].sum()
            if min(bad_bin) == 0:
                print('{} has 0 bad sample!'.format(var))
                col1 = str(var) + '_mergeByBadRate'
                # (1), 确定如何合并类别
                merge_bin = merge_bad0(p_df, var, p_target)
                # (2), 将原始数据转换为合并数据
                p_df[col1] = p_df[var].map(merge_bin)
                del p_df[var]
                p_deleted_var_list.append(var)
                max_pcnt = max_badrate_for_string_1(p_df, col1)
                if max_pcnt > 0.9:
                    print('{} is deleted because of large percentage of single bin'.format(var))
                    del p_df[var]
                    p_deleted_var_list.append(var)
                    continue
                # (3) iif the merged data satisify the requirement, we keep it
                p_merged_var_list[var] = [col1, merge_bin]
                woe_iv = calc_woe_iv(p_df, col1, p_target)
                p_var_woe_list[col1] = woe_iv['WOE']
                p_var_iv_list[col1] = woe_iv['IV']
            else:
                woe_iv = calc_woe_iv(p_df, var, 'target')
                p_var_woe_list[var] = woe_iv['WOE']
                p_var_iv_list[var] = woe_iv['IV']
    print("function woe_iv_for_string finished!...................")


def inner_chi2(p_df, p_total_col, p_bad_col, p_overall_rate):
    """
    计算卡方
    :param p_df:
    :param p_total_col:
    :param p_bad_col:
    :param p_overall_rate:
    :return:
    """
    df2 = p_df.copy()
    df2['expected'] = p_df[p_total_col].apply(lambda x: x * p_overall_rate)
    combined = zip(df2['expected'], df2[p_bad_col])
    chi = [(i[0]-i[1])**2/i[0] for i in combined]

    return_chi2 = sum(chi)
    return return_chi2


def inner_assign_group(p_value, p_bin):
    n = len(p_bin)
    if p_value <= min(p_bin):
        return min(p_bin)
    elif p_value > max(p_bin):
        return 10e10
    else:
        for i in range(n-1):
            if p_bin[i] < p_value <= p_bin[i + 1]:
                return p_bin[i + 1]


def assign_bin(p_x, p_cutoff_points):
    """
    :param p_x: the value of variable
    :param p_cutoff_points: the ChiMerge result for continous variable
    :return: bin number, indexing from 0
    for example, if cutOffPoints = [10,20,30], if x = 7, return Bin 0. If x = 35, return Bin 3
    """
    num_bin = len(p_cutoff_points) + 1
    if p_x <= p_cutoff_points[0]:
        return 'Bin 0'
    elif p_x > p_cutoff_points[-1]:
        return 'Bin {}'.format(num_bin-1)
    else:
        for i in range(0, num_bin-1):
            if p_cutoff_points[i] < p_x <=  p_cutoff_points[i + 1]:
                return 'Bin {}'.format(i+1)


def bad_rate_monotone(p_df, sort_by_var, p_target, p_special_attribute=[]):
    """
    :param p_df: the dataset contains the column which should be monotone with the bad rate and bad column
    :param sort_by_var: the column which should be monotone with the bad rate
    :param p_target: the bad column
    :param p_special_attribute: some attributes should be excluded when checking monotone
    :return:
    """
    df2 = p_df.loc[~p_df[sort_by_var].isin(p_special_attribute)]
    df2 = df2.sort_values([sort_by_var])
    total = df2.groupby([sort_by_var])[p_target].count()
    total = pd.DataFrame({'total': total})
    bad = df2.groupby([sort_by_var])[p_target].sum()
    bad = pd.DataFrame({'bad': bad})
    regroup = total.merge(bad, left_index=True, right_index=True, how='left')
    regroup.reset_index(level=0, inplace=True)
    combined = zip(regroup['total'], regroup['bad'])
    bad_rate = [x[1]*1.0/x[0] for x in combined]
    badrate_monotone = [bad_rate[i] < bad_rate[i+1] for i in range(len(bad_rate)-1)]
    monotone = len(set(badrate_monotone))
    if monotone == 1:
        return True
    else:
        return False


def chi_merge_max_interval(p_df, p_col, p_target, p_max_bin=5, p_special_attribute=[]):

    # p_df = pd1
    # p_col = "day"
    # p_target = "chrun"
    # p_max_bin = 5

    col_level = sorted(list(set(p_df[p_col])))
    if len(col_level) <= p_max_bin:
        print("The number of original levels for {} is less than or equal to max intervals".format(p_col))
        return []

    temp_df2 = p_df.copy()
    n_distinct = len(col_level)
    if n_distinct > 100:
        ind_x = [int(i/100.00 * n_distinct) for i in range(1, 100)]
        split_x = [col_level[i] for i in ind_x]
        temp_df2["temp"] = temp_df2[p_col].map(lambda x: inner_assign_group(x, split_x))
    else:
        temp_df2['temp'] = p_df[p_col]

    total = temp_df2.groupby(['temp'])[p_target].count()
    total = pd.DataFrame({'total': total})
    bad = temp_df2.groupby(['temp'])[p_target].sum()
    bad = pd.DataFrame({'bad': bad})
    regroup = total.merge(bad, left_index=True, right_index=True, how='left')
    regroup.reset_index(level=0, inplace=True)
    N = sum(regroup['total'])
    B = sum(regroup['bad'])
    # the overall bad rate will be used in calculating expected bad count
    overall_rate = B * 1.0 / N

    col_level = sorted(list(set(temp_df2['temp'])))
    group_intervals = [[i] for i in col_level]
    group_num = len(group_intervals)

    while len(group_intervals) > p_max_bin:
        chisq_list = []
        for interval in group_intervals:
            df2b = regroup.loc[regroup['temp'].isin(interval)]
            chisq = inner_chi2(df2b, 'total', 'bad', overall_rate)
            chisq_list.append(chisq)
        min_position = chisq_list.index(min(chisq_list))
        if min_position == 0:
            combined_position = 1
        elif min_position == group_num - 1:
            combined_position = min_position - 1
        else:
            if chisq_list[min_position - 1] <= chisq_list[min_position + 1]:
                combined_position = min_position - 1
            else:
                combined_position = min_position + 1
        group_intervals[min_position] = group_intervals[min_position] + group_intervals[combined_position]
        group_intervals.remove(group_intervals[combined_position])
        group_num = len(group_intervals)

    group_intervals = [sorted(i) for i in group_intervals]
    cut_off_points = [max(i) for i in group_intervals[:-1]]
    cut_off_points = p_special_attribute + cut_off_points
    return cut_off_points


def woe_iv_for_num(p_df, p_str_num_list, p_target, p_deleted_var_list, p_var_iv_list, p_var_woe_list,
                   p_var_cutoff_list):
    for col in p_str_num_list:
        print("{} is in processing".format(col))
        col1 = str(col) + '_Bin'
        # (1), 分割连续变量并保存分割点。-1是一种特殊情况，把它分成一组。
        # if -1 in set(p_df[col]):
        #     special_attribute = [-1]
        # else:
        #     special_attribute = []
        # special_attribute = []
        cut_off_points = chi_merge_max_interval(p_df, col, p_target)
        p_var_cutoff_list[col] = cut_off_points
        p_df[col1] = p_df[col].map(lambda x: assign_bin(x, cut_off_points))

        # (2), 检查是否单调 不单调的话 减少分类
        BRM = bad_rate_monotone(p_df, col1, p_target)
        if not BRM:
            for bins in range(4, 1, -1):
                cut_off_points = chi_merge_max_interval(p_df, col, p_target, p_max_bin=bins)
                p_df[col1] = p_df[col].map(lambda x: assign_bin(x, cut_off_points))
                BRM = bad_rate_monotone(p_df, col1, p_target)
                if BRM:
                    break
            p_var_cutoff_list[col] = cut_off_points

        # (3), 检查某一分类是否占比90%以上 有的话 删除
        maxPcnt = max_badrate_for_string_1(p_df, col1)
        if maxPcnt > 0.9:
            #del p_df[col1]
            p_deleted_var_list.append(col)
            print('we delete {} because the maximum bin occupies more than 90%'.format(col))
            continue
        woe_iv = calc_woe_iv(p_df, col1, 'target')
        p_var_woe_list[col] = woe_iv['IV']
        p_var_iv_list[col] = woe_iv['WOE']
        del p_df[col]

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





