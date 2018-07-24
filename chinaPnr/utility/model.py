#! /usr/bin/env python
# -*- coding:utf-8 -*-


import numpy as np
import pandas as pd


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


def inner_bad_rate_encoding(p_df, p_col, p_target):
    """
    对类别型变量进行bad_rate排序
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


def bad_rate_monotone(p_df, p_var, p_target):
    """
    返回指标是否单调 用于判断分完箱之后是否需要继续合并 如果不单调就继续合并 否则就不用合并
    :param p_df: the dataset contains the column which should be monotone with the bad rate and bad column
    :param p_var: the column which should be monotone with the bad rate
    :param p_target: the bad column
    :return:
    """
    df2 = p_df.sort([p_var])
    total = df2.groupby([p_var])[p_target].count()
    total = pd.DataFrame({'total': total})
    bad = df2.groupby([p_var])[p_target].sum()
    bad = pd.DataFrame({'bad': bad})
    regroup = total.merge(bad, left_index=True, right_index=True, how='left')
    regroup.reset_index(level=0, inplace=True)
    combined = zip(regroup['total'], regroup['bad'])
    bad_rate = [x[1]*1.0/x[0] for x in combined]
    bad_rate_mono = [bad_rate[i] < bad_rate[i+1] for i in range(len(bad_rate)-1)]
    monotone = len(set(bad_rate_mono))
    if monotone == 1:
        return True
    else:
        return False


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