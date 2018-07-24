from __future__ import division
from numpy.random import randn
import numpy as np
import os
import matplotlib.pyplot as plt
np.random.seed(12345)
plt.rc('figure', figsize=(10, 6))
from pandas import Series, DataFrame
import pandas as pd
import random



pd1 = pd.read_csv("data1.csv")


def assign_group(p_value, p_bin):
    n = len(p_bin)
    if p_value <= min(p_bin):
        return min(p_bin)
    elif p_value > max(p_bin):
        return 10e10
    else:
        for i in range(n-1):
            if p_bin[i] < p_value <= p_bin[i + 1]:
                return p_bin[i + 1]


def chi2(df, total_col, bad_col, overallRate):
    """
    计算卡方
    :param df:
    :param total_col:
    :param bad_col:
    :param overallRate:
    :return:
    """
    df2 = df.copy()
    df2['expected'] = df[total_col].apply(lambda x: x*overallRate)
    combined = zip(df2['expected'], df2[bad_col])
    chi = [(i[0]-i[1])**2/i[0] for i in combined]

    return_chi2 = sum(chi)
    return return_chi2


def AssignBin(x, p_cutoff_points, special_attribute=[]):
    '''
    :param x: the value of variable
    :param p_cutoff_points: the ChiMerge result for continous variable
    :param special_attribute:  the special attribute which should be assigned separately
    :return: bin number, indexing from 0
    for example, if cutOffPoints = [10,20,30], if x = 7, return Bin 0. If x = 35, return Bin 3
    '''
    numBin = len(p_cutoff_points) + 1 + len(special_attribute)
    if x in special_attribute:
        i = special_attribute.index(x)+1
        return 'Bin {}'.format(0-i)
    if x<=p_cutoff_points[0]:
        return 'Bin 0'
    elif x > p_cutoff_points[-1]:
        return 'Bin {}'.format(numBin-1)
    else:
        for i in range(0,numBin-1):
            if p_cutoff_points[i] < x <=  p_cutoff_points[i + 1]:
                return 'Bin {}'.format(i+1)

pd1 = pd.read_csv("data1.csv")

pd1.head(5)
pd_test = chi_merge_max_interval(pd1, "day", "chrun", 5)


cutOffPoints = chi_merge_max_interval(pd1, "day", "chrun", 5)
var_cutoff = {}
var_cutoff["day"] = cutOffPoints
pd1["day"] = pd1["day"].map(lambda x: AssignBin(x, cutOffPoints))

def chi_merge_max_interval(p_df, p_col, p_target, p_max_bin=5, special_attribute=[]):

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
        temp_df2["temp"] = temp_df2[p_col].map(lambda x: assign_group(x, split_x))
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
            chisq = chi2(df2b, 'total', 'bad', overall_rate)
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
    cut_off_points = special_attribute + cut_off_points
    return cut_off_points







