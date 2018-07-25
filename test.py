import random
import pandas as pd
import chinaPnr.utility.explore as u_explore
import chinaPnr.utility.modify as u_modify
import chinaPnr.utility.others as u_others
import chinaPnr.utility.sample as u_sample
import chinaPnr.utility.model as u_model
import chinaPnr.utility.assess as u_assess

deleted_var_list = []  # delete the categorical features in one of its single bin occupies more than 90%
encoded_var_list = {}
merged_var_list = {}
var_iv = {}  # save the IV values for binned features
var_woe = {}

# ID的字段名
col_id = 'Idx'
# 目标字段名
p_target = 'target'
# 排除字段名
drop_var = ['ListingInfo']
p_df = pd.read_csv('allData_1.csv', header=0, encoding='gbk')
# string_var_list, number_var_list, all_var_list = u_explore.get_list_for_number_str_col(p_df=p_df,
#                                                                                        p_col_id=col_id,
#                                                                                        p_p_target=p_target,
#                                                                                        p_drop_col=drop_var)
col = "LogInfo1_90_count"
'''
For continous variables, we do the following work:
1, split the variable by ChiMerge (by default into 5 bins)
2, check the bad rate, if it is not monotone, we decrease the number of bins until the bad rate is monotone
3, delete the variable if maximum bin occupies more than 90%
'''


var_cutoff = {}


def woe_iv_for_num(p_df, p_str_num_list, p_target, p_deleted_var_list, p_var_iv_list, p_var_woe_list):
    for col in p_str_num_list:
        print("{} is in processing".format(col))
        col1 = str(col) + '_Bin'
        # (1), 分割连续变量并保存分割点。-1是一种特殊情况，把它分成一组。
        # if -1 in set(p_df[col]):
        #     special_attribute = [-1]
        # else:
        #     special_attribute = []
        # special_attribute = []
        cut_off_points = u_modify.chi_merge_max_interval(p_df, col, p_target)
        var_cutoff[col] = cut_off_points
        p_df[col1] = p_df[col].map(lambda x: u_modify.assign_bin(x, cut_off_points))

        # (2), check whether the bad rate is monotone
        BRM = u_modify.bad_rate_monotone(p_df, col1, p_target)
        if not BRM:
            for bins in range(4,1,-1):
                cut_off_points = u_modify.chi_merge_max_interval(p_df, col, p_target, p_max_bin=bins)
                p_df[col1] = p_df[col].map(lambda x: u_modify.assign_bin(x, cut_off_points))
                BRM = u_modify.bad_rate_monotone(p_df, col1, p_target)
                if BRM:
                    break
            var_cutoff[col] = cut_off_points

        # (3), check whether any single bin occupies more than 90% of the total
        maxPcnt = u_modify.max_badrate_for_string_1(p_df, col1)
        if maxPcnt > 0.9:
            #del p_df[col1]
            p_deleted_var_list.append(col)
            print('we delete {} because the maximum bin occupies more than 90%'.format(col))
            continue
        woe_iv = u_modify.calc_woe_iv(p_df, col1, 'target')
        p_var_woe_list[col] = woe_iv['IV']
        p_var_iv_list[col] = woe_iv['WOE']
        del p_df[col]


# pd1 = pd.read_csv('data1.csv', header=0, encoding='gbk')
# pd1 = pd1.sort_values(["var1"])
# a= range(4,1,-1)
# print()