#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter
import tensorflow as tf
import random
import os
import pickle
import re
from tensorflow.python.ops import math_ops



"""
原始数据文档
https://tianchi.aliyun.com/datalab/dataSet.html?dataId=408
购买样本全部保留
只对未购买的样本进行采样
"""
"""
对训练数据集合进行处理
"""
f = open("ctr_cvr_data/sample_train/sample_skeleton_train.csv",'r')
f_o = open("./ctr_cvr_data/BuyWeight_sample_skeleton_train_sample_2_percent.csv",'w')#采样后输出文件地址及其名称
train_md5_set = set()
index = 0
for line in f:
    tokens = line.strip().split(",")
    #tokens 按照","分割之后，各个值分别为
    #      0     1    2     3                            4        5
    # sample_id,click,buy,md5(common_feature_index),feature_num,feature_list

    # 有购买行为的样本全部都要，对于没有购买行为的进行2.5%采样
    if tokens[2] == '1' or random.uniform(0, 1) < 0.025:
        f_o.write(line)
        train_md5_set.add(tokens[3])
    index += 1
    if index % 1000000 == 0:
        print("current_index:",index)
f.close()
f_o.close()
pickle.dump(train_md5_set, open('./ctr_cvr_data/BuyWeight_train_md5_set.p', 'wb'))
print(0)



# """
# 对测试数据集合进行处理
# """
# f = open("ctr_cvr_data/sample_test/sample_skeleton_test.csv",'r')
# f_o = open("./ctr_cvr_data/BuyWeight_sample_skeleton_test_sample_2_percent.csv",'w')
# test_md5_set = set()
# index = 0
# for line in f:
#     tokens = line.strip().split(",")
#     if tokens[2] == '1' or random.uniform(0, 1) < 0.025:
#         f_o.write(line)
#         test_md5_set.add(tokens[3])
#     index += 1
#     if index % 1000000 == 0:
#         print("current_index:",index)
# f.close()
# f_o.close()
# pickle.dump(test_md5_set, open('./ctr_cvr_data/BuyWeight_test_md5_set.p', 'wb'))
# print(0)



# """
# 训练集公共特征处理，只保留需要的common_feature_index(md5)，这也是为什么在处理样本骨架的时候产出了一份 .p数据了
# """
# train_md5_set = pickle.load(open('./ctr_cvr_data/BuyWeight_train_md5_set.p', mode='rb'))
# print(len(train_md5_set))
# f = open("ctr_cvr_data/sample_train/common_features_train.csv",'r')
# f_o = open("./ctr_cvr_data/BuyWeight_common_features_skeleton_train_sample_2_percent.csv",'w')
#
# index = 0
# for line in f:
#     tokens = line.strip().split(",")
#     # tokens 的各个部分分别为
#     #              0                  1           2
#     # common_feature_index(md5),feature_num,feature_list
#     value=0
#     md5 = tokens[0]
#     if md5 in train_md5_set:
#         f_o.write(line)
#     index += 1
#     if index % 10000 == 0:
#         print("current_index:",index)
# f.close()
# f_o.close()
# print(0)
#
#
#
# """
# 测试集公共特征处理，只保留需要的common_feature_index(md5)，这也是为什么在处理样本骨架的时候产出了一份 .p数据了
# """
# test_md5_set = pickle.load(open('./ctr_cvr_data/BuyWeight_test_md5_set.p', mode='rb'))
# print(len(test_md5_set))
# f = open("ctr_cvr_data/sample_test/common_features_test.csv",'r')
# f_o = open("./ctr_cvr_data/BuyWeight_common_features_skeleton_test_sample_2_percent.csv",'w')
#
# index = 0
# for line in f:
#     tokens = line.strip().split(",")
#     value=0
#     md5 = tokens[0]
#     if md5 in test_md5_set:
#         f_o.write(line)
#     index += 1
#     if index % 10000 == 0:
#         print("current_index:",index)
# f.close()
# f_o.close()
#
# print(0)
#
# """
# 备注：样本格式分析
#
# """