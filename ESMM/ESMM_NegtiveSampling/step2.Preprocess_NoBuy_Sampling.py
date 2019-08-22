#!/usr/bin/env python
# coding: utf-8
"""
数据预处理为FeatureColumn
本程序主要是根据feildid 分散记录的数据源聚合成 feildid为key对应的valuelist的格式，其中valuelist根据"|"分割
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter
import tensorflow as tf
import os
import pickle
import re
from tensorflow.python.ops import math_ops




"""
处理样本骨架的商品域特征
Item Features	
205	Item ID.
206	Category ID to which the item belongs to.
207	Shop ID to which item belongs to.
210	Intention node ID which the item belongs to.
216	Brand ID of the item.
### Combination Features	
508	The combination of features with 109_14 and 206.
509	The combination of features with 110_14 and 207.
702	The combination of features with 127_14 and 216.
853	The combination of features with 150_14 and 210.
### Context Features	
301	A categorical expression of position.
"""
"""
训练集样本骨架 按照fieldname聚合操作
 将feildid 换成 fieldidname
"""

sample_feature_columns = ['sample_id', 'click', 'buy', 'md5', 'feature_num', 'feature_list']
sample_table = pd.read_csv('./ctr_cvr_data/BuyWeight_sample_skeleton_train_sample_2_percent.csv',
                             sep=',', header=None, names=sample_feature_columns, engine='python')
feature_field_list = ['205','206','207','210','216','508','509','702','853','301']
feature_name_list = ['ItemID','CategoryID','ShopID','NodeID','BrandID','Com_CateID','Com_ShopID','Com_BrandID','Com_NodeID','PID']
field_id_name = {'205':'ItemID',
                 '206':'CategoryID',
                 '207':'ShopID',
                 '210':'NodeID',
                 '216':'BrandID',
                 '508':'Com_CateID',
                 '509':'Com_ShopID',
                 '702':'Com_BrandID',
                 '853':'Com_NodeID',
                 '301':'PID'}
entire_fea_dict = {}
for k,v in field_id_name.items():
    entire_fea_dict[v] = [] # 全局特征词典  {fea_field_name->[]}
"""
Features_list : feature_feild_id0x02feature_id0x03feature_value  0x01  feature_feild_id0x02feature_id0x03feature_value
"""
for index, row in sample_table.iterrows():
    fea_dict = {}
    for k,v in field_id_name.items():
        fea_dict[k] = [] # 局部特征词典  {fea_field_id->[]}
    feature_arr = row['feature_list'].split('\001')
    for fea_kv in feature_arr:
        fea_field_id = fea_kv.split('\002')[0]#feature_feild_id
        fea_id_val = fea_kv.split('\002')[1]#feature_id0x03feature_value
        fea_id = fea_id_val.split('\003')[0]#feature_id
        fea_val = fea_id_val.split('\003')[1]#feature_value
        print(fea_field_id,fea_id,fea_val)
        fea_dict[fea_field_id].append(fea_id)#fea_dict记录了所有featureFeildid:featureid列表
    print(fea_dict)
    for k,v in fea_dict.items():
        if len(v) == 0:
            entire_fea_dict[field_id_name[k]].append('<PAD>')
        else:
            entire_fea_dict[field_id_name[k]].append('|'.join(v))
    if index % 10000 == 0:
       print("current_index:",index)

print(entire_fea_dict)

entire_fea_table = pd.DataFrame(data=entire_fea_dict,columns=feature_name_list)

print(sample_table.columns)
print(entire_fea_table.columns)
sample_table = sample_table.drop('feature_list',axis=1)

sample_table = pd.concat([sample_table, entire_fea_table], axis=1, join_axes=[sample_table.index])

sample_table.to_csv('./ctr_cvr_data/BuyWeight_sampled_sample_skeleton_train_sample_feature_column.csv',index=False)
print(0)


"""
 测试集样本骨架样本
 将feildid 换成 fieldidname
"""

sample_feature_columns = ['sample_id', 'click', 'buy', 'md5', 'feature_num', 'feature_list']
sample_table = pd.read_table('./ctr_cvr_data/BuyWeight_sample_skeleton_test_sample_2_percent.csv',
                             sep=',', header=None, names=sample_feature_columns, engine = 'python')
#feature_field_list = ['205','206','207','210','216','508','509','702','853','301']
feature_name_list = ['ItemID','CategoryID','ShopID','NodeID','BrandID','Com_CateID',
                     'Com_ShopID','Com_BrandID','Com_NodeID','PID']
field_id_name = {'205':'ItemID',
                 '206':'CategoryID',
                 '207':'ShopID',
                 '210':'NodeID',
                 '216':'BrandID',
                 '508':'Com_CateID',
                 '509':'Com_ShopID',
                 '702':'Com_BrandID',
                 '853':'Com_NodeID',
                 '301':'PID'}
entire_fea_dict = {}
for k,v in field_id_name.items():
    entire_fea_dict[v] = []
for index, row in sample_table.iterrows():
    feature_arr = row['feature_list'].split('\001')
    fea_dict = {}
    for k,v in field_id_name.items():
        fea_dict[k] = []
    for fea_kv in feature_arr:
        fea_field_id = fea_kv.split('\002')[0]
        fea_id_val = fea_kv.split('\002')[1]
        fea_id = fea_id_val.split('\003')[0]
        fea_val = fea_id_val.split('\003')[1]
        #print(fea_field_id,fea_id,fea_val)
        fea_dict[fea_field_id].append(fea_id)
    #print(fea_dict)
    for k,v in fea_dict.items():
        if len(v) == 0:
            entire_fea_dict[field_id_name[k]].append('<PAD>')
        else:
            entire_fea_dict[field_id_name[k]].append('|'.join(v))
    if index % 10000 == 0:
       print("current_index:",index)

print(entire_fea_dict)

entire_fea_table = pd.DataFrame(data=entire_fea_dict,columns=feature_name_list)

print(sample_table.columns)
print(entire_fea_table.columns)
sample_table = sample_table.drop('feature_list',axis=1)


sample_table = pd.concat( [sample_table, entire_fea_table], axis=1, join_axes=[sample_table.index] )

sample_table.to_csv('./ctr_cvr_data/BuyWeight_sampled_sample_skeleton_test_sample_feature_column.csv',index=False)
print(0)


"""
处理Common 用户特征
User Features	
101	User ID.
109_14	User historical behaviors of category ID and count*.
110_14	User historical behaviors of shop ID and count*.
127_14	User historical behaviors of brand ID and count*.
150_14	User historical behaviors of intention node ID and count*.
121	Categorical ID of User Profile.
122	Categorical group ID of User Profile.
124	Users Gender ID.
125	Users Age ID.
126	Users Consumption Level Type I.
127	Users Consumption Level Type II.
128	Users Occupation: whether or not to work.
129	Users Geography Informations.
"""

# 训练集common feature

black_list = set(['109_14','110_14','127_14','150_14'])

common_table_columns = ['md5', 'feature_num', 'feature_list']
common_table = pd.read_table('./ctr_cvr_data/BuyWeight_common_features_skeleton_train_sample_2_percent.csv',
                                sep=',', header=None, names=common_table_columns, engine = 'python')
feature_name_list = ['UserID', 'User_CateIDs', 'User_ShopIDs', 'User_BrandIDs', 'User_NodeIDs', 'User_Cluster',
                     'User_ClusterID', 'User_Gender', 'User_Age', 'User_Level1', 'User_Level2',
                     'User_Occupation', 'User_Geo']
field_id_name = {'101':'UserID',
                 '109_14':'User_CateIDs',
                 '110_14':'User_ShopIDs',
                 '127_14':'User_BrandIDs',
                 '150_14':'User_NodeIDs',
                 '121':'User_Cluster',
                 '122':'User_ClusterID',
                 '124':'User_Gender',
                 '125':'User_Age',
                 '126':'User_Level1',
                 '127':'User_Level2',
                 '128':'User_Occupation',
                 '129':'User_Geo'}

#black_list = set(['109_14','110_14','127_14','150_14'])
black_list = set(['110_14','150_14'])
entire_fea_dict = {}
for k,v in field_id_name.items():
    entire_fea_dict[v] = []
for index, row in common_table.iterrows():
    feature_arr = row['feature_list'].split('\001')
    fea_dict = {}
    for k,v in field_id_name.items():
        fea_dict[k] = []
    for fea_kv in feature_arr:
        fea_field_id = fea_kv.split('\002')[0]
        fea_id_val = fea_kv.split('\002')[1]
        fea_id = fea_id_val.split('\003')[0]
        fea_val = fea_id_val.split('\003')[1]
        #print(fea_field_id,fea_id,fea_val)
        if fea_field_id in black_list:
            continue
        # Multi-Hot IDs类特征保留前100个ID
        if len(fea_dict[fea_field_id]) < 100:
            fea_dict[fea_field_id].append(fea_id)
    #print(fea_dict)
    for k,v in fea_dict.items():
        if len(v) == 0:
            entire_fea_dict[field_id_name[k]].append('<PAD>')
        else:
            entire_fea_dict[field_id_name[k]].append('|'.join(v))
    if index % 1000 == 0:
       print("current_index:",index)
#print(entire_fea_dict)

entire_fea_table = pd.DataFrame(data=entire_fea_dict, columns=feature_name_list)
print(entire_fea_table.shape)
print(common_table.shape)
common_table = common_table.drop('feature_list',axis=1)

common_table = pd.concat([common_table, entire_fea_table], axis=1, join_axes=[common_table.index])

common_table.to_csv('./ctr_cvr_data/BuyWeight_sampled_common_features_skeleton_train_sample_feature_column.csv',index=False)
print(common_table.shape)
common_table.head()


#  测试集common feature
common_table_columns = ['md5', 'feature_num', 'feature_list']
common_table = pd.read_csv('./ctr_cvr_data/BuyWeight_common_features_skeleton_test_sample_2_percent.csv',
                                sep=',', header=None, names=common_table_columns, engine = 'python')
feature_name_list = ['UserID', 'User_CateIDs', 'User_ShopIDs', 'User_BrandIDs', 'User_NodeIDs', 'User_Cluster',
                     'User_ClusterID', 'User_Gender', 'User_Age', 'User_Level1', 'User_Level2',
                     'User_Occupation', 'User_Geo']
field_id_name = {'101':'UserID',
                 '109_14':'User_CateIDs',
                 '110_14':'User_ShopIDs',
                 '127_14':'User_BrandIDs',
                 '150_14':'User_NodeIDs',
                 '121':'User_Cluster',
                 '122':'User_ClusterID',
                 '124':'User_Gender',
                 '125':'User_Age',
                 '126':'User_Level1',
                 '127':'User_Level2',
                 '128':'User_Occupation',
                 '129':'User_Geo'}
# 为了减少内存占用，方便单机版运行，先去掉Multi-hot特征
#black_list = set(['109_14','110_14','127_14','150_14'])
black_list = set(['110_14','150_14'])
entire_fea_dict = {}
for k,v in field_id_name.items():
    entire_fea_dict[v] = []
for index, row in common_table.iterrows():
    feature_arr = row['feature_list'].split('\001')
    fea_dict = {}
    for k,v in field_id_name.items():
        fea_dict[k] = []
    for fea_kv in feature_arr:
        fea_field_id = fea_kv.split('\002')[0]
        fea_id_val = fea_kv.split('\002')[1]
        fea_id = fea_id_val.split('\003')[0]
        fea_val = fea_id_val.split('\003')[1]
        #print(fea_field_id,fea_id,fea_val)
        if fea_field_id in black_list:
            continue
        # Multi-Hot IDs类特征保留前100个ID  else 就不再添加
        if len(fea_dict[fea_field_id]) < 100:
            fea_dict[fea_field_id].append(fea_id)
    #print(fea_dict)
    for k,v in fea_dict.items():  #k是featureFeildid   v是featureid
        if len(v) == 0:
            entire_fea_dict[field_id_name[k]].append('<PAD>')
        else:
            entire_fea_dict[field_id_name[k]].append('|'.join(v))
    if index % 1000 == 0:
       print("current_index:",index)
print(entire_fea_dict)

entire_fea_table = pd.DataFrame(data=entire_fea_dict, columns=feature_name_list)
print(entire_fea_table.shape)
common_table = common_table.drop('feature_list',axis=1)
common_table = pd.concat([common_table, entire_fea_table], axis=1, join_axes=[common_table.index])

common_table.to_csv('./ctr_cvr_data/BuyWeight_sampled_common_features_skeleton_test_sample_feature_column.csv',index=False)
print(0)
common_table.head()