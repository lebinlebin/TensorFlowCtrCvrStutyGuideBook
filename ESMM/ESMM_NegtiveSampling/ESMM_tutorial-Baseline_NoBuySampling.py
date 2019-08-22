#!/usr/bin/env python
# coding: utf-8
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
先来看看数据
经过Step1&2后，采样2.5%的数据样例，其中训练和测试各是100万条
BuyWeight_sampled_common_features_skeleton_test_sample_feature_column.csv
BuyWeight_sampled_common_features_skeleton_train_sample_feature_column.csv
BuyWeight_sampled_sample_skeleton_test_sample_feature_column.csv
BuyWeight_sampled_sample_skeleton_train_sample_feature_column.csv
"""



# 样本骨架 数据结构
# 训练数据集
sample_feature_columns = ['sample_id', 'click', 'buy', 'md5', 'feature_num', 'ItemID','CategoryID','ShopID','NodeID','BrandID','Com_CateID',
                     'Com_ShopID','Com_BrandID','Com_NodeID','PID']
train_sample_table = pd.read_csv('./ctr_cvr_data/BuyWeight_sampled_sample_skeleton_train_sample_feature_column.csv', sep=',',
                                   dtype={'ItemID': object, 'CategoryID': object, 'ShopID': object, 'PID': object},
                                   header=0, names=None, engine = 'python')
print("训练集样本骨架数据样式：   ",train_sample_table.head())
print("训练集样本骨架数据样式shape：   ",train_sample_table.shape)


sample_feature_columns = ['sample_id', 'click', 'buy', 'md5', 'feature_num', 'ItemID','CategoryID','ShopID','NodeID','BrandID','Com_CateID',
                     'Com_ShopID','Com_BrandID','Com_NodeID','PID']
test_sample_table = pd.read_csv('./ctr_cvr_data/BuyWeight_sampled_sample_skeleton_test_sample_feature_column.csv', sep=',',
                                  dtype={'ItemID': object, 'CategoryID': object, 'ShopID': object, 'PID': object},
                                  header=0, names=None, engine = 'python')
print("测试集样本骨架数据样式：   ",test_sample_table.head())
print("测试集样本骨架数据样式shape：   ",test_sample_table.shape)

#
# # Common Feature 数据结构
#
# common_feature_columns = ['md5', 'feature_num', 'UserID', 'User_CateIDs', 'User_ShopIDs', 'User_BrandIDs', 'User_NodeIDs', 'User_Cluster',
#                      'User_ClusterID', 'User_Gender', 'User_Age', 'User_Level1', 'User_Level2',
#                      'User_Occupation', 'User_Geo']
# train_common_features = pd.read_csv('./ctr_cvr_data/BuyWeight_sampled_common_features_skeleton_train_sample_feature_column.csv', sep=',',
#                                       header=0, names=None, engine = 'python')
# print("训练集样本公共数据样式：   ",train_common_features.head())
# print("训练集样本公共数据样式shape：   ",train_common_features.shape)
#
#
# common_feature_columns = ['md5', 'feature_num', 'UserID', 'User_CateIDs', 'User_ShopIDs', 'User_BrandIDs', 'User_NodeIDs', 'User_Cluster',
#                      'User_ClusterID', 'User_Gender', 'User_Age', 'User_Level1', 'User_Level2',
#                      'User_Occupation', 'User_Geo']
# test_common_features = pd.read_csv('./ctr_cvr_data/BuyWeight_sampled_common_features_skeleton_test_sample_feature_column.csv', sep=',',
#                                      header=0, names=None, engine = 'python')
# print("测试集样本公共数据样式：   ",test_common_features.head())
# print("测试集样本公共数据样式shape：   ",test_common_features.shape)
#
#
# # 两表join示例
# print("训练集骨架数据shape：   ",train_sample_table.shape)
# print("训练集公共部分数据shape：   ",train_common_features.shape)
#
#
# print("测试集样本骨架数据样式shape：   ",test_sample_table.shape)
# print("测试集样本公共数据样式shape：   ",test_common_features.shape)
#
# merge_data = pd.merge(train_sample_table, train_common_features, on='md5',how='inner')
#
# print("拼接之后的样本数据shape：    \n",merge_data.shape)
# print("拼接之后的样本的列columns：    \n",merge_data.columns)
# print("拼接之后的头部数据：  \n",merge_data.head())
#
#
# # 实现数据预处理
# # train_sample_table.head()
#
#
# # 打印Column和Types，确保Train和测试集可以一起序列化
# print(train_sample_table['ItemID'].head())
# print(test_sample_table.head()['ItemID'])
# print(train_common_features.head()['UserID'])
# print(test_sample_table.dtypes)
# print(train_sample_table.dtypes)
#
#
#
# print("训练集公共特征的列：    ",train_common_features.columns)
# print("train_common_features.head",train_common_features['feature_num'].head())
# #train_sample_table.columns
#
#
# # 打印Unique ID数
# value1 = set(train_common_features['UserID'].tolist())
# print(len(train_sample_table['ItemID'].unique()))
# print(len(train_sample_table['CategoryID'].unique()))
# print(len(train_sample_table['ShopID'].unique()))
# print(len(train_sample_table['NodeID'].unique()))
# print(len(train_sample_table['BrandID'].unique()))
# print(len(train_sample_table['Com_ShopID'].unique()))
# print(len(train_sample_table['Com_BrandID'].unique()))
# print(len(train_sample_table['Com_NodeID'].unique()))
# print(len(train_sample_table['PID'].unique()))
#
#
# #11176 640062 90 6111 258552 101090 4695 91412 43051 3
#
# value1 = set(train_sample_table['ShopID'].tolist())
# value2 = set(test_sample_table['ShopID'].tolist())
#
# value1 = set(train_common_features['UserID'].tolist())
# value2 = set(test_common_features['UserID'].tolist())
# print("value train",len(value1))
# print("value test",len(value2))
# print("inner product",len(value1&value2))
#
#
# def load_ESMM_Train_and_Test_Data():
#     """
#     Load Dataset from File
#     """
#     sample_feature_columns = ['sample_id', 'click', 'buy', 'md5', 'feature_num', 'ItemID','CategoryID','ShopID','NodeID','BrandID','Com_CateID',
#                      'Com_ShopID','Com_BrandID','Com_NodeID','PID']
#
#     common_feature_columns = ['md5', 'feature_num', 'UserID', 'User_CateIDs', 'User_ShopIDs', 'User_BrandIDs', 'User_NodeIDs', 'User_Cluster',
#                      'User_ClusterID', 'User_Gender', 'User_Age', 'User_Level1', 'User_Level2',
#                      'User_Occupation', 'User_Geo']
#
#     # 强制转化为其中部分列为object，是因为训练和测试某些列，Pandas load类型不一致，影响后面的序列化
#     train_sample_table = pd.read_csv('./ctr_cvr_data/BuyWeight_sampled_sample_skeleton_train_sample_feature_column.csv', sep=',',
#                                        dtype={'ItemID': object, 'CategoryID': object, 'ShopID': object, 'PID': object},
#                                        header=0, names=None, engine = 'python')
#     train_common_features = pd.read_csv('./ctr_cvr_data/BuyWeight_sampled_common_features_skeleton_train_sample_feature_column.csv',
#                                           sep=',', header=0, names=None, engine = 'python')
#
#     test_sample_table = pd.read_csv('./ctr_cvr_data/BuyWeight_sampled_sample_skeleton_test_sample_feature_column.csv', sep=',',
#                                       dtype={'ItemID': object, 'CategoryID': object, 'ShopID': object, 'PID': object},
#                                       header=0, names=None, engine = 'python')
#     test_common_features = pd.read_csv('./ctr_cvr_data/BuyWeight_sampled_common_features_skeleton_test_sample_feature_column.csv',
#                                          sep=',', header=0, names=None, engine = 'python')
#
#     # itemID转数字字典
#     ItemID_set = set()
#     for val in train_sample_table['ItemID'].str.split('|'):
#         ItemID_set.update(val)
#     for val in test_sample_table['ItemID'].str.split('|'):
#         ItemID_set.update(val)
#     ItemID_set.add('<PAD>')
#
#     ItemID2int = {val:ii for ii, val in enumerate(ItemID_set)}
#     #itemID 转成等长数字列表，示例，其实itemID是One Hot的，不需要此操作
#     ItemID_map = {val:[ItemID2int[row] for row in val.split('|')]
#                   for ii,val in enumerate(set(train_sample_table['ItemID']))}
#     test_ItemID_map = {val:[ItemID2int[row] for row in val.split('|')]
#                        for ii,val in enumerate(set(test_sample_table['ItemID']))}
#     # merge train & test
#     ItemID_map.update(test_ItemID_map)
#     ItemID_map_max_len = 1
#     print("ItemID_map max_len:", ItemID_map_max_len)
#     for key in ItemID_map:
#         for cnt in range(ItemID_map_max_len - len(ItemID_map[key])):
#             ItemID_map[key].insert(len(ItemID_map[key]) + cnt,ItemID2int['<PAD>'])
#     train_sample_table['ItemID'] = train_sample_table['ItemID'].map(ItemID_map)
#     test_sample_table['ItemID'] = test_sample_table['ItemID'].map(ItemID_map)
#     print("ItemID finish")
#
#
#     #User_CateIDs转数字字典
#     User_CateIDs_set = set()
#     for val in train_common_features['User_CateIDs'].str.split('|'):
#         User_CateIDs_set.update(val)
#     for val in test_common_features['User_CateIDs'].str.split('|'):
#         User_CateIDs_set.update(val)
#     User_CateIDs_set.add('<PAD>')
#     User_CateIDs2int = {val:ii for ii, val in enumerate(User_CateIDs_set)}
#     #User_CateIDs 转成等长数字列表
#     User_CateIDs_map = {val:[User_CateIDs2int[row] for row in val.split('|')]
#                         for ii,val in enumerate(set(train_common_features['User_CateIDs']))}
#     test_User_CateIDs_map = {val:[User_CateIDs2int[row] for row in val.split('|')]
#                              for ii,val in enumerate(set(test_common_features['User_CateIDs']))}
#     # merge train & test
#     User_CateIDs_map.update(test_User_CateIDs_map)
#     User_CateIDs_map_max_len = 100
#     print("User_CateIDs_map max_len:", User_CateIDs_map_max_len)
#     for key in User_CateIDs_map:
#         for cnt in range(User_CateIDs_map_max_len - len(User_CateIDs_map[key])):
#             User_CateIDs_map[key].insert(len(User_CateIDs_map[key]) + cnt,User_CateIDs2int['<PAD>'])
#     train_common_features['User_CateIDs'] = train_common_features['User_CateIDs'].map(User_CateIDs_map)
#     test_common_features['User_CateIDs'] = test_common_features['User_CateIDs'].map(User_CateIDs_map)
#     print("User_CateIDs finish")
#
#     #User_BrandIDs转数字字典
#     User_BrandIDs_set = set()
#     for val in train_common_features['User_BrandIDs'].str.split('|'):
#         User_BrandIDs_set.update(val)
#     for val in test_common_features['User_BrandIDs'].str.split('|'):
#         User_BrandIDs_set.update(val)
#     User_BrandIDs_set.add('<PAD>')
#     User_BrandIDs2int = {val:ii for ii, val in enumerate(User_BrandIDs_set)}
#     #User_BrandIDs 转成等长数字列表
#     User_BrandIDs_map = {val:[User_BrandIDs2int[row] for row in val.split('|')]
#                          for ii,val in enumerate(set(train_common_features['User_BrandIDs']))}
#     test_User_BrandIDs_map = {val:[User_BrandIDs2int[row] for row in val.split('|')]
#                               for ii,val in enumerate(set(test_common_features['User_BrandIDs']))}
#     # merge train & test
#     User_BrandIDs_map.update(test_User_BrandIDs_map)
#     User_BrandIDs_map_max_len = 100
#     print("User_BrandIDs_map max_len:", User_BrandIDs_map_max_len)
#     for key in User_BrandIDs_map:
#         for cnt in range(User_BrandIDs_map_max_len - len(User_BrandIDs_map[key])):
#             User_BrandIDs_map[key].insert(len(User_BrandIDs_map[key]) + cnt,User_BrandIDs2int['<PAD>'])
#     train_common_features['User_BrandIDs'] = train_common_features['User_BrandIDs'].map(User_BrandIDs_map)
#     test_common_features['User_BrandIDs'] = test_common_features['User_BrandIDs'].map(User_BrandIDs_map)
#     print("User_BrandIDs finish")
#
#
#     #userID 转数字字典
#     UserID_set = set()
#     for val in train_common_features['UserID']:
#         UserID_set.add(val)
#     for val in test_common_features['UserID']:
#         UserID_set.add(val)
#     UserID2int = {val: ii for ii, val in enumerate(UserID_set)}
#     UserID_map_max_len = 1
#     print("UserID_map max_len:", UserID_map_max_len)
#     train_common_features['UserID'] = train_common_features['UserID'].map(UserID2int)
#     test_common_features['UserID'] = test_common_features['UserID'].map(UserID2int)
#     print("UserID finish")
#
#     #User_Cluster 转数字字典
#     User_Cluster_set = set()
#     for val in train_common_features['User_Cluster']:
#         User_Cluster_set.add(val)
#     for val in test_common_features['User_Cluster']:
#         User_Cluster_set.add(val)
#     User_Cluster2int = {val: ii for ii, val in enumerate(User_Cluster_set)}
#     User_Cluster_map_max_len = 1
#     print("User_Cluster_map max_len:", User_Cluster_map_max_len)
#     train_common_features['User_Cluster'] = train_common_features['User_Cluster'].map(User_Cluster2int)
#     test_common_features['User_Cluster'] = test_common_features['User_Cluster'].map(User_Cluster2int)
#     print("User_Cluster finish")
#
#     #CategoryID 转数字字典
#     CategoryID_set = set()
#     for val in train_sample_table['CategoryID']:
#         CategoryID_set.add(val)
#     for val in test_sample_table['CategoryID']:
#         CategoryID_set.add(val)
#     CategoryID2int = {val:ii for ii, val in enumerate(CategoryID_set)}
#     CategoryID_map_max_len = 1
#     print("CategoryID_map max_len:", CategoryID_map_max_len)
#     train_sample_table['CategoryID'] = train_sample_table['CategoryID'].map(CategoryID2int)
#     test_sample_table['CategoryID'] = test_sample_table['CategoryID'].map(CategoryID2int)
#     print("CategoryID finish")
#
#     #ShopID 转数字字典
#     ShopID_set = set()
#     for val in train_sample_table['ShopID']:
#         ShopID_set.add(val)
#     for val in test_sample_table['ShopID']:
#         ShopID_set.add(val)
#     ShopID2int = {val:ii for ii, val in enumerate(ShopID_set)}
#     ShopID_map_max_len = 1
#     print("ShopID_map max_len:", ShopID_map_max_len)
#     train_sample_table['ShopID'] = train_sample_table['ShopID'].map(ShopID2int)
#     test_sample_table['ShopID'] = test_sample_table['ShopID'].map(ShopID2int)
#     print("ShopID finish")
#
#     #BrandID 转数字字典
#     BrandID_set = set()
#     for val in train_sample_table['BrandID']:
#         BrandID_set.add(val)
#     for val in test_sample_table['BrandID']:
#         BrandID_set.add(val)
#     BrandID2int = {val:ii for ii, val in enumerate(BrandID_set)}
#     BrandID_map_max_len = 1
#     print("BrandID_map max_len:", UserID_map_max_len)
#     train_sample_table['BrandID'] = train_sample_table['BrandID'].map(BrandID2int)
#     test_sample_table['BrandID'] = test_sample_table['BrandID'].map(BrandID2int)
#     print("BrandID finish")
#
#     #Com_CateID 转数字字典
#     Com_CateID_set = set()
#     for val in train_sample_table['Com_CateID']:
#         Com_CateID_set.add(val)
#     for val in test_sample_table['Com_CateID']:
#         Com_CateID_set.add(val)
#     Com_CateID2int = {val:ii for ii, val in enumerate(Com_CateID_set)}
#     Com_CateID_map_max_len = 1
#     print("Com_CateID_map max_len:", Com_CateID_map_max_len)
#     train_sample_table['Com_CateID'] = train_sample_table['Com_CateID'].map(Com_CateID2int)
#     test_sample_table['Com_CateID'] = test_sample_table['Com_CateID'].map(Com_CateID2int)
#     print("Com_CateID finish")
#
#     #Com_ShopID 转数字字典
#     Com_ShopID_set = set()
#     for val in train_sample_table['Com_ShopID']:
#         Com_ShopID_set.add(val)
#     for val in test_sample_table['Com_ShopID']:
#         Com_ShopID_set.add(val)
#     Com_ShopID2int = {val:ii for ii, val in enumerate(Com_ShopID_set)}
#     Com_ShopID_map_max_len = 1
#     print("Com_ShopID_map max_len:", Com_ShopID_map_max_len)
#     train_sample_table['Com_ShopID'] = train_sample_table['Com_ShopID'].map(Com_ShopID2int)
#     test_sample_table['Com_ShopID'] = test_sample_table['Com_ShopID'].map(Com_ShopID2int)
#     print("Com_ShopID finish")
#
#     #Com_BrandID 转数字字典
#     Com_BrandID_set = set()
#     for val in train_sample_table['Com_BrandID']:
#         Com_BrandID_set.add(val)
#     for val in test_sample_table['Com_BrandID']:
#         Com_BrandID_set.add(val)
#     Com_BrandID2int = {val:ii for ii, val in enumerate(Com_BrandID_set)}
#     Com_BrandID_map_max_len = 1
#     print("Com_BrandID_map max_len:", UserID_map_max_len)
#     train_sample_table['Com_BrandID'] = train_sample_table['Com_BrandID'].map(Com_BrandID2int)
#     test_sample_table['Com_BrandID'] = test_sample_table['Com_BrandID'].map(Com_BrandID2int)
#     print("Com_BrandID finish")
#
#     #PID 转数字字典
#     PID_set = set()
#     for val in train_sample_table['PID']:
#         PID_set.add(val)
#     for val in test_sample_table['PID']:
#         PID_set.add(val)
#     PID2int = {val:ii for ii, val in enumerate(PID_set)}
#     PID_map_max_len = 1
#     print("PID_map max_len:", PID_map_max_len)
#     train_sample_table['PID'] = train_sample_table['PID'].map(PID2int)
#     test_sample_table['PID'] = test_sample_table['PID'].map(PID2int)
#     print("PID finish")
#
#
#     #按照md5合并两个表
#     train_data = pd.merge(train_sample_table, train_common_features, on='md5',how='inner')
#     test_data = pd.merge(test_sample_table, test_common_features, on='md5',how='inner')
#
#     print("Sample/Common Merged")
#
#
#     # 将数据分成X和y两张表
#     feature_fields = ['UserID','ItemID','User_Cluster', 'CategoryID','ShopID','BrandID','Com_CateID','Com_ShopID',
#                       'Com_BrandID','PID','User_CateIDs','User_BrandIDs']
#     target_fields = ['click','buy']
#     train_features_pd, train_targets_pd = train_data[feature_fields], train_data[target_fields]
#     train_features = train_features_pd.values
#     train_targets_values = train_targets_pd.values
#
#     test_features_pd, test_targets_pd = test_data[feature_fields], test_data[target_fields]
#     test_features = test_features_pd.values
#     test_targets_values = test_targets_pd.values
#
#     return UserID_map_max_len, ItemID_map_max_len, User_Cluster_map_max_len, User_CateIDs_map_max_len, User_BrandIDs_map_max_len, \
#            CategoryID_map_max_len, ShopID_map_max_len, BrandID_map_max_len, Com_CateID_map_max_len,Com_ShopID_map_max_len, \
#            Com_BrandID_map_max_len, PID_map_max_len, UserID2int, ItemID2int,User_Cluster2int, User_CateIDs2int, User_BrandIDs2int,  \
#            CategoryID2int, ShopID2int, BrandID2int, Com_CateID2int, Com_ShopID2int, Com_BrandID2int, PID2int, train_features, \
#            train_targets_values, train_data, test_features, test_targets_values, test_data
#
#
#
# """
# 加载数据并保存到本地
# """
# UserID_map_max_len, ItemID_map_max_len, User_Cluster_map_max_len, User_CateIDs_map_max_len, User_BrandIDs_map_max_len, \
# CategoryID_map_max_len, ShopID_map_max_len, BrandID_map_max_len, Com_CateID_map_max_len,Com_ShopID_map_max_len, \
# Com_BrandID_map_max_len, PID_map_max_len, UserID2int, ItemID2int,User_Cluster2int, User_CateIDs2int, User_BrandIDs2int,  \
# CategoryID2int, ShopID2int, BrandID2int, Com_CateID2int, Com_ShopID2int, Com_BrandID2int, PID2int, train_features, \
# train_targets_values, train_data, test_features, test_targets_values, test_data = load_ESMM_Train_and_Test_Data()
# print(0)
#
# pickle.dump((UserID_map_max_len, ItemID_map_max_len, User_Cluster_map_max_len, User_CateIDs_map_max_len,
#              User_BrandIDs_map_max_len, CategoryID_map_max_len, ShopID_map_max_len, BrandID_map_max_len,
#              Com_CateID_map_max_len,Com_ShopID_map_max_len, Com_BrandID_map_max_len, PID_map_max_len, UserID2int,
#              ItemID2int,User_Cluster2int, User_CateIDs2int, User_BrandIDs2int,  CategoryID2int, ShopID2int,
#              BrandID2int, Com_CateID2int, Com_ShopID2int, Com_BrandID2int, PID2int, train_features,
#              train_targets_values, train_data, test_features, test_targets_values, test_data), open('./save/preprocess.p', 'wb'))
# print(0)
#
#
#
# print(User_CateIDs2int['<PAD>'])
# print(User_BrandIDs2int['<PAD>'])
#
#
# test_features[0:1,0:100]
# train_features[0:1,0:100]
# train_features.take(11,1)[1000]
#
#
#
# test_targets_values[0:10]
# print(train_targets_values.shape)
# print(test_targets_values.shape)
#
#
# user_cateids = np.zeros([2, 100])
# for i in range(2):
#     user_cateids[i] = train_features.take(10,1)[i]
# print(user_cateids)
#
#
# # ### 从本地读取数据
#
# UserID_map_max_len, ItemID_map_max_len, User_Cluster_map_max_len, User_CateIDs_map_max_len,\
# User_BrandIDs_map_max_len, CategoryID_map_max_len, ShopID_map_max_len, BrandID_map_max_len, \
# Com_CateID_map_max_len,Com_ShopID_map_max_len, Com_BrandID_map_max_len, PID_map_max_len, UserID2int,\
# ItemID2int,User_Cluster2int, User_CateIDs2int, User_BrandIDs2int,  CategoryID2int, ShopID2int, BrandID2int, \
# Com_CateID2int, Com_ShopID2int, Com_BrandID2int, PID2int, train_features, train_targets_values, train_data, \
# test_features, test_targets_values, test_data = pickle.load(open('./save/preprocess.p', mode='rb'))
# print(0)
#
# """
# 模型设计
# 模型架构
# Embedding Lookup 示例
# """
# import tensorflow as tf
# import numpy as np
#
# c = np.random.random([10,1])
# b = tf.nn.embedding_lookup(c, [1])
# a = tf.nn.embedding_lookup(c, 1)
# with tf.Session() as sess:
#     sess.run(tf.initialize_all_variables())
#     print(sess.run(b))
#     print(sess.run(a))
#     print(c)
#
#
# # ### Tensorflow slice 示例
#
# t = tf.constant([[1,100],[2,99],[3,98]])
# t1 = tf.slice(t, [0,1], [-1, 1])  # [[100, 99, 98]]
# t2 =  1-t1 # [[-99, -98, -97]]
# t3 = tf.concat([t1,t2],axis=1)
# print(t.shape)
# print(t1.shape)
# with tf.Session() as sess:
#     sess.run(tf.initialize_all_variables())
#     #print(sess.run(t1))
#     #print(sess.run(t2))
#     print(sess.run(t3))
#
#
#
# # ### 训练测试集Split 示例
#
# ## 计算AUC
# import numpy as np
# from sklearn.model_selection import train_test_split
# X, y = np.arange(10).reshape((5, 2)), range(5)
# #X
# #list(y)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# # X_train
# # y_train
# print(X)
# print(y)
# print(X_test)
# print(y_test)
#
#
# # ### AUC计算
#
# # 整个Batch AUC计算，适合CTR、CTCVR
# def calc_auc(raw_arr):
#
#     # sort by pred value, from small to big
#     arr = sorted(raw_arr, key=lambda d:d[2])
#
#     auc = 0.0
#     fp1, tp1, fp2, tp2 = 0.0, 0.0, 0.0, 0.0
#     for record in arr:
#         fp2 += record[0] # noclick
#         tp2 += record[1] # click
#         auc += (fp2 - fp1) * (tp2 + tp1)
#         fp1, tp1 = fp2, tp2
#
#     # if all nonclick or click, disgard
#     threshold = len(arr) - 1e-3
#     if tp2 > threshold or fp2 > threshold:
#         return -0.5
#
#     if tp2 * fp2 > 0.0:  # normal auc
#         return (1.0 - auc / (2.0 * tp2 * fp2))
#     else:
#         return None
#
# ### AUC 带Filter计算（CVR AUC只需要计算Click=1的样本子集）
# def calc_auc_with_filter(raw_arr, filter_arr):
#     ## get filter array row indexes
#     filter_index = np.nonzero(filter_arr)[0].tolist()
#     input_arr = [raw_arr[index] for index in filter_index]
#     auc_val = calc_auc(input_arr)
#     return auc_val
#
# """
# 构建神经网络
# 定义输入
# 定义输入的占位符
# """
#
#
# def get_inputs():
#     UserID = tf.placeholder(tf.int32, [None, 1], name="UserID")
#     ItemID = tf.placeholder(tf.int32, [None, 1], name="ItemID")
#     User_Cluster = tf.placeholder(tf.int32, [None, 1], name="User_Cluster")
#     User_CateIDs = tf.placeholder(tf.int32, [None, 100], name="User_CateIDs")
#     User_BrandIDs = tf.placeholder(tf.int32, [None, 100], name="User_BrandIDs")
#
#     CategoryID = tf.placeholder(tf.int32, [None, 1], name="CategoryID")
#     ShopID = tf.placeholder(tf.int32, [None, 1], name="ShopID")
#     BrandID = tf.placeholder(tf.int32, [None, 1], name="BrandID")
#     Com_CateID = tf.placeholder(tf.int32, [None, 1], name="Com_CateID")
#     Com_ShopID = tf.placeholder(tf.int32, [None, 1], name="Com_ShopID")
#     Com_BrandID = tf.placeholder(tf.int32, [None, 1], name="Com_BrandID")
#     PID = tf.placeholder(tf.int32, [None, 1], name="PID")
#
#     targets = tf.placeholder(tf.float32, [None, 2], name="targets")
#     LearningRate = tf.placeholder(tf.float32, name = "LearningRate")
#     return  UserID, ItemID, User_Cluster, CategoryID, ShopID, BrandID, Com_CateID,\
#             Com_ShopID, Com_BrandID, PID, User_CateIDs, User_BrandIDs, targets, LearningRate
#
#
# # 特征MaxID计算
# # 方便Embedding初始化
#
# #嵌入矩阵的维度
# embed_dim = 12
# #userID个数
# UserID_max = max(UserID2int.values()) + 1
# #itemID个数
# ItemID_max = max(ItemID2int.values()) + 1
# User_Cluster_max = max(User_Cluster2int.values()) + 1
# User_CateIDs_max = max(User_CateIDs2int.values()) + 1
# User_BrandIDs_max = max(User_BrandIDs2int.values()) + 1
#
# CategoryID_max = max(CategoryID2int.values()) + 1
# ShopID_max = max(ShopID2int.values()) + 1
# BrandID_max = max(BrandID2int.values()) + 1
# Com_CateID_max = max(Com_CateID2int.values()) + 1
# Com_ShopID_max = max(Com_ShopID2int.values()) + 1
# Com_BrandID_max = max(Com_BrandID2int.values()) + 1
# PID_max = max(PID2int.values()) + 1
#
# #变长特征pooling方式
# combiner = "sum"
#
# print(UserID_max, ItemID_max, User_Cluster_max, User_CateIDs_max, User_BrandIDs_max, CategoryID_max, ShopID_max, BrandID_max,
#       Com_CateID_max, Com_ShopID_max, Com_BrandID_max, PID_max)
#
#
# # ### 对所有输入做Embedding
# def define_embedding_layers(UserID, ItemID, User_Cluster, CategoryID, ShopID, BrandID, Com_CateID,
#                             Com_ShopID, Com_BrandID, PID, User_CateIDs, User_BrandIDs):
#
#     UserID_embed_matrix = tf.Variable(tf.random_normal([UserID_max, embed_dim], 0, 0.001))
#     UserID_embed_layer = tf.nn.embedding_lookup(UserID_embed_matrix, UserID)
#     if combiner == "sum":
#         UserID_embed_layer = tf.reduce_sum(UserID_embed_layer, axis=1, keep_dims=True)
#
#     ItemID_embed_matrix = tf.Variable(tf.random_uniform([ItemID_max, embed_dim], 0, 0.001))
#     ItemID_embed_layer = tf.nn.embedding_lookup(ItemID_embed_matrix, ItemID)
#     if combiner == "sum":
#         ItemID_embed_layer = tf.reduce_sum(ItemID_embed_layer, axis=1, keep_dims=True)
#
#     User_Cluster_embed_matrix = tf.Variable(tf.random_uniform([User_Cluster_max, embed_dim], 0, 0.001))
#     User_Cluster_embed_layer = tf.nn.embedding_lookup(User_Cluster_embed_matrix, User_Cluster)
#     if combiner == "sum":
#         User_Cluster_embed_layer = tf.reduce_sum(User_Cluster_embed_layer, axis=1, keep_dims=True)
#
#     User_CateIDs_embed_matrix = tf.Variable(tf.random_uniform([User_CateIDs_max, embed_dim], 0, 0.001))
#     User_CateIDs_embed_layer = tf.nn.embedding_lookup(User_CateIDs_embed_matrix, User_CateIDs)
#     if combiner == "sum":
#         User_CateIDs_embed_layer = tf.reduce_sum(User_CateIDs_embed_layer, axis=1, keep_dims=True)
#
#     User_BrandIDs_embed_matrix = tf.Variable(tf.random_uniform([User_BrandIDs_max, embed_dim], 0, 0.001))
#     User_BrandIDs_embed_layer = tf.nn.embedding_lookup(User_BrandIDs_embed_matrix, User_BrandIDs)
#     if combiner == "sum":
#         User_BrandIDs_embed_layer = tf.reduce_sum(User_BrandIDs_embed_layer, axis=1, keep_dims=True)
#
#     CategoryID_embed_matrix = tf.Variable(tf.random_uniform([CategoryID_max, embed_dim], 0, 0.001))
#     CategoryID_embed_layer = tf.nn.embedding_lookup(CategoryID_embed_matrix, CategoryID)
#     if combiner == "sum":
#         CategoryID_embed_layer = tf.reduce_sum(CategoryID_embed_layer, axis=1, keep_dims=True)
#
#     ShopID_embed_matrix = tf.Variable(tf.random_uniform([ShopID_max, embed_dim], 0, 0.001))
#     ShopID_embed_layer = tf.nn.embedding_lookup(ShopID_embed_matrix, ShopID)
#     if combiner == "sum":
#         ShopID_embed_layer = tf.reduce_sum(ShopID_embed_layer, axis=1, keep_dims=True)
#
#     BrandID_embed_matrix = tf.Variable(tf.random_uniform([BrandID_max, embed_dim], 0, 0.001))
#     BrandID_embed_layer = tf.nn.embedding_lookup(BrandID_embed_matrix, BrandID)
#     if combiner == "sum":
#         BrandID_embed_layer = tf.reduce_sum(BrandID_embed_layer, axis=1, keep_dims=True)
#
#     Com_CateID_embed_matrix = tf.Variable(tf.random_uniform([Com_CateID_max, embed_dim], 0, 0.001))
#     Com_CateID_embed_layer = tf.nn.embedding_lookup(Com_CateID_embed_matrix, Com_CateID)
#     if combiner == "sum":
#         Com_CateID_embed_layer = tf.reduce_sum(Com_CateID_embed_layer, axis=1, keep_dims=True)
#
#     Com_ShopID_embed_matrix = tf.Variable(tf.random_uniform([Com_ShopID_max, embed_dim], 0, 0.001))
#     Com_ShopID_embed_layer = tf.nn.embedding_lookup(Com_ShopID_embed_matrix, Com_ShopID)
#     if combiner == "sum":
#         Com_ShopID_embed_layer = tf.reduce_sum(Com_ShopID_embed_layer, axis=1, keep_dims=True)
#
#     Com_BrandID_embed_matrix = tf.Variable(tf.random_uniform([Com_BrandID_max, embed_dim], 0, 0.001))
#     Com_BrandID_embed_layer = tf.nn.embedding_lookup(Com_BrandID_embed_matrix, Com_BrandID)
#     if combiner == "sum":
#         Com_BrandID_embed_layer = tf.reduce_sum(Com_BrandID_embed_layer, axis=1, keep_dims=True)
#
#
#     PID_embed_matrix = tf.Variable(tf.random_uniform([PID_max, embed_dim], 0, 0.001))
#     PID_embed_layer = tf.nn.embedding_lookup(PID_embed_matrix, PID)
#     if combiner == "sum":
#         PID_embed_layer = tf.reduce_sum(PID_embed_layer, axis=1, keep_dims=True)
#
#     '''
#     esmm_embedding_layer = tf.concat([UserID_embed_layer, ItemID_embed_layer, User_Cluster_embed_layer,\
#                                      CategoryID_embed_layer, ShopID_embed_layer, BrandID_embed_layer,\
#                                      Com_CateID_embed_layer, Com_ShopID_embed_layer, Com_BrandID_embed_layer,\
#                                       PID_embed_layer,], 2)
#     esmm_embedding_layer = tf.reshape(esmm_embedding_layer, [-1, embed_dim * 10])
#     '''
#     '''
#     # 数据量较小，选择UID特征和其他一些低维度特征
#     esmm_embedding_layer = tf.concat([UserID_embed_layer,User_Cluster_embed_layer,\
#                                      CategoryID_embed_layer,\
#                                      Com_CateID_embed_layer,\
#                                       PID_embed_layer,], 2)
#     esmm_embedding_layer = tf.reshape(esmm_embedding_layer, [-1, embed_dim * 5])
#     '''
#     # 数据量较小，选择User Cluster and 其他一些较低低维度特征
#     esmm_embedding_layer = tf.concat([User_CateIDs_embed_layer,User_BrandIDs_embed_layer,ItemID_embed_layer,CategoryID_embed_layer,
#                                       Com_CateID_embed_layer,PID_embed_layer,], 2)
#     esmm_embedding_layer = tf.reshape(esmm_embedding_layer, [-1, embed_dim * 6])
#     return esmm_embedding_layer
#
#
#
#
# def define_ctr_layer(esmm_embedding_layer):
#     ctr_layer_1 = tf.layers.dense(esmm_embedding_layer, 200, activation=tf.nn.relu)
#     ctr_layer_2 = tf.layers.dense(ctr_layer_1, 80, activation=tf.nn.relu)
#     ctr_layer_3 = tf.layers.dense(ctr_layer_2, 2) # [nonclick, click]
#     ctr_prob = tf.nn.softmax(ctr_layer_3) + 0.00000001
#     return ctr_prob
#
#
#
#
# def define_cvr_layer(esmm_embedding_layer):
#     cvr_layer_1 = tf.layers.dense(esmm_embedding_layer, 200, activation=tf.nn.relu)
#     cvr_layer_2 = tf.layers.dense(cvr_layer_1, 80, activation=tf.nn.relu)
#     cvr_layer_3 = tf.layers.dense(cvr_layer_2, 2) # [nonbuy, buy]
#     cvr_prob = tf.nn.softmax(cvr_layer_3) + 0.00000001
#     return cvr_prob
#
#
#
# #  由于demo数据过小，购买过于稀疏，设计cvr和ctr倒数第二层以前完全共享
# def define_ctr_cvr_layer(esmm_embedding_layer):
#     layer_1 = tf.layers.dense(esmm_embedding_layer, 128 , activation=tf.nn.relu)
#     layer_2 = tf.layers.dense(layer_1, 16, activation=tf.nn.relu)
#     layer_3 = tf.layers.dense(layer_2, 2)
#     ctr_prob = tf.nn.softmax(layer_3) + 0.00000001
#     layer_4 = tf.layers.dense(layer_2, 2)
#     cvr_prob = tf.nn.softmax(layer_4) + 0.00000001
#     return ctr_prob, cvr_prob
#
#
# # ## 构建计算图
# tf.reset_default_graph()
# train_graph = tf.Graph()
# with train_graph.as_default():
#     #获取输入占位符
#     UserID, ItemID, User_Cluster, CategoryID, ShopID, BrandID, Com_CateID,Com_ShopID, Com_BrandID, PID, \
#     User_CateIDs, User_BrandIDs, targets, lr = get_inputs()
#
#     # Embedding Input Layer
#     esmm_embedding_layer = define_embedding_layers(UserID, ItemID, User_Cluster, CategoryID, ShopID, BrandID, Com_CateID,
#                                                    Com_ShopID, Com_BrandID, PID, User_CateIDs, User_BrandIDs)
#
#     # CTR Network
#     #ctr_prob = define_ctr_layer(esmm_embedding_layer)
#
#     # CVR Network
#     #cvr_prob = define_cvr_layer(esmm_embedding_layer)
#
#     # 由于demo数据过小，购买过于稀疏，设计cvr和ctr倒数第二层以前完全共享
#     ctr_prob, cvr_prob = define_ctr_cvr_layer(esmm_embedding_layer)
#
#     with tf.name_scope("loss"):
#
#         ctr_prob_one = tf.slice(ctr_prob, [0,1], [-1, 1]) # [batch_size , 1]
#         cvr_prob_one = tf.slice(cvr_prob, [0,1], [-1, 1]) # [batchsize, 1 ]
#
#         ctcvr_prob_one = ctr_prob_one * cvr_prob_one # [ctr*cvr]
#         ctcvr_prob = tf.concat([1 - ctcvr_prob_one, ctcvr_prob_one], axis=1)
#
#         ctr_label =  tf.slice(targets, [0,0], [-1, 1]) # target: [click, buy]
#         ctr_label = tf.concat([1 - ctr_label, ctr_label], axis=1) # [1-click, click]
#
#         cvr_label = tf.slice(targets, [0,1], [-1, 1])
#         ctcvr_label = tf.concat([1 - cvr_label, cvr_label], axis=1)
#
#         # 单列，判断Click是否=1
#         ctr_clk = tf.slice(targets, [0,0], [-1, 1])
#         ctr_clk_dup = tf.concat([ctr_clk, ctr_clk], axis=1)
#
#         # clicked subset CVR loss
#         cvr_loss = - tf.multiply(tf.log(cvr_prob) * ctcvr_label, ctr_clk_dup)
#         # batch CTR loss
#         ctr_loss = - tf.log(ctr_prob) * ctr_label # -y*log(p)-(1-y)*log(1-p)
#         # batch CTCVR loss
#         ctcvr_loss = - tf.log(ctcvr_prob) * ctcvr_label
#
#         #loss = tf.reduce_mean(ctr_loss + ctcvr_loss + cvr_loss)
#         #loss = tf.reduce_mean(ctr_loss + ctcvr_loss)
#         #loss = tf.reduce_mean(ctr_loss + cvr_loss)
#         loss = tf.reduce_mean(cvr_loss)
#         ctr_loss = tf.reduce_mean(ctr_loss)
#         cvr_loss = tf.reduce_mean(cvr_loss)
#         ctcvr_loss = tf.reduce_mean(ctcvr_loss)
#
#     # 优化损失
#     #train_op = tf.train.AdamOptimizer(lr).minimize(loss)  #cost
#     global_step = tf.Variable(0, name="global_step", trainable=False)
#     optimizer = tf.train.AdamOptimizer(lr)
#     gradients = optimizer.compute_gradients(loss)  #cost
#     train_op = optimizer.apply_gradients(gradients, global_step=global_step)
#
#
#
# # ## 训练网络
# # ### 取得batch
#
# def get_batches(Xs, ys, batch_size):
#     for start in range(0, len(Xs), batch_size):
#         end = min(start + batch_size, len(Xs))
#         yield Xs[start:end], ys[start:end]
#
#
# # ### 超参
# # Number of Epochs
# num_epochs = 1
# # Batch Size
# batch_size = 10000
#
# # Test Batch Size
# test_batch_size = 10000
#
# # Learning Rate
# learning_rate = 0.01
# # Show stats for every n number of batches
# show_every_n_batches = 10
# show_test_every_n_batches = 10
#
# save_dir = './save'
#
#
# # ### 训练开始
# # get_ipython().run_line_magic('matplotlib', 'inline')
# # get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
# import matplotlib.pyplot as plt
# import time
# import datetime
#
# losses = {'train':[], 'test':[]}
#
# ctr_auc_stat = {'train':[], 'test':[]}
# cvr_auc_stat = {'train':[], 'test':[]}
# ctcvr_auc_stat = {'train':[], 'test':[]}
#
#
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
# #     sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options),graph=train_graph) as sess:
#     #搜集数据给tensorBoard用
#     # Keep track of gradient values and sparsity
#     grad_summaries = []
#     for g, v in gradients:
#         if g is not None:
#             grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name.replace(':', '_')), g)
#             sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name.replace(':', '_')), tf.nn.zero_fraction(g))
#             grad_summaries.append(grad_hist_summary)
#             grad_summaries.append(sparsity_summary)
#     grad_summaries_merged = tf.summary.merge(grad_summaries)
#
#     # Output directory for models and summaries
#     timestamp = str(int(time.time()))
#     out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
#     print("Writing to {}\n".format(out_dir))
#
#     # Summaries for loss and accuracy
#     loss_summary = tf.summary.scalar("loss", loss)
#
#     # Train Summaries
#     train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
#     train_summary_dir = os.path.join(out_dir, "summaries", "train")
#     train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
#
#     # Inference summaries
#     inference_summary_op = tf.summary.merge([loss_summary])
#     inference_summary_dir = os.path.join(out_dir, "summaries", "inference")
#     inference_summary_writer = tf.summary.FileWriter(inference_summary_dir, sess.graph)
#
#     sess.run(tf.global_variables_initializer())
#     saver = tf.train.Saver()
#
#     #将数据集分成训练集和测试集，随机种子固定，用训练集拆分出来训练和测试，当天随机切分
#     #train_X,test_X, train_y, test_y = train_test_split(features,
#     #                                                    targets_values,
#     #                                                   test_size = 0.2,
#     #                                                   random_state = 0)
#
#     # 训练集和测试集用两天的数据，前一天训练，后一天测试
#     train_X, train_y = train_features, train_targets_values
#     test_X, test_y = test_features, test_targets_values
#
#
#     for epoch_i in range(num_epochs):
#
#         train_ctr_auc_arr = []
#         train_cvr_auc_arr = []
#         train_ctcvr_auc_arr = []
#
#         test_ctr_auc_arr = []
#         test_cvr_auc_arr = []
#         test_ctcvr_auc_arr = []
#
#         train_batches = get_batches(train_X, train_y, batch_size)
#         test_batches = get_batches(test_X, test_y, test_batch_size)
#
#
#         #训练的迭代，保存训练损失
#         for batch_i in range(len(train_X) // batch_size):
#             x, y = next(train_batches)
#
#             #categories = np.zeros([batch_size, 18])
#             #for i in range(batch_size):
#                 #categories[i] = x.take(6,1)[i]
#             item_id = np.zeros([batch_size, 1])
#             for i in range(batch_size):
#                 item_id[i] = x.take(1,1)[i]
#
#             #User_CateIDs, User_BrandIDs
#             user_cateids = np.zeros([batch_size, 100])
#             for i in range(batch_size):
#                 user_cateids[i] = x.take(10,1)[i]
#             user_brandids = np.zeros([batch_size, 100])
#             for i in range(batch_size):
#                 user_brandids[i] = x.take(11,1)[i]
#
#             feed = {
#                 UserID : np.reshape(x.take(0,1), [batch_size, 1]),
#                 ItemID: item_id,
#                 User_Cluster : np.reshape(x.take(2,1), [batch_size, 1]),
#                 CategoryID : np.reshape(x.take(3,1), [batch_size, 1]),
#                 ShopID : np.reshape(x.take(4,1), [batch_size, 1]),
#                 BrandID : np.reshape(x.take(5,1), [batch_size, 1]),
#                 Com_CateID : np.reshape(x.take(6,1), [batch_size, 1]),
#                 Com_ShopID : np.reshape(x.take(7,1), [batch_size, 1]),
#                 Com_BrandID : np.reshape(x.take(8,1), [batch_size, 1]),
#                 PID : np.reshape(x.take(9,1), [batch_size, 1]),
#                 User_CateIDs: user_cateids,
#                 User_BrandIDs: user_brandids,
#                 #movie_categories: categories,  #x.take(6,1)
#                 targets: y,
#                 #np.reshape(y, [batch_size, 2]),
#                 lr: learning_rate}
#
#             step, train_loss, train_ctr_loss, train_cvr_loss, train_ctcvr_loss,train_ctr_prob, train_cvr_prob, train_ctcvr_prob,\
#             train_ctr_label, train_cvr_label, train_ctcvr_label, train_ctr_click,summaries, _ = \
#                 sess.run([global_step, loss, ctr_loss, cvr_loss, ctcvr_loss, ctr_prob, cvr_prob, ctcvr_prob,
#                                     ctr_label, ctcvr_label, ctcvr_label, ctr_clk, train_summary_op, train_op], feed)  #cost
#             losses['train'].append(train_loss)
#             train_summary_writer.add_summary(summaries, step)  #
#
#
#             print("train batch click num:", len(np.nonzero(y[:,0:1])[0]),
#                     " buy num:", len(np.nonzero(y[:,1:2])[0]))
#
#             ctr_input_arr = np.concatenate((train_ctr_label, train_ctr_prob[:, 1:2]), axis=1)
#             train_ctr_auc = calc_auc(ctr_input_arr)
#             if train_ctr_auc > 0:
#                 train_ctr_auc_arr.append(train_ctr_auc)
#
#             cvr_input_arr = np.concatenate((train_cvr_label, train_cvr_prob[:, 1:2]), axis=1)
#             train_cvr_auc = calc_auc_with_filter(cvr_input_arr, train_ctr_click)
#             if train_cvr_auc > 0:
#                 train_cvr_auc_arr.append(train_cvr_auc)
#
#             ctcvr_input_arr = np.concatenate((train_ctcvr_label, train_ctcvr_prob[:, 1:2]), axis=1)
#             train_ctcvr_auc = calc_auc(ctcvr_input_arr)
#             if train_ctcvr_auc > 0:
#                 train_ctcvr_auc_arr.append(train_ctcvr_auc)
#
#
#             # Show every <show_every_n_batches> batches
#             if batch_i > 0 and (epoch_i * (len(train_X) // batch_size) + batch_i) % show_every_n_batches == 0:
#                 # 累积 show_every_n_batches 个batch的Train AUC
#                 print (len(train_ctr_auc_arr),len(train_cvr_auc_arr) , len(train_ctcvr_auc_arr))
#                 train_ctr_auc = train_ctr_auc if len(train_ctr_auc_arr) == 0  else sum(train_ctr_auc_arr) / float(len(train_ctr_auc_arr))
#                 train_cvr_auc = train_cvr_auc if len(train_cvr_auc_arr) == 0  else sum(train_cvr_auc_arr) / float(len(train_cvr_auc_arr))
#                 train_ctcvr_auc = train_ctcvr_auc if len(train_ctcvr_auc_arr) == 0  else sum(train_ctcvr_auc_arr) / float(len(train_ctcvr_auc_arr))
#                 # 保存 AUC
#                 ctr_auc_stat['train'].append(train_ctr_auc)
#                 cvr_auc_stat['train'].append(train_cvr_auc)
#                 ctcvr_auc_stat['train'].append(train_ctcvr_auc)
#                 # 清空，并继续累积
#                 train_ctr_auc_arr.clear()
#                 train_cvr_auc_arr.clear()
#                 train_ctcvr_auc_arr.clear()
#
#                 time_str = datetime.datetime.now().isoformat()
#                 print('{}: Epoch {} Batch {}/{}  train_loss={:.3f} train_ctr_loss={:.3f} '
#                       'train_cvr_loss={:.3f} train_ctcvr_loss={:.3f} train_ctr_auc={:.3f} train_cvr_auc={:.3f} '
#                       'train_ctcvr_auc={:.3f}'.format(
#                     time_str,
#                     epoch_i,
#                     batch_i,
#                     (len(train_X) // batch_size),
#                     train_loss,
#                     train_ctr_loss,
#                     train_cvr_loss,
#                     train_ctcvr_loss,
#                     train_ctr_auc,
#                     train_cvr_auc,
#                     train_ctcvr_auc))
#
#         #使用测试数据的迭代
#         for batch_i  in range(len(test_X) // test_batch_size):
#             x, y = next(test_batches)
#
#             #user_id = np.zeros([test_batch_size, 1])
#             item_id = np.zeros([test_batch_size, 1])
#             for i in range(test_batch_size):
#                 #user_id[i] = x.take(0,1)[i]
#                 item_id[i] = x.take(1,1)[i]
#             #User_CateIDs, User_BrandIDs
#             user_cateids = np.zeros([test_batch_size, 100])
#             for i in range(batch_size):
#                 user_cateids[i] = x.take(10,1)[i]
#             user_brandids = np.zeros([test_batch_size, 100])
#             for i in range(batch_size):
#                 user_brandids[i] = x.take(11,1)[i]
#             feed = {
#                 UserID : np.reshape(x.take(0,1), [test_batch_size, 1]),
#                 ItemID: item_id,
#                 User_Cluster : np.reshape(x.take(2,1), [test_batch_size, 1]),
#                 CategoryID : np.reshape(x.take(3,1), [test_batch_size, 1]),
#                 ShopID : np.reshape(x.take(4,1), [test_batch_size, 1]),
#                 BrandID : np.reshape(x.take(5,1), [test_batch_size, 1]),
#                 Com_CateID : np.reshape(x.take(6,1), [test_batch_size, 1]),
#                 Com_ShopID : np.reshape(x.take(7,1), [test_batch_size, 1]),
#                 Com_BrandID : np.reshape(x.take(8,1), [test_batch_size, 1]),
#                 PID : np.reshape(x.take(9,1), [test_batch_size, 1]),
#                 User_CateIDs: user_cateids,
#                 User_BrandIDs: user_brandids,
#                 targets: np.reshape(y, [test_batch_size, 2]),
#                 lr: learning_rate}
#
#             step, test_loss, test_ctr_loss, test_cvr_loss, test_ctcvr_loss,test_ctr_prob, test_cvr_prob, test_ctcvr_prob,\
#             test_ctr_label, test_cvr_label, test_ctcvr_label, test_ctr_click,summaries \
#                 = sess.run([global_step, loss, ctr_loss, cvr_loss, ctcvr_loss,
#                                   ctr_prob, cvr_prob, ctcvr_prob,ctr_label, ctcvr_label,
#                                   ctcvr_label, ctr_clk,inference_summary_op], feed)  #cost
#
#             #保存测试损失
#             losses['test'].append(test_loss)
#             inference_summary_writer.add_summary(summaries, step)  #
#             print("test batch click num:", len(np.nonzero(y[:,0:1])[0]),
#                     " buy num:", len(np.nonzero(y[:,1:2])[0]))
#
#             ctr_input_arr = np.concatenate((test_ctr_label, test_ctr_prob[:, 1:2]), axis=1)
#             test_ctr_auc = calc_auc(ctr_input_arr)
#             if test_ctr_auc > 0:
#                 test_ctr_auc_arr.append(test_ctr_auc)
#
#             cvr_input_arr = np.concatenate((test_cvr_label, test_cvr_prob[:, 1:2]), axis=1)
#             test_cvr_auc = calc_auc_with_filter(cvr_input_arr, test_ctr_click)
#             if test_cvr_auc > 0:
#                 test_cvr_auc_arr.append(test_cvr_auc)
#
#             ctcvr_input_arr = np.concatenate((test_ctcvr_label, test_ctcvr_prob[:, 1:2]), axis=1)
#             test_ctcvr_auc = calc_auc(ctcvr_input_arr)
#             if test_ctcvr_auc > 0:
#                 test_ctcvr_auc_arr.append(test_ctcvr_auc)
#
#             time_str = datetime.datetime.now().isoformat()
#             if batch_i > 0 and (epoch_i * (len(test_X) // test_batch_size) + batch_i) % show_test_every_n_batches == 0:
#
#                 # 累积 show_every_n_batches 个batch的Train AUC
#                 print (len(test_ctr_auc_arr),len(test_cvr_auc_arr) , len(test_ctcvr_auc_arr))
#                 test_ctr_auc = test_ctr_auc if len(test_ctr_auc_arr) == 0  else sum(test_ctr_auc_arr) / float(len(test_ctr_auc_arr))
#                 test_cvr_auc = test_cvr_auc if len(test_cvr_auc_arr) == 0  else sum(test_cvr_auc_arr) / float(len(test_cvr_auc_arr))
#                 test_ctcvr_auc = test_ctcvr_auc if len(test_ctcvr_auc_arr) == 0  else sum(test_ctcvr_auc_arr) / float(len(test_ctcvr_auc_arr))
#                 # 保存 AUC
#                 ctr_auc_stat['test'].append(test_ctr_auc)
#                 cvr_auc_stat['test'].append(test_cvr_auc)
#                 ctcvr_auc_stat['test'].append(test_ctcvr_auc)
#                 # 清空，并继续累积
#                 test_ctr_auc_arr.clear()
#                 test_cvr_auc_arr.clear()
#                 test_ctcvr_auc_arr.clear()
#
#                 print('{}: Epoch {} Batch {}/{}  test_loss = {:.3f} test_ctr_loss = {:.3f} test_cvr_loss = {:.3f} '
#                       'test_ctcvr_loss = {:.3f}  test_ctr_auc = {:.3f} test_cvr_auc = {:.3f} test_ctcvr_auc = {:.3f}'.format(
#                     time_str,
#                     epoch_i,
#                     batch_i,
#                     (len(test_X) // test_batch_size),
#                     test_loss,
#                     test_ctr_loss,
#                     test_cvr_loss,
#                     test_ctcvr_loss,
#                     test_ctr_auc,
#                     test_cvr_auc,
#                     test_ctcvr_auc))
#
#     # Save Model
#     saver.save(sess, save_dir)  #, global_step=epoch_i
#     print('Model Trained and Saved')
#
#
# # ## 在 TensorBoard 中查看可视化结果
# # tensorboard --logdir /PATH_TO_CODE/runs/1543772895/summaries/
# # ## 辅助函数
#
# import tensorflow as tf
# import os
# import pickle
#
# def save_params(params):
#     """
#     Save parameters to file
#     """
#     pickle.dump(params, open('./save/params.p', 'wb'))
#
#
# def load_params():
#     """
#     Load parameters from file
#     """
#     return pickle.load(open('./save/params.p', mode='rb'))
#
#
# # ## 保存参数
# # 保存`save_dir` 在生成预测时使用。
#
# save_params((save_dir))
#
# load_dir = load_params()
#
#
# # ## 显示训练Loss
#
# plt.plot(losses['train'], label='Training loss')
# plt.legend()
# _ = plt.ylim()
#
#
# # ## 显示测试Loss
# # 迭代次数再增加一些，后面出现严重过拟合的情况
#
# plt.plot(losses['test'], label='Test loss')
# plt.legend()
# _ = plt.ylim()
#
#
# # ## 显示训练CTR AUC
#
# plt.plot(ctr_auc_stat['train'], label='Training AUC')
# plt.legend()
# _ = plt.ylim()
# print(ctr_auc_stat['train'])
#
#
# # ## 显示测试 CTR AUC
#
# plt.plot(ctr_auc_stat['test'], label='Test AUC')
# plt.legend()
# _ = plt.ylim()
# print(ctr_auc_stat['test'])
#
#
# # ## 显示训练CVR AUC
#
# plt.plot(cvr_auc_stat['train'], label='Training AUC')
# plt.legend()
# _ = plt.ylim()
# print(cvr_auc_stat['train'])
#
#
# # ## 显示测试 CVR AUC
#
# plt.plot(cvr_auc_stat['test'], label='Test AUC')
# plt.legend()
# _ = plt.ylim()
# print(cvr_auc_stat['test'])
#
#
# # ## 显示训练CTCVR AUC
#
# plt.plot(ctcvr_auc_stat['train'], label='Training AUC')
# plt.legend()
# _ = plt.ylim()
# print(ctcvr_auc_stat['train'])
#
#
# # ## 显示测试 CTCVR AUC
#
# plt.plot(ctcvr_auc_stat['test'], label='Test AUC')
# plt.legend()
# _ = plt.ylim()
# print(ctcvr_auc_stat['test'])
#
#
# """
# tensorboard	--logdir = runs
# """
#
# """
# 总结
# ESMM模型利用用户行为序列数据在完整样本空间建模，避免了传统CVR模型经常遭遇的样本选择偏差和训练数据稀疏的问题，
# 取得了显著的效果。另一方面，ESMM模型首次提出了利用学习CTR和CTCVR的辅助任务迂回学习CVR的思路。
# ESMM模型中的BASE子网络可以替换为任意的学习模型，因此ESMM的框架可以非常容易地和其他学习模型集成，
# 从而吸收其他学习模型的优势，进一步提升学习效果，想象空间巨大。
# 引用
# [1] Xiao Ma, Liqin Zhao, Guan Huang, Zhi Wang, Zelin Hu, Xiaoqiang Zhu, and Kun Gai. 2018.
# Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate. SIGIR (2018).
# """
