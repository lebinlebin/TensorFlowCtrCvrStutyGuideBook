#!/usr/bin/env python
# coding: utf-8
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter
import tensorflow as tf
import os
import pickle
import matplotlib.pyplot as plt
import time
import datetime
import numpy as np
import re
from tensorflow.python.ops import math_ops
import os
import pickle

"""
本文件主要用于测试ESMM模型中用到的关键函数
"""
def load_ESMM_Train_and_Test_Data():
    """
    Load Dataset from File
    """
    sample_feature_columns = ['sample_id', 'click', 'buy', 'md5', 'feature_num', 'ItemID','CategoryID','ShopID','NodeID','BrandID','Com_CateID',
                     'Com_ShopID','Com_BrandID','Com_NodeID','PID']
    
    common_feature_columns = ['md5', 'feature_num', 'UserID', 'User_CateIDs', 'User_ShopIDs', 'User_BrandIDs', 'User_NodeIDs', 'User_Cluster', 
                     'User_ClusterID', 'User_Gender', 'User_Age', 'User_Level1', 'User_Level2', 
                     'User_Occupation', 'User_Geo']
    
    # 强制转化为其中部分列为object，是因为训练和测试某些列，Pandas load类型不一致，影响后面的序列化
    train_sample_table = pd.read_csv('../ctr_cvr_data/BuyWeight_sampled_sample_skeleton_train_sample_feature_column.csv', sep=',',
                                       dtype={'ItemID': object, 'CategoryID': object, 'ShopID': object, 'PID': object},
                                       header=0, names=None, engine = 'python')
    train_common_features = pd.read_csv('../ctr_cvr_data/BuyWeight_sampled_common_features_skeleton_train_sample_feature_column.csv',
                                          sep=',', header=0, names=None, engine = 'python')
    
    test_sample_table = pd.read_csv('../ctr_cvr_data/BuyWeight_sampled_sample_skeleton_test_sample_feature_column.csv', sep=',',
                                      dtype={'ItemID': object, 'CategoryID': object, 'ShopID': object, 'PID': object},
                                      header=0, names=None, engine = 'python')
    test_common_features = pd.read_csv('../ctr_cvr_data/BuyWeight_sampled_common_features_skeleton_test_sample_feature_column.csv',
                                         sep=',', header=0, names=None, engine = 'python')
    
    """
    itemID转数字字典
    第一步：构建特征取值种类set。 将训练集和测试的所有ItemID列distinct之后建立一个set集合，包含ItemID所有可能出现的情形包括为空情况
    第二步：构建映射MAP。 根据第一步的set集合，处理成一个MAP, 其中key为 ItemID原始值，value为 一个从0开始一直到ItemID个数的整型数字。
    {itemID值:下标int_id} 的词典。
    """
    ItemID_set = set()
    for val in train_sample_table['ItemID'].str.split('|'):
        ItemID_set.update(val)
    for val in test_sample_table['ItemID'].str.split('|'):
        ItemID_set.update(val)
    ItemID_set.add('<PAD>')
    #生成一个词典，这个词典的key是ItemID_set的下标index，值为ItemID
    #{itemID值:下标int_id} 的词典
    ItemID2int = { val:ii for ii, val in enumerate(ItemID_set) }

    #{值 -> 编码 } 的形式
    #itemID 转成等长数字列表，示例，其实itemID是One Hot的，不需要此操作
    # 将itemID映射为指定数值
    #ItemID_map格式为  val -> [ItemID2int[row] for row in val.split('|')]
    ItemID_map = { val:[ItemID2int[row] for row in val.split('|')]
                  for ii,val in enumerate(set(train_sample_table['ItemID']))}
    #            字典推导式 以“:”分割的前后两部分分别为 词典的key:value 返回一个词典
    #    [ItemID2int[row] for row in val.split('|')] 是列表推导式 返回一个列表
    # 这里返回的是 如： {12345:[1234]}  ":"的前一部分表示train_sample_table['ItemID']对应的值，这里只有一个元素,帖子id,":"后边为一个列表，其中存储的是":"前边的ItemID值对应的编码值。编码的映射逻辑在Map(ItemID2int)中
    #对于多个值的情况下：返回的如： {12345|455151:[1234,1235]}
    test_ItemID_map = { val:[ItemID2int[row] for row in val.split('|')]  for ii,val in enumerate(set(test_sample_table['ItemID']))}
    # merge train & test
    ItemID_map.update(test_ItemID_map)

    """
    embedding空间补足。
    即：例如，对于我们想将ItemID映射成为一个二维的向量，那么这个特征就有两个空需要我们填补。 ([],[])
    如果这个数据下有多个值比如: train_sample_table['ItemID'] 其中某个值为  12345|455151 那么 => [1234,1238] 组成一个二维的list
    但是如果， train_sample_table['ItemID'] 其中某个值只有一个元素  12345 则 =>依然返回一个二维数组  [1234,默认值] 组成一个二维的list，这里的默认值其实就是<PAD>对应的下标id值
    """
    ItemID_map_max_len = 1 #映射空间维度
    for key in ItemID_map:
        # print(">>>key")
        # print(key)  #key为ItemID，value为编码之后的值,是一个列表
        # print(">>>value")
        # print(ItemID_map[key])
        for cnt in range(ItemID_map_max_len - len(ItemID_map[key])):
            ItemID_map[key].insert(len(ItemID_map[key]) + cnt,ItemID2int['<PAD>'])



    train_sample_table['ItemID'] = train_sample_table['ItemID'].map(ItemID_map)
    test_sample_table['ItemID'] = test_sample_table['ItemID'].map(ItemID_map)
    # print(train_sample_table['ItemID'])
    print("ItemID finish")
    




    """
    User_CateIDs转数字字典
    第一步：构建特征取值种类set。 将训练集和测试的所有User_CateIDs列中按照"|"split之后，distinct之后建立一个set集合，包含CateID所有可能出现的情形包括为空情况
    第二步：构建映射MAP。 根据第一步的set集合，处理成一个MAP, 其中key为 User_CateIDs原始值，value为 一个从0开始一直到 CateID set元素 个数的整型数字。
    {CateID值:下标int_id} 的词典。
    """
    User_CateIDs_set = set()
    for val in train_common_features['User_CateIDs'].str.split('|'):
        User_CateIDs_set.update(val)
    for val in test_common_features['User_CateIDs'].str.split('|'):
        User_CateIDs_set.update(val)
    User_CateIDs_set.add('<PAD>')

    #User_CateIDs2int是一个词典
    #key为User_CateIDsset中的每一个元素，val为唯一标号的id从0开始。
    User_CateIDs2int = { val:ii for ii, val in enumerate(User_CateIDs_set) }

    #User_CateIDs 转成等长数字列表
    User_CateIDs_map = {val:[User_CateIDs2int[row] for row in val.split('|')]
                        for ii,val in enumerate(set(train_common_features['User_CateIDs']))}

    test_User_CateIDs_map = {val:[User_CateIDs2int[row] for row in val.split('|')]
                             #将对应的set(12155|15151|1111,55555|444444|1111,...)中的User_CateID转换为 对应的编码id，组成一个list, 即[]
                             for ii,val in enumerate(set(test_common_features['User_CateIDs']))}
    #这个set里面是set(12155|15151|1111,55555|444444|1111,...)
    # 那么 ii为元素下标，val对应的是一个个元素如 12155|15151|1111,55555|444444|1111
    # merge train & test
    User_CateIDs_map.update(test_User_CateIDs_map)
    """
    我们将用户浏览的CateID序列特征，编码为统一的100维的一个向量。不足100的用<PAD>对应的编码id补足。
    """
    User_CateIDs_map_max_len = 100
    # print("User_CateIDs_map max_len:", User_CateIDs_map_max_len)#由于再采样逻辑里面已经限制了最多有100个元素

    #不够100个的用User_CateIDs2int['<PAD>'] 补齐
    for key in User_CateIDs_map:
        # print("User_CateIDs_map  key >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        # print(key)
        # print("User_CateIDs_map  value >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")#是一个list
        # print(User_CateIDs_map[key])
        #将User_CateIDs_map的value是一个list 转化为固定长度的list 指定为User_CateIDs_map_max_len
        for cnt in range(User_CateIDs_map_max_len - len(User_CateIDs_map[key])):
            User_CateIDs_map[key].insert(len(User_CateIDs_map[key]) + cnt,User_CateIDs2int['<PAD>'])
    """
    User_CateIDs_map 是一个映射，这个映射是 449114|452429|450954|454027|452275|450655|455951|450656|450942|450944|450873|451448|451116|451391|446451|449224|451115 --> list(0,2,12545,1515,555)
    将具体的ID值映射为固定的整数id值，这个id值是从0到n的，n为这个ID取值的种类
    """
    """
    这里操作是把 
    key:如：449114|452429|450954|454027|452275|450655|455951|450656|450942|450944|450873|451448|451116|451391|446451|449224|451115
    映射为指定长度的list并且这个list中的元素是经过映射之后的唯一的id从0开始与key中“|”分割的每个元素一一对应
    """
    train_common_features['User_CateIDs'] = train_common_features['User_CateIDs'].map(User_CateIDs_map)
    test_common_features['User_CateIDs'] = test_common_features['User_CateIDs'].map(User_CateIDs_map)
    print("User_CateIDs finish")



    #User_BrandIDs转数字字典
    User_BrandIDs_set = set()
    for val in train_common_features['User_BrandIDs'].str.split('|'):
        User_BrandIDs_set.update(val)
    for val in test_common_features['User_BrandIDs'].str.split('|'):
        User_BrandIDs_set.update(val)
    User_BrandIDs_set.add('<PAD>')
    User_BrandIDs2int = {val:ii for ii, val in enumerate(User_BrandIDs_set)}
    #User_BrandIDs 转成等长数字列表
    User_BrandIDs_map = {val:[User_BrandIDs2int[row] for row in val.split('|')]
                         for ii,val in enumerate(set(train_common_features['User_BrandIDs']))}
    test_User_BrandIDs_map = {val:[User_BrandIDs2int[row] for row in val.split('|')]
                              for ii,val in enumerate(set(test_common_features['User_BrandIDs']))}
    # merge train & test
    User_BrandIDs_map.update(test_User_BrandIDs_map)
    User_BrandIDs_map_max_len = 100
    print("User_BrandIDs_map max_len:", User_BrandIDs_map_max_len)
    for key in User_BrandIDs_map:
        for cnt in range(User_BrandIDs_map_max_len - len(User_BrandIDs_map[key])):
            User_BrandIDs_map[key].insert(len(User_BrandIDs_map[key]) + cnt,User_BrandIDs2int['<PAD>'])
    train_common_features['User_BrandIDs'] = train_common_features['User_BrandIDs'].map(User_BrandIDs_map)
    test_common_features['User_BrandIDs'] = test_common_features['User_BrandIDs'].map(User_BrandIDs_map)
    print("User_BrandIDs finish")
    
    
    #userID 转数字字典
    UserID_set = set()
    for val in train_common_features['UserID']:
        UserID_set.add(val)
    for val in test_common_features['UserID']:
        UserID_set.add(val)
    UserID2int = {val:ii for ii, val in enumerate(UserID_set)}
    UserID_map_max_len = 1
    print("UserID_map max_len:", UserID_map_max_len)
    train_common_features['UserID'] = train_common_features['UserID'].map(UserID2int)
    test_common_features['UserID'] = test_common_features['UserID'].map(UserID2int)
    print("UserID finish")
    
    #User_Cluster 转数字字典
    User_Cluster_set = set()
    for val in train_common_features['User_Cluster']:
        User_Cluster_set.add(val)
    for val in test_common_features['User_Cluster']:
        User_Cluster_set.add(val)
    User_Cluster2int = {val:ii for ii, val in enumerate(User_Cluster_set)}
    User_Cluster_map_max_len = 1
    print("User_Cluster_map max_len:", User_Cluster_map_max_len)
    train_common_features['User_Cluster'] = train_common_features['User_Cluster'].map(User_Cluster2int)
    test_common_features['User_Cluster'] = test_common_features['User_Cluster'].map(User_Cluster2int)
    print("User_Cluster finish")
    
    #CategoryID 转数字字典
    CategoryID_set = set()
    for val in train_sample_table['CategoryID']:
        CategoryID_set.add(val)
    for val in test_sample_table['CategoryID']:
        CategoryID_set.add(val)
    CategoryID2int = {val:ii for ii, val in enumerate(CategoryID_set)}
    CategoryID_map_max_len = 1
    print("CategoryID_map max_len:", CategoryID_map_max_len)
    train_sample_table['CategoryID'] = train_sample_table['CategoryID'].map(CategoryID2int)
    test_sample_table['CategoryID'] = test_sample_table['CategoryID'].map(CategoryID2int)
    print("CategoryID finish")
    
    #ShopID 转数字字典
    ShopID_set = set()
    for val in train_sample_table['ShopID']:
        ShopID_set.add(val)
    for val in test_sample_table['ShopID']:
        ShopID_set.add(val)
    ShopID2int = {val:ii for ii, val in enumerate(ShopID_set)}
    ShopID_map_max_len = 1
    print("ShopID_map max_len:", ShopID_map_max_len)
    train_sample_table['ShopID'] = train_sample_table['ShopID'].map(ShopID2int)
    test_sample_table['ShopID'] = test_sample_table['ShopID'].map(ShopID2int)
    print("ShopID finish")

    #BrandID 转数字字典
    BrandID_set = set()
    for val in train_sample_table['BrandID']:
        BrandID_set.add(val)
    for val in test_sample_table['BrandID']:
        BrandID_set.add(val)
    BrandID2int = {val:ii for ii, val in enumerate(BrandID_set)}
    BrandID_map_max_len = 1
    print("BrandID_map max_len:", UserID_map_max_len)
    train_sample_table['BrandID'] = train_sample_table['BrandID'].map(BrandID2int)
    test_sample_table['BrandID'] = test_sample_table['BrandID'].map(BrandID2int)
    print("BrandID finish")
    
    #Com_CateID 转数字字典
    Com_CateID_set = set()
    for val in train_sample_table['Com_CateID']:
        Com_CateID_set.add(val)
    for val in test_sample_table['Com_CateID']:
        Com_CateID_set.add(val)
    Com_CateID2int = {val:ii for ii, val in enumerate(Com_CateID_set)}
    Com_CateID_map_max_len = 1
    print("Com_CateID_map max_len:", Com_CateID_map_max_len)
    train_sample_table['Com_CateID'] = train_sample_table['Com_CateID'].map(Com_CateID2int)
    test_sample_table['Com_CateID'] = test_sample_table['Com_CateID'].map(Com_CateID2int)
    print("Com_CateID finish")
    
    #Com_ShopID 转数字字典
    Com_ShopID_set = set()
    for val in train_sample_table['Com_ShopID']:
        Com_ShopID_set.add(val)
    for val in test_sample_table['Com_ShopID']:
        Com_ShopID_set.add(val)
    Com_ShopID2int = {val:ii for ii, val in enumerate(Com_ShopID_set)}
    Com_ShopID_map_max_len = 1
    print("Com_ShopID_map max_len:", Com_ShopID_map_max_len)
    train_sample_table['Com_ShopID'] = train_sample_table['Com_ShopID'].map(Com_ShopID2int)
    test_sample_table['Com_ShopID'] = test_sample_table['Com_ShopID'].map(Com_ShopID2int)
    print("Com_ShopID finish")
    
    #Com_BrandID 转数字字典
    Com_BrandID_set = set()
    for val in train_sample_table['Com_BrandID']:
        Com_BrandID_set.add(val)
    for val in test_sample_table['Com_BrandID']:
        Com_BrandID_set.add(val)
    Com_BrandID2int = {val:ii for ii, val in enumerate(Com_BrandID_set)}
    Com_BrandID_map_max_len = 1
    print("Com_BrandID_map max_len:", UserID_map_max_len)
    train_sample_table['Com_BrandID'] = train_sample_table['Com_BrandID'].map(Com_BrandID2int)
    test_sample_table['Com_BrandID'] = test_sample_table['Com_BrandID'].map(Com_BrandID2int)
    print("Com_BrandID finish")
    
    #PID 转数字字典
    PID_set = set()
    for val in train_sample_table['PID']:
        PID_set.add(val)
    for val in test_sample_table['PID']:
        PID_set.add(val)
    PID2int = {val:ii for ii, val in enumerate(PID_set)}
    PID_map_max_len = 1
    print("PID_map max_len:", PID_map_max_len)
    train_sample_table['PID'] = train_sample_table['PID'].map(PID2int)
    test_sample_table['PID'] = test_sample_table['PID'].map(PID2int)
    print("PID finish")




    """
    对原始数据处理之后，需要验证结果
    """
    #按照md5合并两个表
    train_data = pd.merge(train_sample_table, train_common_features, on='md5',how='inner')
    test_data = pd.merge(test_sample_table, test_common_features, on='md5',how='inner')

    # print("数据处理结束后1>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    # print(train_data.take(10,1))
    # print("数据处理结束后2>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    #将看不到的列显示完整
    pd.set_option('display.max_columns', 100)
    print(train_data.head)

    print("Sample/Common Merged")
    #将数据分成X和y两张表
    feature_fields = ['UserID','ItemID','User_Cluster', 'CategoryID','ShopID','BrandID','Com_CateID','Com_ShopID',
                      'Com_BrandID','PID','User_CateIDs','User_BrandIDs']
    target_fields = ['click','buy']
    print("train_data========================================")
    print(train_data[feature_fields])
    print(train_data[target_fields])
    train_features_pd, train_targets_pd = train_data[feature_fields], train_data[target_fields]
    train_features = train_features_pd.values
    print("print(train_features) ==== train_features_pd.values")
    print(train_features)
    train_targets_values = train_targets_pd.values
    """
    这里已经将原来User_CateIDs和User_BrandIDs单独的list数据和外层的所有元素打平成了一个大的list，如下形式
      list([137246, 124075, 125612, 123305, 315826, 268231, 244034, 192716, 218546, 179757, 25541, 291766, 58019, 259585, 270031, 67377, 160921, 148311, 68600, 226723, 20693, 160534, 330409, 52402, 134490, 205624, 182480, 93129, 24509, 264608, 96742, 306760, 177577, 17993, 202864, 34709, 199273, 68967, 31826, 64608, 217193, 321502, 137176, 212952, 122019, 164776, 193153, 215633, 271469, 280087, 198807, 315902, 66385, 127334, 209335, 221290, 335462, 136643, 192926, 277163, 177088, 268398, 78196, 35676, 189500, 174293, 154617, 344947, 60565, 83500, 135405, 90040, 157541, 30716, 208891, 174014, 230917, 98688, 329187, 237830, 262110, 17358, 201981, 58429, 328689, 196060, 290999, 199336, 194543, 247198, 101496, 261774, 166451, 127061, 35083, 271100, 229026, 192495, 235545, 259926])]]
    
    这里总结一下list特征在模型训练中的用法：
    即数据源初始的时候是对如User_CateIDs和User_BrandIDs这样的特征单独做一个list，里面是用户行为的序列；
    之后数据做了一步映射，即把用户浏览的CateIDs和BrandIDs映射成为一个固定的整数id值list，转化为一个list
    此时：
    User_CateIDs和User_BrandIDs
    [1,2,3,4] [5,6,7,8]的形式 ，并且他们list的长度是各自固定的，比如User_CateIDs为100个和User_BrandIDs为120个
    然后在输入到模型训练时，是他们和其他单特征一起组成一个大的list
    如[6,3,534,4545,[1,2,3,4] [5,6,7,8]] => [6,3,534,4545,1,2,3,4,5,6,7,8]
    模型在训练时候，要做embeeding，会对各个特征的取值在list中的位置做拆分，例如1号特征占据的是第一个位置，而User_BrandIDs占据的是第12--112个位置的特征
    
    然后这个User_CateIDs 为100列特征会映射(embedding)到一个固定的维度向量比如12维度  即100=>12的全连接层
    """
    print("print(train_features) ==== train_targets_pd.values")
    print(train_targets_values)
    """
[[0 0]
 [0 0]
 [0 0]
 ...
 [0 0]
 [0 0]
 [0 0]]
    """
    test_features_pd, test_targets_pd = test_data[feature_fields], test_data[target_fields]
    test_features = test_features_pd.values
    test_targets_values = test_targets_pd.values
    
    return UserID_map_max_len, ItemID_map_max_len, User_Cluster_map_max_len, User_CateIDs_map_max_len, User_BrandIDs_map_max_len, \
           CategoryID_map_max_len, ShopID_map_max_len, BrandID_map_max_len, Com_CateID_map_max_len,Com_ShopID_map_max_len, \
           Com_BrandID_map_max_len, PID_map_max_len, UserID2int, ItemID2int,User_Cluster2int, User_CateIDs2int, User_BrandIDs2int,  \
           CategoryID2int, ShopID2int, BrandID2int, Com_CateID2int, Com_ShopID2int, Com_BrandID2int, PID2int, \
           train_features, train_targets_values, train_data, test_features, test_targets_values, test_data
          


"""
加载数据并保存到本地
"""

print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>load_ESMM_Train_and_Test_Data()测试<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
UserID_map_max_len, ItemID_map_max_len, User_Cluster_map_max_len, User_CateIDs_map_max_len, User_BrandIDs_map_max_len, \
CategoryID_map_max_len, ShopID_map_max_len, BrandID_map_max_len, Com_CateID_map_max_len,Com_ShopID_map_max_len, \
Com_BrandID_map_max_len, PID_map_max_len, \
UserID2int, ItemID2int,User_Cluster2int, User_CateIDs2int, User_BrandIDs2int,  \
CategoryID2int, ShopID2int, BrandID2int, Com_CateID2int, Com_ShopID2int, Com_BrandID2int, PID2int, \
train_features, train_targets_values, train_data, test_features, test_targets_values, test_data = load_ESMM_Train_and_Test_Data()
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>load_ESMM_Train_and_Test_Data()测试结束<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")


# #数据校验
# print(User_CateIDs2int['<PAD>'])
# print(User_BrandIDs2int['<PAD>'])
# print(test_features[0:1,0:100]) #取出第1行的第1：100列数据
# print(train_features[0:1,0:100])
# print(train_features.take(11,1)[1000])
# print(test_targets_values[0:10])
# print(train_targets_values.shape)
# print(test_targets_values.shape)
#
#
#
# print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>train_features.take(10,1)测试<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
# user_cateids = np.zeros([2, 100])
# for i in range(2):
#     user_cateids[i] = train_features.take(10,1)[i]
# print(user_cateids)
# print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>train_features.take(10,1)测试结束<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
#

# """
# 模型设计
# 模型架构
# Embedding Lookup 示例
# """
#
# c = np.random.random([10, 1])
# b = tf.nn.embedding_lookup(c, [1])
# a = tf.nn.embedding_lookup(c, 1)
# with tf.Session() as sess:
#     sess.run(tf.initialize_all_variables())
#     print(sess.run(b))
#     print(sess.run(a))
#     print(c)
#
#
#
#
#
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
# """
# tf.reduce_sum()  测试
# # 'x' is [[1, 1, 1]
# #         [1, 1, 1]]
# x = tf.constant([[1, 1, 1], [1, 1, 1]])
# tf.reduce_sum(x) ==> 6 #对tensor x求和，6个1相加
# tf.reduce_sum(x, 0)=tf.reduce_sum(x, axis=0) ==> [2, 2, 2] #tensorflow中axis=0表示列，1表示行,所以上面是对x的列求和
# tf.reduce_sum(x, 1) ==> [3, 3]
# tf.reduce_sum(x, 1, keep_dims=True) ==> [[3], [3]] #对x求每一行的和，且保留原x的维度信息。
# tf.reduce_sum(x, [0, 1]) ==> 6 #对x的列和行求和
# """
# # 'x' is [[1, 1, 1]
# #         [1, 1, 1]]
# x = tf.constant([[1, 1, 1], [1, 1, 1]])
# tf.reduce_sum(x)
# tf.reduce_sum(x, 0)
# tf.reduce_sum(x, axis=0) #tensorflow中axis=0表示列，1表示行,所以上面是对x的列求和
# tf.reduce_sum(x, 1)
# tf.reduce_sum(x, 1, keep_dims=True)  #对x求每一行的和，且保留原x的维度信息。
# tf.reduce_sum(x, [0, 1]) #对x的列和行求和
#
#
# """
# tf.nn.embedding_lookup函数的用法
# tf.nn.embedding_lookup函数的用法主要是选取一个张量里面索引对应的元素。
# tf.nn.embedding_lookup（tensor, id）:tensor就是输入张量，id就是张量对应的索引，其他的参数不介绍。
# """
# import tensorflow as tf;
# import numpy as np;
# c = np.random.random([10,1])
# b = tf.nn.embedding_lookup(c, [1, 3])
# with tf.Session() as sess:
#     sess.run(tf.initialize_all_variables())
#     print (sess.run(b))
#     print (c)
#
#
# input_x = tf.placeholder(tf.int32, [None, 3], name='input_x')
# keep_prob = tf.placeholder(tf.float32, name='keep_prob')
# embedding = tf.get_variable('embedding', [5, 6])  #产生5行6列数据矩阵
# embedding_inputs = tf.nn.embedding_lookup(embedding, input_x) #
# x = [0, 1, 4]
# with tf.Session() as session:
#     session.run(tf.global_variables_initializer())
#     # 代码解释   这里session会run两个op，一个以list会给出[embedding, embedding_inputs] 其中两个op中的变量通过feed_dict={input_x: [x]}给出
#     y1, y2 = session.run([embedding, embedding_inputs], feed_dict={input_x: [x]}) #这里是lookup的id为[0,1,4]位置的数据
#     print(y1)
#     print(y2)
# """
#     print(y1)
# [[-3.7497959e-01 -1.3574934e-01  6.3621998e-04 -3.2882205e-01  -2.6098430e-02 -6.4995855e-01]
#  [ 4.9875981e-01  7.5718999e-02  1.6283578e-01  1.7291355e-01   6.0358101e-01 -2.5407076e-01]
#  [ 7.0739985e-03  5.7720226e-01 -7.6602280e-02 -5.9611917e-02   1.0062820e-01 -3.7122670e-01]
#  [ 1.3654292e-01 -4.5706773e-01 -2.6346862e-02  1.5305221e-02   1.6879094e-01 -1.1148870e-01]
#  [ 3.3369654e-01 -5.9345675e-01 -1.8395966e-01 -6.0551977e-01   5.6261355e-01  1.8064469e-01]]
#
#     print(y2)
# [[[-3.7497959e-01 -1.3574934e-01  6.3621998e-04 -3.2882205e-01   -2.6098430e-02 -6.4995855e-01]
#   [ 4.9875981e-01  7.5718999e-02  1.6283578e-01  1.7291355e-01    6.0358101e-01 -2.5407076e-01]
#   [ 3.3369654e-01 -5.9345675e-01 -1.8395966e-01 -6.0551977e-01    5.6261355e-01  1.8064469e-01]]]
# """
#
#
#
#
#
# """
# tf.random_normal()函数
# tf.random_normal()函数用于从服从指定正太分布的数值中取出指定个数的值。
# tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
# •	    shape: 输出张量的形状，必选
# •	    mean: 正态分布的均值，默认为0
# •	    stddev: 正态分布的标准差，默认为1.0
# •	    dtype: 输出的类型，默认为tf.float32
# •	    seed: 随机数种子，是一个整数，当设置之后，每次生成的随机数都一样
# •	    name: 操作的名称
# 以下程序定义一个w1变量：
# 变量w1声明之后并没有被赋值，需要在Session中调用run(tf.global_variables_initializer())方法初始化之后才会被具体赋值。
# tf中张量与常规向量不同的是执行"print w1"输出的是w1的形状和数据类型等属性信息，获取w1的值需要调用sess.run(w1)方法。
# 输出：
# <tf.Variable 'Variable:0' shape=(2, 3) dtype=float32_ref>
# [[-0.81131822  1.48459876  0.06532937]
#  [-2.4427042   0.0992484   0.59122431]]
# """
#
# # -*- coding: utf-8 -*-)
# import tensorflow as tf
#
# w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     # sess.run(tf.initialize_all_variables())  #比较旧一点的初始化变量方法
#     print (w1)
#     print (sess.run(w1))
#
#
#
# # ### Tensorflow slice 示例
# import tensorflow as tf
# import numpy as np
#
# t = tf.constant([[1,100],[2,99],[3,98]])
# t1 = tf.slice(t, [0,1], [-1, 1])  #  [0,1] 中的0表示从第0行开始抽取   1表示从[1,100]中的第几个元素开始抽取，这里从第一个开始也就是100开始抽取
#                                   # [-1, 1] 第一个1决定了，从第0行以起始位置抽取多少行，"-1"表示所有行，第二个"1"表示抽取1个元素
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
# x = [[1, 2, 3], [4, 5, 6]]
# y = np.arange(24).reshape([2, 3, 4])
# z = tf.constant([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]], [[13, 14, 15], [16, 17, 18]]])
# sess = tf.Session()
# begin_x = [1, 0]  # 第一个1，决定了从x的第二行[4,5,6]开始，第二个0，决定了从[4,5,6] 中的4开始抽取
# size_x = [1, 2]  # 第一个1决定了，从第二行以起始位置抽取1行，也就是只抽取[4,5,6] 这一行，在这一行中从4开始抽取2个元素
# out = tf.slice(x, begin_x, size_x)
# print
# sess.run(out)  # 结果:[[4 5]]
#
# begin_y = [1, 0, 0]
# size_y = [1, 2, 3]
# out = tf.slice(y, begin_y, size_y)
# print
# sess.run(out)  # 结果:[[[12 13 14] [16 17 18]]]
#
# begin_z = [0, 1, 1]
# size_z = [-1, 1, 2]
# out = tf.slice(z, begin_z, size_z)
# print
# sess.run(out)  # size[i]=-1 表示第i维从begin[i]剩余的元素都要被抽取，结果：[[[ 5  6]] [[11 12]] [[17 18]]]
#
#
#
#
# """
# tf.concat()
# tensorflow中用来拼接张量的函数tf.concat()，用法:
# tf.concat([tensor1, tensor2, tensor3,...], axis)
# 先给出tf源代码中的解释:
#   t1 = [[1, 2, 3], [4, 5, 6]]
#   t2 = [[7, 8, 9], [10, 11, 12]]
#   tf.concat([t1, t2], 0)  # [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
#   tf.concat([t1, t2], 1)  # [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]
#
#   # tensor t3 with shape [2, 3]
#   # tensor t4 with shape [2, 3]
#   tf.shape(tf.concat([t3, t4], 0))  # [4, 3]
#   tf.shape(tf.concat([t3, t4], 1))  # [2, 6]
# 这里解释了当axis=0和axis=1的情况，怎么理解这个axis呢？其实这和numpy中的np.concatenate()用法是一样的。
# axis=0     代表在第0个维度拼接
# axis=1     代表在第1个维度拼接
# 对于一个二维矩阵，第0个维度代表最外层方括号所框下的子集，第1个维度代表内部方括号所框下的子集。维度越高，括号越小。
# 对于这种情况，我可以再解释清楚一点:
# 对于[ [ ], [ ]]和[[ ], [ ]]，低维拼接等于拿掉最外面括号，高维拼接是拿掉里面的括号(保证其他维度不变)。注意：tf.concat()拼接的张量只会改变一个维度，其他维度是保存不变的。比如两个shape为[2,3]的矩阵拼接，要么通过axis=0变成[4,3]，要么通过axis=1变成[2,6]。改变的维度索引对应axis的值。
# 这样就可以理解多维矩阵的拼接了，可以用axis的设置来从不同维度进行拼接。
# 对于三维矩阵的拼接，自然axis取值范围是[0, 1, 2]。
# 对于axis等于负数的情况
# 负数在数组索引里面表示倒数(countdown)。比如，对于列表ls = [1,2,3]而言，ls[-1] = 3，表示读取倒数第一个索引对应值。
# axis=-1表示倒数第一个维度，对于三维矩阵拼接来说，axis=-1等价于axis=2。同理，axis=-2代表倒数第二个维度，对于三维矩阵拼接来说，axis=-2等价于axis=1。
# 一般在维度非常高的情况下，我们想在最'高'的维度进行拼接，一般就直接用countdown机制，直接axis=-1就搞定了
# """