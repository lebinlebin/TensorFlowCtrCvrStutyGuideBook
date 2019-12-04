import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import json
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.python.feature_column.feature_column import _LazyBuilder

"""
本文件测试用户浏览序列类特征与权重类特征进行对应加权
同时，用户浏览序列与对应的帖子的属性进行共享emdbeding
数据embedding之后的数据
"""


def shared_embedding_column_with_hash_bucket():

    features = {'L1': [[410387, 415955, 412596, 416526, 416805, 408844, 418514, 411611, 415266],
                          [410387, 415955, 412596, 416526, 416805, 408844, 418514, 411611, 415266]],
                'LW1': [[44.0, 33.0, 17.0, 6.0, 3.0, 2.0, 1.0, 1.0, 1.0],[44.0, 33.0, 17.0, 6.0, 3.0, 2.0, 1.0, 1.0, 1.0]],
                 'a2': [[410387],[415955]]
                 }
    """
    这两个编码的映射hash_bucket_size要是统一的一个值，这里是40
    """
    brandlist = tf.feature_column.categorical_column_with_hash_bucket(key='L1',hash_bucket_size=40,dtype=tf.int64)
    brandweighteds = tf.feature_column.weighted_categorical_column(brandlist, 'LW1',dtype=tf.float32)
    brand = tf.feature_column.categorical_column_with_hash_bucket(key='a2',hash_bucket_size=40,dtype=tf.int64)
    brand_embed = feature_column.shared_embedding_columns([brandweighteds, brand], 5, combiner='sum',shared_embedding_collection_name="brand")  # mark
    print(brand_embed)
    color_dense_tensor = feature_column.input_layer(features,brand_embed)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print('shared_embedding_columns' + '_' * 40)
        print(session.run(color_dense_tensor))


shared_embedding_column_with_hash_bucket()