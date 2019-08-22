# -*- coding: utf-8 -*-
"""
1.将连续值特征按照数值大小分类
用tf.feature_column.bucketized_column函数将连续值按照指定的阈值进行分段，从而将连续值映射到离散值上。

2.将整数值直接映射到one-hot编码
如果连续值特征列的数据是整数，则还可以直接用tf.feature_column. categorical_column_with_identity函数将其映射成one-hot编码。
函数tf.feature_column.categorical_column_with_identity的参数和返回值解读如下。
•	需要传入两个必填的参数：列名称（key）、类的总数（num_buckets）。其中，num_buckets的值一定要大于key列中所有数据的最大值。
•	返回值：为_IdentityCategoricalColumn对象。该对象是使用稀疏矩阵的方式存放转化后的数据。如果要将该返回值作为输入层传入后续的网络，则需要用indicator_column函数将其转化为稠密矩阵。

"""
import tensorflow as tf

def test_numeric_cols_to_bucketized():
    price = tf.feature_column.numeric_column('price')#定义连续数值的特征列

    #将连续数值转成离散值的特征列,离散值共分为3段：小于3、在3与5之间、大于5
    price_bucketized = tf.feature_column.bucketized_column(  price, boundaries=[3.,5.])

    features = {                        #传定义字典
          'price': [[2.], [6.]],
      }

    net = tf.feature_column.input_layer(features,[ price,price_bucketized]) #生成输入层张量
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(net.eval()) 

test_numeric_cols_to_bucketized()


def test_numeric_cols_to_identity():
    tf.reset_default_graph()
    price = tf.feature_column.numeric_column('price')#定义连续数值的特征列
    categorical_column = tf.feature_column.categorical_column_with_identity('price', 6)
    print(type(categorical_column))
    with tf.device('/cpu:0'):
        one_hot_style = tf.feature_column.indicator_column(categorical_column)
    features = {                        #传定义字典
          'price': [[2], [4]],
      }
    with tf.device('/cpu:0'):
        net = tf.feature_column.input_layer(features,[ price,one_hot_style]) #生成输入层张量
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(net.eval()) 

test_numeric_cols_to_identity()