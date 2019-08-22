# coding: utf-8
"""
构建分布式Tensorflow模型系列四:特征工程
特征工程是机器学习流程中重要的一个环节，即使是通常用来做端到端学习的深度学习模型在训练之前也免不了要做一些特征工程相关的工作。
Tensorflow平台提供的FeatureColumn API为特征工程提供了强大的支持。
Feature cloumns是原始数据和Estimator模型之间的桥梁，它们被用来把各种形式的原始数据转换为模型能够使用的格式。
深度神经网络只能处理数值数据，网络中的每个神经元节点执行一些针对输入数据和网络权重的乘法和加法运算。
然而，现实中的有很多非数值的类别数据，比如产品的品牌、类目等，这些数据如果不加转换，神经网络是无法处理的。
另一方面，即使是数值数据，在扔给网络进行训练之前有时也需要做一些处理，比如标准化、离散化等。

"""
import tensorflow as tf
import numpy as np

from tensorflow import feature_column
from tensorflow.python.feature_column.feature_column import _LazyBuilder
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#
# def model_fn(features, target, mode, params)
#   predictions = tf.stack(tf.fully_connected, [50, 50, 1])
#   loss = tf.losses.mean_squared_error(target, predictions)
#   train_op = tf.train.create_train_op(
#     loss, tf.train.get_global_step(),
#     params['learning_rate'], params['optimizer'])
#   return EstimatorSpec(mode=mode,predictions=predictions,loss=loss, train_op=train_op)
#
#
#
# #一个使用Head简化model_fn编写的例子如下：
# def model_fn(features, target, mode, params):
#   last_layer = tf.stack(tf.fully_connected, [50, 50])
#   head = tf.multi_class_head(n_classes=10)
#   return head.create_estimator_spec(
#     features, mode, last_layer,
#     label=target,
#     train_op_fn=lambda loss: my_optimizer.minimize(loss, tf.train.get_global_step())
#
#
#
#
# def model_fn(features, target, mode, params):
#       last_layer = tf.stack(tf.fully_connected, [50, 50])
#       head1 = tf.multi_class_head(n_classes=2, label_name=’y’, head_name =’h1’)
#       head2 = tf.multi_class_head(n_classes=10, label_name=’z’, head_name =’h2’)
#       head = tf.multi_head([head1, head2])
#       return head.create_model_fn_ops(features,
#                                       features, mode, last_layer,
#                                       label=target,
#                                       train_op_fn=lambda loss: my_optimizer.minimize(loss, tf.train.get_global_step()))
#

# import tensorflow as tf
# import tensorflow.contrib.eager as tfe
# tfe.enable_eager_execution() #开启Eager模式
# a = tf.constant([5], dtype=tf.int32)
# for i in range(a):
#   print (i)



"""
线性模型LinearModel
对所有特征进行线性加权操作（数值和权重值相乘）。
语法格式
linear_model(
    features,
    feature_columns,
    units=1,
    sparse_combiner='sum',
    weight_collections=None,
    trainable=True
)
测试代码
"""

import tensorflow as tf
from tensorflow.python.feature_column.feature_column import _LazyBuilder
def get_linear_model_bias():
    with tf.variable_scope('linear_model', reuse=True):
        return tf.get_variable('bias_weights')

def get_linear_model_column_var(column):
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,'linear_model/' + column.name)[0]
featrues = {
        'price': [[1.0], [5.0], [10.0]],
        'color': [['R'], ['G'], ['B']]
    }
price_column = tf.feature_column.numeric_column('price')
color_column = tf.feature_column.categorical_column_with_vocabulary_list('color',['R', 'G', 'B'])
prediction = tf.feature_column.linear_model(featrues, [price_column, color_column])

bias = get_linear_model_bias()
price_var = get_linear_model_column_var(price_column)
color_var = get_linear_model_column_var(color_column)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    sess.run(tf.tables_initializer())

    sess.run(bias.assign([7.0]))
    sess.run(price_var.assign([[10.0]]))
    sess.run(color_var.assign([[2.0], [2.0], [2.0]]))

    predication_result = sess.run([prediction])

    print(prediction)
    print(predication_result)

