# -*- coding: utf-8 -*-
"""
"""

import tensorflow as tf
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 设置当前使用的GPU设备仅为0号设备  设备名称为'/gpu:0'

tf.reset_default_graph()

vocabulary_size = 3  # 假如有3个词，向量为0，1，2

"""
原始数据
"""
sparse_input_a = tf.SparseTensor(  # 定义一个稀疏矩阵,  值为：
    indices=((0, 0), (1, 0), (1, 1)),  # [2]   只有一个序列
    values=(2, 0, 1),  # [0, 1] 有两个序列
    dense_shape=(2, 2))

sparse_input_b = tf.SparseTensor(  # 定义一个稀疏矩阵,  值为：
    indices=((0, 0), (1, 0), (1, 1)),  # [1]
    values=(1, 2, 0),  # [2, 0]
    dense_shape=(2, 2))


"""
词嵌入初始化
"""
embedding_dimension_a = 2
embedding_values_a = (  # 为稀疏矩阵的三个值（0，1，2）匹配词嵌入初始值
    (1., 2.),  # id 0
    (3., 4.),  # id 1
    (5., 6.)  # id 2
)
embedding_dimension_b = 3
embedding_values_b = (  # 为稀疏矩阵的三个值（0，1，2）匹配词嵌入初始值
    (11., 12., 13.),  # id 0
    (14., 15., 16.),  # id 1
    (17., 18., 19.)  # id 2
)


def _get_initializer(embedding_dimension, embedding_values):  # 自定义初始化词嵌入
    def _initializer(shape, dtype, partition_info):
        return embedding_values
    return _initializer


categorical_column_a = tf.contrib.feature_column.sequence_categorical_column_with_identity(  # 带序列的离散列
    key='a', num_buckets=vocabulary_size)  # vocabulary_size = 3   #假如有3个词，向量为0，1，2  这里就是定义词向量的个数，又多少词就有多少词向量

"""
获取词向量，维数设定为2
"""
embedding_column_a = tf.feature_column.embedding_column(  # 将离散列转为词向量
    categorical_column_a, dimension=embedding_dimension_a,  ##embedding_dimension_a = 2
    initializer=_get_initializer(embedding_dimension_a, embedding_values_a))


categorical_column_b = tf.contrib.feature_column.sequence_categorical_column_with_identity(
    key='b', num_buckets=vocabulary_size)
"""
获取词向量，维数设定为3
"""
embedding_column_b = tf.feature_column.embedding_column(
    categorical_column_b, dimension=embedding_dimension_b,  ## embedding_dimension_b = 3
    initializer=_get_initializer(embedding_dimension_b, embedding_values_b))

"""
词向量共享
词向量维度设定为2
"""
shared_embedding_columns = tf.feature_column.shared_embedding_columns(  # 共享列
    [categorical_column_b, categorical_column_a],
    dimension=embedding_dimension_a,  ##embedding_dimension_a = 2
    initializer=_get_initializer(embedding_dimension_a, embedding_values_a))





features = {  # 将a,b合起来
    'a': sparse_input_a,
    'b': sparse_input_b,
}

"""
对接真实数据
"""
input_layer, sequence_length = tf.contrib.feature_column.sequence_input_layer(  # 定义序列输入层
    features,
    feature_columns=[embedding_column_b, embedding_column_a])



input_layer2, sequence_length2 = tf.contrib.feature_column.sequence_input_layer(  # 定义序列输入层
    features,
    feature_columns=shared_embedding_columns)

global_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)  # 返回图中的张量（2个嵌入词权重）
print([v.name for v in global_vars])

with tf.train.MonitoredSession() as sess:
    # print(global_vars[0].eval(session=sess))  # 输出词向量的初始值
    # print(global_vars[1].eval(session=sess))
    # print(global_vars[2].eval(session=sess))
    # print(sequence_length.eval(session=sess))
    print(input_layer.eval(session=sess))  # 输出序列输入层的内容
    # print(sequence_length2.eval(session=sess))
    # print(input_layer2.eval(session=sess))  # 输出序列输入层的内容

    # print(sess.run([shared_embedding_columns]))