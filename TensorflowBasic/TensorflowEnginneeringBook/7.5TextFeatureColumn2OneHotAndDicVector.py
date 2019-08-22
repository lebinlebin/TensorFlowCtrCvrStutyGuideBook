# -*- coding: utf-8 -*-
"""
"""
import tensorflow as tf
from tensorflow.python.feature_column.feature_column import _LazyBuilder
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# 将离散文本按照指定范围散列
def test_categorical_cols_to_hash_bucket():
    tf.reset_default_graph()
    some_sparse_column = tf.feature_column.categorical_column_with_hash_bucket(
        'sparse_feature', hash_bucket_size=5)  # 稀疏矩阵，单独放进去会出错

    builder = _LazyBuilder({
        'sparse_feature': [['a'], ['x']],
    })
    id_weight_pair = some_sparse_column._get_sparse_tensors(builder)  #

    with tf.Session() as sess:
        id_tensor_eval = id_weight_pair.id_tensor.eval()
        print("稀疏矩阵：\n", id_tensor_eval)

        dense_decoded = tf.sparse_tensor_to_dense(id_tensor_eval, default_value=-1).eval(session=sess)
        print("稠密矩阵：\n", dense_decoded)


test_categorical_cols_to_hash_bucket()

from tensorflow.python.ops import lookup_ops


# 将离散文本按照指定词表与指定范围，混合散列
def test_with_1d_sparse_tensor():
    tf.reset_default_graph()
    # 混合散列
    body_style = tf.feature_column.categorical_column_with_vocabulary_list(
        'name', vocabulary_list=['anna', 'gary', 'bob'], num_oov_buckets=2)  # 稀疏矩阵

    # 稠密矩阵
    builder = _LazyBuilder({
        'name': ['anna', 'gary', 'alsa'],
    })

    # 稀疏矩阵
    builder2 = _LazyBuilder({
        'name': tf.SparseTensor(
            indices=((0,), (1,), (2,)),
            values=('anna', 'gary', 'alsa'),
            dense_shape=(3,)),
    })

    id_weight_pair = body_style._get_sparse_tensors(builder)  #
    id_weight_pair2 = body_style._get_sparse_tensors(builder2)  #

    with tf.Session() as sess:
        sess.run(lookup_ops.tables_initializer())

        id_tensor_eval = id_weight_pair.id_tensor.eval()
        print("稀疏矩阵：\n", id_tensor_eval)
        id_tensor_eval2 = id_weight_pair2.id_tensor.eval()
        print("稀疏矩阵2：\n", id_tensor_eval2)

        dense_decoded = tf.sparse_tensor_to_dense(id_tensor_eval, default_value=-1).eval(session=sess)
        print("稠密矩阵：\n", dense_decoded)


test_with_1d_sparse_tensor()


# 将离散文本转为onehot特征列
def test_categorical_cols_to_onehot():
    tf.reset_default_graph()
    some_sparse_column = tf.feature_column.categorical_column_with_hash_bucket(
        'sparse_feature', hash_bucket_size=5)  # 定义散列特征列

    # 转换成one-hot特征列
    one_hot_style = tf.feature_column.indicator_column(some_sparse_column)

    features = {
        'sparse_feature': [['a'], ['x']],
    }

    net = tf.feature_column.input_layer(features, one_hot_style)  # 生成输入层张量
    with tf.Session() as sess:  # 通过会话输出数据
        print(net.eval())


test_categorical_cols_to_onehot()


# 将离散文本转为onehot词嵌入特征列
def test_categorical_cols_to_embedding():
    tf.reset_default_graph()
    some_sparse_column = tf.feature_column.categorical_column_with_hash_bucket(
        'sparse_feature', hash_bucket_size=5)  # 稀疏矩阵，单独放进去会出错

    embedding_col = tf.feature_column.embedding_column(some_sparse_column, dimension=3)

    features = {
        'sparse_feature': [['a'], ['x']],
    }

    # 生成输入层张量
    cols_to_vars = {}
    net = tf.feature_column.input_layer(features, embedding_col, cols_to_vars)

    with tf.Session() as sess:  # 通过会话输出数据
        sess.run(tf.global_variables_initializer())
        print("test_categorical_cols_to_embedding")
        print(net.eval())


test_categorical_cols_to_embedding()


"""
# input_layer中的顺序

5.多特征列的顺序
在大多数情况下，会将转化好的特征列统一放到input_layer函数中制作成一个输入样本。
input_layer函数支持的输入类型有以下4种：
•	numeric_column特征列。
•	bucketized_column特征列。
•	indicator_column特征列。
•	embedding_column特征列。
如果要将7.4.3小节中的hash值或词表散列的值传入input_layer函数中，则需要先将其转化成indicator_column类型或embedding_column类型。
当多个类型的特征列放在一起时，系统会按照特征列的名字进行排序。
"""
def test_order():
    tf.reset_default_graph()
    numeric_col = tf.feature_column.numeric_column('numeric_col')
    some_sparse_column = tf.feature_column.categorical_column_with_hash_bucket(
        'asparse_feature', hash_bucket_size=5)  # 稀疏矩阵，单独放进去会出错

    embedding_col = tf.feature_column.embedding_column(some_sparse_column, dimension=3)
    # 转换成one-hot特征列
    one_hot_col = tf.feature_column.indicator_column(some_sparse_column)
    print(one_hot_col.name)
    print(embedding_col.name)
    print(numeric_col.name)

    features = {
        'numeric_col': [[3], [6]],
        'asparse_feature': [['a'], ['x']],
    }

    # 生成输入层张量
    cols_to_vars = {}
    net = tf.feature_column.input_layer(features, [numeric_col, one_hot_col, embedding_col], cols_to_vars)

    with tf.Session() as sess:  # 通过会话输出数据
        sess.run(tf.global_variables_initializer())
        print(net.eval())


test_order()
