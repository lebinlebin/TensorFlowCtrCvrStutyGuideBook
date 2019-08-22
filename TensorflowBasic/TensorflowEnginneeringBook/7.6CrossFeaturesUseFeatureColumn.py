# -*- coding: utf-8 -*-
"""
将多个单列特征混合起来生成交叉列，并将交叉列作为新的样本特征，与原始的样本数据一起输入模型进行计算。
用tf.feature_column.crossed_column函数将特征列b和c混合在一起，生成交叉列。该函数有以下两个必填参数。
•	key：要进行交叉计算的列。以列表形式传入（代码中是[b,ꞌcꞌ]）。
•	hash_bucket_size：要散列的数值范围（代码中是5）。表示将特征列交叉合并后，经过hash算法计算并散列成0~4之间的整数。
"""

"""
提示：
tf.feature_column.crossed_column函数的输入参数key是一个列表类型。该列表的元素可以是指定的列名称（字符串形式），也可以是具体的特征列对象（张量形式）。
如果传入的是特征列对象，则还要考虑特征列类型的问题。因为tf.feature_column.crossed_column函数不支持对numeric_column类型的特征列做交叉运算，
所以，如果要对numeric_column类型的列做交叉运算，则需要用bucketized_column函数或categorical_column_with_identity函数将numeric_column类型转化后才能使用。
"""
import tensorflow as tf
from tensorflow.python.feature_column.feature_column import _LazyBuilder


def test_crossed():
    a = tf.feature_column.numeric_column('a', dtype=tf.int32, shape=(2,))
    b = tf.feature_column.bucketized_column(a, boundaries=(0, 1))  # 离散值转化
    crossed = tf.feature_column.crossed_column([b, 'c'], hash_bucket_size=5)  # 生成交叉列

    builder = _LazyBuilder({  # 生成模拟输入的数据
        'a':
            tf.constant(((-1., -1.5), (.5, 1.))),
        'c':
            tf.SparseTensor(
                indices=((0, 0), (1, 0), (1, 1)),
                values=['cA', 'cB', 'cC'],
                dense_shape=(2, 2)),
    })
    id_weight_pair = crossed._get_sparse_tensors(builder)  # 生成输入层张量
    with tf.Session() as sess2:  # 建立会话session，取值
        id_tensor_eval = id_weight_pair.id_tensor.eval()
        print(id_tensor_eval)  # 输出稀疏矩阵

        dense_decoded = tf.sparse_tensor_to_dense(id_tensor_eval, default_value=-1).eval(session=sess2)
        print(dense_decoded)  # 输出稠密矩阵


test_crossed()

