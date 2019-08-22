import tensorflow as tf
from tensorflow import feature_column
from tensorflow.python.feature_column.feature_column import _LazyBuilder

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def test_crossed_column():
    """ crossed column测试 """
    featrues = {
        'price': [['A'], ['B'], ['C']],
        'color': [['R'], ['G'], ['B']]
    }
    price = feature_column.categorical_column_with_vocabulary_list('price', ['A', 'B', 'C', 'D'])
    color = feature_column.categorical_column_with_vocabulary_list('color', ['R', 'G', 'B'])
    p_x_c = feature_column.crossed_column([price, color], 16)

    with tf.device('/cpu:0'):
        p_x_c_identy = feature_column.indicator_column(p_x_c)
    #p_x_c_identy = feature_column.indicator_column(p_x_c)
        p_x_c_identy_dense_tensor = feature_column.input_layer(featrues, [p_x_c_identy])
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print(session.run([p_x_c_identy_dense_tensor]))
test_crossed_column()


# def test_crossed_column():
#     """ crossed column测试 """
#     #源数据
#     featrues = {
#         'price': [['A'], ['B'], ['C']], # 0,1,2
#         'color': [['R'], ['G'], ['B']]  # 0,1,2
#     }
#     # categorical_column
#     price = feature_column.categorical_column_with_vocabulary_list('price', ['A', 'B', 'C', 'D'])
#     color = feature_column.categorical_column_with_vocabulary_list('color', ['R', 'G', 'B'])
#
#     #crossed_column 产生稀疏表示
#     p_x_c = feature_column.crossed_column([price, color], 16)
#
#     # 稠密表示
#     p_x_c_identy = feature_column.indicator_column(p_x_c)
#
#     # crossed_column 连接 源数据
#     p_x_c_identy_dense_tensor = feature_column.input_layer(featrues, [p_x_c_identy])
#
#     with tf.Session() as session:
#         session.run(tf.global_variables_initializer())
#         session.run(tf.tables_initializer())
#         print(session.run([p_x_c_identy_dense_tensor]))
# test_crossed_column()

