import tensorflow as tf
from tensorflow.python.feature_column.feature_column import _LazyBuilder


def get_linear_model_bias():
    with tf.variable_scope('linear_model', reuse=True):
        return tf.get_variable('bias_weights')


def get_linear_model_column_var(column):
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                             'linear_model/' + column.name)[0]

featrues = {
        'price': [[1.0], [5.0], [10.0]],
        'color': [['R'], ['G'], ['B']]
    }

price_column = tf.feature_column.numeric_column('price')
color_column = tf.feature_column.categorical_column_with_vocabulary_list('color',
                                                                      ['R', 'G', 'B'])
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