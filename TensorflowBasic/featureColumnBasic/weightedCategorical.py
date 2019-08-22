import tensorflow as tf
from tensorflow.python.feature_column.feature_column import _LazyBuilder

features = {'color': [['R'], ['A'], ['G'], ['B'],['R']],
                  'weight': [[1.0], [5.0], [4.0], [8.0],[3.0]]}

color_f_c = tf.feature_column.categorical_column_with_vocabulary_list(
    'color', ['R', 'G', 'B','A'], dtype=tf.string, default_value=-1
)

column = tf.feature_column.weighted_categorical_column(color_f_c, 'weight')

indicator = tf.feature_column.indicator_column(column)
tensor = tf.feature_column.input_layer(features, [indicator])

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    print(session.run([tensor]))
