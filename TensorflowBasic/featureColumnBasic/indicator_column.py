"""
tensorflow-gpu  bug
这里对于windows版本的tensorflow-gpu版本对于
tf.feature_column.indicator_column

有bug。
同时tf.one_hot()也有这个bug

解决方法是放在CPU上运行
with tf.device('/cpu:0'):
    indicator = tf.feature_column.indicator_column(column)
    tensor = tf.feature_column.input_layer(pets, [indicator])
"""
import tensorflow as tf
pets = {'pets': ['rabbit','pig','dog','mouse','cat']}
column = tf.feature_column.categorical_column_with_vocabulary_list(
    key='pets',
    vocabulary_list=['cat','dog','rabbit','pig'], #与categorical_column_with_identity这里不需要对原始数据进行数字表示
    dtype=tf.string,
    default_value=-1,
    num_oov_buckets=3)
indicator = tf.feature_column.indicator_column(column)
# 通过indicator_column，将稀疏的转换成dense，也就是one-hot形式，只是multi-hot
tensor = tf.feature_column.input_layer(pets, [indicator])

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    print(session.run([tensor]))
