import tensorflow as tf

pets = {'pets': ['rabbit','pig','dog','mouse','cat']}  

column = tf.feature_column.categorical_column_with_vocabulary_list(
    key='pets',
    vocabulary_list=['cat','dog','rabbit','pig'], 
    dtype=tf.string, 
    default_value=-1,
    num_oov_buckets=3)

indicator = tf.feature_column.indicator_column(column)
tensor = tf.feature_column.input_layer(pets, [indicator])

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    print(session.run([tensor]))