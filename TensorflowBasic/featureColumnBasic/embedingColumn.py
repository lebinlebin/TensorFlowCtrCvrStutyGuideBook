import tensorflow as tf

features = {'pets': ['dog','cat','rabbit','pig','mouse']}  

pets_f_c = tf.feature_column.categorical_column_with_vocabulary_list(
    'pets',
    ['cat','dog','rabbit','pig'], 
    dtype=tf.string, 
    default_value=-1)

column = tf.feature_column.embedding_column(pets_f_c, 3)
tensor = tf.feature_column.input_layer(features, [column])

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())

    print(session.run([tensor]))