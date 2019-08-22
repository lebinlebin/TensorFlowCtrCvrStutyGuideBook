import os
import tensorflow as tf

pets = {'pets': ['rabbit','pig','dog','mouse','cat']}  

dir_path = os.path.dirname(os.path.realpath(__file__))
fc_path=os.path.join(dir_path,'pets_fc.txt')

column=tf.feature_column.categorical_column_with_vocabulary_file(
        key="pets",
        vocabulary_file=fc_path,
        num_oov_buckets=0)

indicator = tf.feature_column.indicator_column(column)
tensor = tf.feature_column.input_layer(pets, [indicator])

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    print(session.run([tensor]))