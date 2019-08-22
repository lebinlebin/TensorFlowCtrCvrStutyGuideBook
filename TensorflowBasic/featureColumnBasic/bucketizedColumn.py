import tensorflow as tf

years = {'years': [1999,2013,1987,2005]}  

years_fc = tf.feature_column.numeric_column('years')
column = tf.feature_column.bucketized_column(years_fc, [1990, 2000, 2010])
print(column)
tensor = tf.feature_column.input_layer(years, [column])

with tf.Session() as session:
    print(session.run([tensor]))