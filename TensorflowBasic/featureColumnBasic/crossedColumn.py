import tensorflow as tf

featrues = {
        'longtitude': [19,61,30,9,45],
        'latitude': [45,40,72,81,24]
    }

longtitude = tf.feature_column.numeric_column('longtitude')
latitude = tf.feature_column.numeric_column('latitude')

longtitude_b_c = tf.feature_column.bucketized_column(longtitude, [33,66])
latitude_b_c  = tf.feature_column.bucketized_column(latitude,[33,66])

column = tf.feature_column.crossed_column([longtitude_b_c, latitude_b_c], 12)
with tf.device('/cpu:0'):
    indicator = tf.feature_column.indicator_column(column)
    tensor = tf.feature_column.input_layer(featrues, [indicator])

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    print(session.run([tensor]))