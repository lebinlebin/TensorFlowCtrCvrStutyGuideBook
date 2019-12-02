import tensorflow as tf

price = {'price': [[1.], [2.], [3.], [4.],[-1.]]}

column = tf.feature_column.numeric_column('price', normalizer_fn=lambda x:x+2)
column2 = tf.feature_column.numeric_column('price', shape=(),normalizer_fn=lambda x:x+2)
print (column)
print (column2)
tensor1 = tf.feature_column.input_layer(price,[column])
tensor2 = tf.feature_column.input_layer(price,[column2])

with tf.Session() as session:
    print(session.run([tensor1]))
    print(session.run([tensor2]))