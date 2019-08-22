import tensorflow as tf
x=[[1, 1, 1]]
# 'x' is [[1, 1, 1]
#         [1, 1, 1]]
print(tf.reduce_sum(x)) #==> 6
# print(tf.reduce_sum(x, 0)) #==> [2, 2, 2]
print(tf.reduce_sum(x, 1)) #==> [3, 3]
# print(tf.reduce_sum(x, 1, keep_dims=True)) #==> [[3], [3]]
print(tf.reduce_sum(x, [0, 1])) #==> 6

"""
reduce_sum应该理解为压缩求和，用于降维



# 'x' is [[1, 1, 1]

#         [1, 1, 1]]

#求和

tf.reduce_sum(x) ==> 6

#按列求和

tf.reduce_sum(x, 0) ==> [2, 2, 2]

#按行求和

tf.reduce_sum(x, 1) ==> [3, 3]

#按照行的维度求和

tf.reduce_sum(x, 1, keep_dims=True) ==> [[3], [3]]

#行列求和

tf.reduce_sum(x, [0, 1]) ==> 6
"""