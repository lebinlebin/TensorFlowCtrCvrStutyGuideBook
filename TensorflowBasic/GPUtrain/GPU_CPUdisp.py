"""
显示本机GPU地址
"""

import os
from tensorflow.python.client import device_lib

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "99"

if __name__ == "__main__":
    print(device_lib.list_local_devices())


# import tensorflow as tf
# idx_0 = tf.placeholder(tf.int64, [None])
# mask = tf.one_hot(idx_0, 3, axis=-1)
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# a = sess.run([mask],feed_dict={idx_0:[0,1,2]})
# print(a)

"""
https://stackoverflow.com/questions/41115476/tensorflow-gpu-cuda-error-launch-failed-on-tf-one-hot
以上代码报错如下：
#Error polling for event status: failed to query event: CUDA_ERROR_ILLEGAL_INSTRUCTION
#tensorflow\stream_executor\cuda\cuda_driver.cc:1110] could not synchronize on CUDA context: CUDA_ERROR_ILLEGAL_INSTRUCTION :: 
#common_runtime\gpu\gpu_event_mgr.cc:208] Unexpected Event status: 1

原因：
Config of the PC:
TensorFlow 0.12.0-rc1
Python 3.5
CUDA 8.0
cuDNN 5.1
OS: Windows 10
GPU: GeForce GTX 970
tf.one_hot run ok when running on Linux CPU, Linux GPU (GeForce GTX 660), Windows 10 CPU. Not ok on the Windows 10 GPU.
On the Windows 10 GPU, tf.matmul, tf.reduce_mean, tf.reduce_sum are run ok. But tf.one_hot is not ok.
Is that a bug, or I miss something? Thanks.
(Edit 2016-12-16)
I have run the code on the same machine, in Xubuntu, GPU. The code run fine. So I think that is a problem in TensorFlow-Windows.
"""
# with tf.device('/cpu:0'):
#     b = tf.one_hot(a, 123)


# import tensorflow as tf
# idx_0 = tf.placeholder(tf.int64, [None])
# # mask = tf.one_hot(idx_0, 3, axis=-1)
# with tf.device('/cpu:0'):
#     mask = tf.one_hot(idx_0, 3, axis=-1)
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# a = sess.run([mask],feed_dict={idx_0:[0,1,2]})
# print(a)

"""
输出正确结果：
[array([[1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.]], dtype=float32)]
"""


