import tensorflow as tf
# 为显示图片
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import pylab
# %pylab inline
# 为数据操作
import pandas as pd
import numpy as np
# 精度3位
np.set_printoptions(precision=3)
# 用于显示数据
def display(alist, show = True):
    print('type:%s\nshape: %s' %(alist[0].dtype,alist[0].shape))
    if show:
        for i in range(3):
            print('样本%s\n%s' %(i,alist[i]))

scalars = np.array([1,2,3],dtype=np.int64)#
print('\n标量')
display(scalars)

vectors = np.array([[0.1,0.1,0.1],
                   [0.2,0.2,0.2],
                   [0.3,0.3,0.3]],dtype=np.float32)
print('\n向量')
display(vectors)

matrices = np.array([np.array((vectors[0],vectors[0])),
                    np.array((vectors[1],vectors[1])),
                    np.array((vectors[2],vectors[2]))],dtype=np.float32)
print('\n矩阵')
display(matrices)

# shape of image：(806,806,3)
img=mpimg.imread('E:\CODEING\codeingForSelfStudy\CTR_CVR_DEEPLEARNING\esmm_public_Guidebook\TensorflowBasic\TfDataDatasetInputPipline\data\YJango.jpg') # 我的头像
tensors = np.array([img,img,img])
# show image
print('\n张量')
display(tensors, show = False)
plt.imshow(img)
pylab.show()

print("test -1 ")
print(matrices[1].reshape(-1))
#[0.2 0.2 0.2 0.2 0.2 0.2]# reshape(-1))其实是将2x3的向量falt成一行，我们不知道有多少列
print("test tensor ")
print(tensors[0].tostring())
"""
二、写入TFRecord file


1. 打开TFRecord file
writer = tf.python_io.TFRecordWriter('%s.tfrecord' %'test')
2. 创建样本写入字典
这里准备一个样本一个样本的写入TFRecord file中。
先把每个样本中所有feature的信息和值存到字典中，key为feature名，value为feature值。
feature值需要转变成tensorflow指定的feature类型中的一个：
2.1. 存储类型
•	int64：tf.train.Feature(int64_list = tf.train.Int64List(value=输入))
•	float32：tf.train.Feature(float_list = tf.train.FloatList(value=输入))
•	string：tf.train.Feature(bytes_list=tf.train.BytesList(value=输入))
•	注：输入必须是list(向量)
2.2. 如何处理类型是张量的feature
tensorflow feature类型只接受list数据，但如果数据类型是矩阵或者张量该如何处理？
两种方式：
•	转成list类型：将张量fatten成list(也就是向量)，再用写入list的方式写入。
•	转成string类型：将张量用.tostring()转换成string类型，再用tf.train.Feature(bytes_list=tf.train.BytesList(value=[input.tostring()]))来存储。
•	形状信息：不管那种方式都会使数据丢失形状信息，所以在向该样本中写入feature时应该额外加入shape信息作为额外feature。
shape信息是int类型，这里我是用原feature名字+'_shape'来指定shape信息的feature名。
"""
writer = tf.python_io.TFRecordWriter('%s.tfrecord' %'test')
# 这里我们将会写3个样本，每个样本里有4个feature：标量，向量，矩阵，张量
for i in range(3):
  # 创建字典
  features = {}
  # 写入标量，类型Int64，由于是标量，所以"value=[scalars[i]]" 变成list
  features['scalar'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[scalars[i]]))

  # 写入向量，类型float，本身就是list，所以"value=vectors[i]"没有中括号
  features['vector'] = tf.train.Feature(float_list=tf.train.FloatList(value=vectors[i]))

  # 写入矩阵，类型float，本身是矩阵，一种方法是将矩阵flatten成list
  features['matrix'] = tf.train.Feature(float_list=tf.train.FloatList(value=matrices[i].reshape(-1)))#[0.2 0.2 0.2 0.2 0.2 0.2]
  # 然而矩阵的形状信息(2,3)会丢失，需要存储形状信息，随后可转回原形状
  features['matrix_shape'] = tf.train.Feature(int64_list=tf.train.Int64List(value=matrices[i].shape))

  # 写入张量，类型float，本身是三维张量，另一种方法是转变成字符类型存储，随后再转回原类型
  features['tensor'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[tensors[i].tostring()]))
  # 存储丢失的形状信息(806,806,3)
  features['tensor_shape'] = tf.train.Feature(int64_list=tf.train.Int64List(value=tensors[i].shape))



  # 将存有所有feature的字典送入tf.train.Features中
  tf_features = tf.train.Features(feature= features)

  # 再将其变成一个样本example
  tf_example = tf.train.Example(features = tf_features)
  # 序列化该样本
  tf_serialized = tf_example.SerializeToString()

  # 写入一个序列化的样本
  writer.write(tf_serialized)
  # 由于上面有循环3次，所以到此我们已经写了3个样本
# 关闭文件
writer.close()

"""
从TFRecord文件导入准换为Dataset
"""


# 从多个tfrecord文件中导入数据到Dataset类 （这里用两个一样）
filenames = ["E:\CODEING\codeingForSelfStudy\CTR_CVR_DEEPLEARNING\esmm_public_Guidebook\TensorflowBasic\TfDataDatasetInputPipline\\test.tfrecord", "E:\CODEING\codeingForSelfStudy\CTR_CVR_DEEPLEARNING\esmm_public_Guidebook\TensorflowBasic\TfDataDatasetInputPipline\\test.tfrecord"]
# filenames = ["test.tfrecord"]
print(scalars[0].dtype)
print(vectors[0].dtype)
print(matrices[0].dtype)
print( tensors[0].dtype)

print(scalars[0].shape)
print(matrices[0].shape)
print(tensors[0].shape)
print((len(matrices[0].shape),))
print((len(tensors[0].shape),))
# int64
# float32
# float32
# uint8
# ()
# (2, 3)
# (200, 200, 3)
# (2,)
# (3,)
dataset = tf.data.TFRecordDataset(filenames)
data_info = pd.DataFrame({'name':['scalar','vector','matrix','matrix_shape','tensor','tensor_shape'],
                         # 'type':[scalars[0].dtype,vectors[0].dtype,matrices[0].dtype,tf.int64, tensors[0].dtype,tf.int64],
                         # 'shape':[scalars[0].shape,(3,),matrices[0].shape,(len(matrices[0].shape),),tensors[0].shape,(len(tensors[0].shape),)],
                          'type':[tf.int64,tf.float32,tf.float32,tf.int64, tf.uint8,tf.int64],
'shape':[(),(3,),(2,3),(2,),(200,200,3),(3,)],
                         'isbyte':[False,False,True,False,False,False],
                         'length_type':['fixed','fixed','var','fixed','fixed','fixed']},
                         columns=['name','type','shape','isbyte','length_type','default'])
print(">>>>>>>>>>>>>>>>>>>>>>>data_info<<<<<<<<<<<<<<<<<<<<<<<<<")
print(data_info)


def parse_function(example_proto):
    # 只接受一个输入：example_proto，也就是序列化后的样本tf_serialized
    dics = {  # 这里没用default_value，随后的都是None
        'scalar': tf.FixedLenFeature(shape=(), dtype=tf.int64, default_value=None),

        # vector的shape刻意从原本的(3,)指定成(1,3)
        'vector': tf.FixedLenFeature(shape=(1, 3), dtype=tf.float32),

        # 使用 VarLenFeature来解析
        'matrix': tf.VarLenFeature(dtype=np.dtype('float32')),
        'matrix_shape': tf.FixedLenFeature(shape=(2,), dtype=tf.int64),

        # tensor在写入时 使用了toString()，shape是()
        # 但这里的type不是tensor的原type，而是字符化后所用的tf.string，随后再回转成原tf.uint8类型
        'tensor': tf.FixedLenFeature(shape=(), dtype=tf.string),
        'tensor_shape': tf.FixedLenFeature(shape=(3,), dtype=tf.int64)}
# 把序列化样本和解析字典送入函数里得到解析的样本
    parsed_example = tf.parse_single_example(example_proto, dics)
# 解码字符
    parsed_example['tensor'] = tf.decode_raw(parsed_example['tensor'], tf.uint8)
    # 稀疏表示 转为 密集表示
    parsed_example['matrix'] = tf.sparse_tensor_to_dense(parsed_example['matrix'])
    # 转变matrix形状
    parsed_example['matrix'] = tf.reshape(parsed_example['matrix'], parsed_example['matrix_shape'])

    # 转变tensor形状
    parsed_example['tensor'] = tf.reshape(parsed_example['tensor'], parsed_example['tensor_shape'])
# 返回所有feature
    return parsed_example


new_dataset = dataset.map(parse_function)

# 创建获取数据集中样本的迭代器
iterator = new_dataset.make_one_shot_iterator()


# 获得下一个样本
next_element = iterator.get_next()
# 创建Session
sess = tf.InteractiveSession()

# 获取
i = 1
while True:
    # 不断的获得下一个样本
    try:
        # 获得的值直接属于graph的一部分，所以不再需要用feed_dict来喂
        scalar,vector,matrix,tensor = sess.run([next_element['scalar'],
                                                next_element['vector'],
                                                next_element['matrix'],
                                                next_element['tensor']])
    # 如果遍历完了数据集，则返回错误
    except tf.errors.OutOfRangeError:
        print("End of dataset")
        break
    else:
        # 显示每个样本中的所有feature的信息，只显示scalar的值
        print('==============example %s ==============' %i)
        print('scalar: value: %s | shape: %s | type: %s' %(scalar, scalar.shape, scalar.dtype))
        print('vector shape: %s | type: %s' %(vector.shape, vector.dtype))
        print('matrix shape: %s | type: %s' %(matrix.shape, matrix.dtype))
        print('tensor shape: %s | type: %s' %(tensor.shape, tensor.dtype))
    i+=1
    plt.imshow(tensor)
    pylab.show()

shuffle_dataset = new_dataset.shuffle(buffer_size=10000)
iterator = shuffle_dataset.make_one_shot_iterator()
next_element = iterator.get_next()

i = 1
while True:
    try:
        scalar = sess.run(next_element['scalar'])
    except tf.errors.OutOfRangeError:
        print("End of dataset")
        break
    else:
        print('example %s | scalar: value: %s' %(i,scalar))
    i+=1


batch_dataset = shuffle_dataset.batch(4)
iterator = batch_dataset.make_one_shot_iterator()
next_element = iterator.get_next()

i = 1
while True:
    # 不断的获得下一个样本
    try:
        scalar = sess.run(next_element['scalar'])
    except tf.errors.OutOfRangeError:
        print("End of dataset")
        break
    else:
        print('example %s | scalar: value: %s' %(i,scalar))
    i+=1



batch_padding_dataset = new_dataset.padded_batch(4,
                        padded_shapes={'scalar': [],
                                       'vector': [-1,5],
                                       'matrix': [None,None],
                                       'matrix_shape': [None],
                                       'tensor': [None,None,None],
                                       'tensor_shape': [None]})
iterator = batch_padding_dataset.make_one_shot_iterator()
next_element = iterator.get_next()

i = 1
while True:
    try:
        scalar,vector,matrix,tensor = sess.run([next_element['scalar'],
                                                next_element['vector'],
                                                next_element['matrix'],
                                                next_element['tensor']])
    except tf.errors.OutOfRangeError:
        print("End of dataset")
        break
    else:
        print('==============example %s ==============' %i)
        print('scalar: value: %s | shape: %s | type: %s' %(scalar, scalar.shape, scalar.dtype))
        print('padded vector value\n%s:\nvector shape: %s | type: %s' %(vector, vector.shape, vector.dtype))
        print('matrix shape: %s | type: %s' %(matrix.shape, matrix.dtype))
        print('tensor shape: %s | type: %s' %(tensor.shape, tensor.dtype))
    i+=1
