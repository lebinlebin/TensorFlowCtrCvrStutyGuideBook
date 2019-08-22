# coding: utf-8
"""
构建分布式Tensorflow模型系列四:特征工程
特征工程是机器学习流程中重要的一个环节，即使是通常用来做端到端学习的深度学习模型在训练之前也免不了要做一些特征工程相关的工作。
Tensorflow平台提供的FeatureColumn API为特征工程提供了强大的支持。
Feature cloumns是原始数据和Estimator模型之间的桥梁，它们被用来把各种形式的原始数据转换为模型能够使用的格式。
深度神经网络只能处理数值数据，网络中的每个神经元节点执行一些针对输入数据和网络权重的乘法和加法运算。
然而，现实中的有很多非数值的类别数据，比如产品的品牌、类目等，这些数据如果不加转换，神经网络是无法处理的。
另一方面，即使是数值数据，在扔给网络进行训练之前有时也需要做一些处理，比如标准化、离散化等。

"""
# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt
# import json
# import tensorflow as tf
# from tensorflow import feature_column as fc

# Represent a tf.float64 scalar.
# numeric_feature_column=tf.feature_column.numeric_column(key="SepalLength", dtype=tf.float64)

# Represent a 10-element vector in which each cell contains a tf.float32.
# vector_feature_column = tf.feature_column.numeric_column(key="Bowling", shape=10)
# Represent a 10x5 matrix in which each cell contains a tf.float32.
# matrix_feature_column = tf.feature_column.numeric_column(key="MyMatrix", shape=[10,5])

import tensorflow as tf
from tensorflow import feature_column
from tensorflow.python.feature_column.feature_column import _LazyBuilder

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

"""
我们还可以为numeric column指定数值变换的函数normalizer_fn，为对原始数据做一些变换操作。
"""
def test_numeric():
    price = {'price': [[1.], [2.], [3.], [4.]]}  # 4行样本 shape = [4,1]
    builder = _LazyBuilder(price)
    def transform_fn(x):
        return x + 2
    price_column = feature_column.numeric_column('price', normalizer_fn=transform_fn)
    price_transformed_tensor = price_column._get_dense_tensor(builder)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as session:
        print(session.run([price_transformed_tensor]))
    # 使用input_layer
    price_transformed_tensor = feature_column.input_layer(price, [price_column])
    with tf.Session() as session:
        print('use input_layer' + '_' * 40)
        print(session.run([price_transformed_tensor])) #run后边接的是一个数组

test_numeric()
"""
[array([[3.],
       [4.],
       [5.],
       [6.]], dtype=float32)]
use input_layer________________________________________
[array([[3.],
       [4.],
       [5.],
       [6.]], dtype=float32)]
"""

"""
Bucketized column
Bucketized column用来把numeric column的值按照提供的边界（boundaries)离散化为多个值。
离散化是特征工程常用的一种方法。例如，把年份离散化为4个阶段
"""
print(">>>>>>>>>>>>>>>>>>>>>>>>>> Bucketized column <<<<<<<<<<<<<<<<<<<<<<<<<<<<")

# First, convert the raw input to a numeric column.
numeric_feature_column = tf.feature_column.numeric_column("Year")
# Then, bucketize the numeric column on the years 1960, 1980, and 2000.
bucketized_feature_column = tf.feature_column.bucketized_column(
    source_column = numeric_feature_column,
    boundaries = [1960, 1980, 2000]
)


def test_bucketized_column():
    price = {'price': [[5.], [15.], [25.], [35.]]}  # 4行样本 shape =[4,1]
    price_column = feature_column.numeric_column('price')
    bucket_price = feature_column.bucketized_column(price_column, [10, 20, 30, 40])
    price_bucket_tensor = feature_column.input_layer(price, [bucket_price])
    with tf.Session() as session:
        print(session.run([price_bucket_tensor]))
test_bucketized_column()
"""
>>>>>>>>>>>>>>>>>>>>>>>>>> Bucketized column <<<<<<<<<<<<<<<<<<<<<<<<<<<<
[array([[1., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0.],
       [0., 0., 1., 0., 0.],
       [0., 0., 0., 1., 0.]], dtype=float32)]
"""

"""
与Bucketized column类似，Categorical identity column用单个唯一值表示bucket。
顾名思义，Categorical vocabulary column把一个vocabulary中的string映射为数值型的类别特征，是做one-hot编码的很好的方法。
在tensorflow中有两种提供词汇表的方法，一种是用list，另一种是用file，对应的feature column分别为：
•	tf.feature_column.categorical_column_with_vocabulary_list
•	tf.feature_column.categorical_column_with_vocabulary_file
两者的定义如下：
"""

# Create categorical output for an integer feature named "my_feature_b",
# The values of my_feature_b must be >= 0 and < num_buckets
identity_feature_column = tf.feature_column.categorical_column_with_identity(
    key='my_feature_b',
    num_buckets=4) # Values [0, 4)





print(">>>>>>>>>>>>>>>>>>>>>>>>>>categorical_column_with_vocabulary_list && indicator_column <<<<<<<<<<<<<<<<<<<<<<<<<<<<")
# Given input "feature_name_from_input_fn" which is a string,
# create a categorical feature by mapping the input to one of
# the elements in the vocabulary list.
vocabulary_feature_column = \
	tf.feature_column.categorical_column_with_vocabulary_list(
        key="string",
        vocabulary_list=["kitchenware", "electronics", "sports"] )


def test_categorical_column_with_vocabulary_list():
    # 源数据
    color_data = {'color': [['R', 'R'], ['G', 'R'], ['B', 'G'], ['A', 'A']]}  # 4行样本 shape = [4,2]
    builder = _LazyBuilder(color_data)

    # 获取源数据的categorical_column
    color_column = feature_column.categorical_column_with_vocabulary_list(
        key= 'color',
        vocabulary_list=['R', 'G', 'B'],
        dtype=tf.string,
        default_value=-1
    )

    #源数据 转化为tensor 稀疏表示
    color_column_tensor = color_column._get_sparse_tensors(builder)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print(session.run([color_column_tensor.id_tensor]))

    # 通过indicator_column，将稀疏的转换成dense，也就是one-hot形式，只是multi-hot
    color_column_identy = feature_column.indicator_column(color_column)
    color_dense_tensor = feature_column.input_layer(color_data, [color_column_identy])
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print('use input_layer' + '_' * 40)
        print(session.run([color_dense_tensor]))

test_categorical_column_with_vocabulary_list()

"""
数据源：
['R', 'R'],
['G', 'R'], 
['B', 'G'], 
['A', 'A']
>>>>>>>>>>>>>>>>>>>>>>>>>> test_categorical_column_with_vocabulary_list <<<<<<<<<<<<<<<<<<<<<<<<<<<<
数据的稀疏表示：
[SparseTensorValue(indices=array(
      [[0, 0], #第0行0列
       [0, 1],
       [1, 0],
       [1, 1],
       [2, 0],
       [2, 1],
       [3, 0],
       [3, 1]], dtype=int64), values=array([ 0,  0,  1,  0,  2,  1, -1, -1], dtype=int64), dense_shape=array([4, 2], dtype=int64))]
use input_layer________________________________________
[array([[2., 0., 0.],
       [1., 1., 0.],
       [0., 1., 1.],
       [0., 0., 0.]], dtype=float32)]
"""
"""
indicator_column解析：
def indicator_column(categorical_column):
  Represents multi-hot representation of given categorical column.
  Used to wrap any `categorical_column_*` (e.g., to feed to DNN). Use `embedding_column` if the inputs are sparse.
  ```python
  name = indicator_column( categorical_column_with_vocabulary_list(
      'name', ['bob', 'george', 'wanda'])
  columns = [name, ...]
  features = tf.parse_example(..., features=make_parse_example_spec(columns))
  dense_tensor = input_layer(features, columns)

  dense_tensor == [[1, 0, 0]]  # If "name" bytes_list is ["bob"]
  dense_tensor == [[1, 0, 1]]  # If "name" bytes_list is ["bob", "wanda"]
  dense_tensor == [[2, 0, 0]]  # If "name" bytes_list is ["bob", "bob"]
  ```
  Args:
    categorical_column: A `_CategoricalColumn` which is created by  `categorical_column_with_*` or `crossed_column` functions.
  Returns:
    An `_IndicatorColumn`.
"""



"""
Hashed Column
为类别特征提供词汇表有时候会过于繁琐，特别是在词汇表非常大的时候，词汇表会非常消耗内存。
tf.feature_column.categorical_column_with_hash_bucket 允许用户指定类别的总数，通过hash的方式来得到最终的类别ID。
# pseudocode
feature_id = hash(raw_feature) % hash_buckets_size
用hash的方式产生类别ID，不可避免地会遇到hash冲突的问题，即可有多多个原来不相同的类别会产生相同的类别ID。
因此，设置hash_bucket_size参数会显得比较重要。实践表明，hash冲突不会对神经网络模型造成太大的影响，
因为模型可以通过其他特征作进一步区分。
需要注意的是，使用hash bucket的时候，原始值中-1或者空字符串""会被忽略，不会输出结果。
"""



print(">>>>>>>>>>>>>>>>>>>>>>>>>> test_categorical_column_with_hash_bucket <<<<<<<<<<<<<<<<<<<<<<<<<<<<")
def test_categorical_column_with_hash_bucket():
    #源数据
    color_data = {'color': [[2], [5], [-1], [0]]}  # 4行样本 shape=[4,1]
    builder = _LazyBuilder(color_data)

    # categorical_column
    color_column = feature_column.categorical_column_with_hash_bucket('color', 7, dtype=tf.int32)

    # tensor
    color_column_tensor = color_column._get_sparse_tensors(builder)#稀疏表示
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print(session.run([color_column_tensor.id_tensor]))


    # 通过indicator_column，将稀疏的转换成dense，也就是one-hot形式，只是multi-hot
    color_column_identy = feature_column.indicator_column(color_column)

    #input_layer连接数据源和声明的column生成新的tensor
    color_dense_tensor = feature_column.input_layer(color_data, [color_column_identy])

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print('use input_layer' + '_' * 40)
        print(session.run([color_dense_tensor]))

test_categorical_column_with_hash_bucket()

"""
>>>>>>>>>>>>>>>>>>>>>>>>>> test_categorical_column_with_hash_bucket <<<<<<<<<<<<<<<<<<<<<<<<<<<<
[SparseTensorValue(indices=array(
      [[0, 0],
       [1, 0],
       [3, 0]], dtype=int64), values=array([5, 1, 2], dtype=int64), dense_shape=array([4, 1], dtype=int64))]
备注:
1. 映射逻辑：feature_id = hash(raw_feature) % hash_buckets_size ； categorical_column_with_hash_bucket将原始数据映射为 0--6的数字
2. trick： 使用hash bucket的时候，原始值中-1或者空字符串""会被忽略，不会输出结果。
0行  hash(2)%7 = 5
1行  hash(5)%7 = 1
2行  -1 返回 空
3行  hash(0)%7 = 2

use input_layer________________________________________
[array([[0., 0., 0., 0., 0., 1., 0.],
       [0., 1., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 1., 0., 0., 0., 0.]], dtype=float32)]
"""






"""
Crossed column
交叉组合特征也是一种很常用的特征工程手段，尤其是在使用LR模型时。Crossed column仅仅适用于sparser特征，产生的依然是sparsor特征。
tf.feature_column.crossed_column(
    keys,
    hash_bucket_size,
    hash_key=None
)
具体地，Crossed 特征对keys 的笛卡尔积执行hash操作，再把hash的结果对hash_bucket_size取模得到最终的结果：
Hash(cartesian product of features) % hash_bucket_size。
"""
print(">>>>>>>>>>>>>>>>>>>>>>>>>> Crossed column <<<<<<<<<<<<<<<<<<<<<<<<<<<<")
def test_crossed_column():
    """ crossed column测试 """
    #源数据
    featrues = {
        'price': [['A'], ['B'], ['C']], # 0,1,2
        'color': [['R'], ['G'], ['B']]  # 0,1,2
    }
    # categorical_column
    price = feature_column.categorical_column_with_vocabulary_list('price', ['A', 'B', 'C', 'D'])
    color = feature_column.categorical_column_with_vocabulary_list('color', ['R', 'G', 'B'])

    #crossed_column 产生稀疏表示
    p_x_c = feature_column.crossed_column([price, color], 16)

    # 稠密表示
    p_x_c_identy = feature_column.indicator_column(p_x_c)

    # crossed_column 连接 源数据
    p_x_c_identy_dense_tensor = feature_column.input_layer(featrues, [p_x_c_identy])

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print(session.run([p_x_c_identy_dense_tensor]))
test_crossed_column()

"""
>>>>>>>>>>>>>>>>>>>>>>>>>> Crossed column <<<<<<<<<<<<<<<<<<<<<<<<<<<<
[array([[0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]],
      dtype=float32)]

"""

"""
embedding columns
Indicator columns 和 embedding columns 不能直接作用在原始特征上，而是作用在categorical columns上。
在前面的众多例子中，我们已经使用过indicator_column来把categorical column得到的稀疏tensor转换为one-hot或者multi-hot形式的稠密tensor，这里就不赘述了。
当某些特征的类别数量非常大时，使用indicator_column来把原始数据转换为神经网络的输入就变得非常不灵活，
这时通常使用embedding column把原始特征映射为一个低维稠密的实数向量。同一类别的embedding向量间的距离通常可以用来度量类别直接的相似性。
Embedding column与indicator column之间的区别：
Embedding column会将数据映射到一个固定的相对降维的数据向量上不如说3维向量；
而indicator column是multihot的，即按照那个最大种类作为维度，有值的取为1,没有值的取为0
tf.feature_column.embedding_column(
    categorical_column,
    dimension,
    combiner='mean',
    initializer=None,
    ckpt_to_load_from=None,
    tensor_name_in_ckpt=None,
    max_norm=None,
    trainable=True
)

•	categorical_column: 使用categoryical_column产生的sparsor column
•	dimension: 定义embedding的维数
•	combiner: 对于多个entries进行的推导。默认是meam, 但是 sqrtn在词袋模型中，有更好的准确度。
•	initializer: 初始化方法，默认使用高斯分布来初始化。
•	tensor_name_in_ckpt: 可以从check point中恢复
•	ckpt_to_load_from: check point file，这是在 tensor_name_in_ckpt 不为空的情况下设置的.
•	max_norm: 默认是l2
•	trainable: 是否可训练的，默认是true
"""
print(">>>>>>>>>>>>>>>>>>>>>>>>>> test_embedding <<<<<<<<<<<<<<<<<<<<<<<<<<<<")
def test_embedding():
    tf.set_random_seed(1)
    #源数据
    color_data = {'color': [['R', 'G'], ['G', 'A'], ['B', 'B'], ['A', 'A']]}  # 4行样本
    builder = _LazyBuilder(color_data)

    # categorical_column  要想转为 embedding 先将源数据的clomn表达为categorical_column 这里只是声明没有源数据
    color_column = feature_column.categorical_column_with_vocabulary_list(
        'color', ['R', 'G', 'B'], dtype=tf.string, default_value=-1
    )
    # tensor 数据源  将数据源表达成tensor
    color_column_tensor = color_column._get_sparse_tensors(builder)

    #获取embedding_column； 第一个参数是：categorical_column；  第二个参数是维度
    color_embedding_column = feature_column.embedding_column(color_column, 4, combiner='sum')


    # 转化为tensor  input_layer(数据源，column)  连接起数据源和embedding_column
    color_embeding_dense_tensor = feature_column.input_layer(color_data, [color_embedding_column])

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print(session.run([color_column_tensor.id_tensor]))
        print('embeding' + '_' * 40)
        print(session.run([color_embeding_dense_tensor]))

test_embedding()

"""
>>>>>>>>>>>>>>>>>>>>>>>>>> test_embedding <<<<<<<<<<<<<<<<<<<<<<<<<<<<
数据源输入：
['R', 'G']
['G', 'A']
['B', 'B']
['A', 'A']

输出：
[SparseTensorValue(indices=array(
      [[0, 0], #R->0
       [0, 1], #G->1
       [1, 0], #G->1
       [1, 1], #A->-1 默认值
       [2, 0], #B->2
       [2, 1], #B->2 
       [3, 0], #A->-1 
       [3, 1]] #A->-1       , dtype=int64), values=array([ 0,  1,  1, -1,  2,  2, -1, -1], dtype=int64), dense_shape=array([4, 2], dtype=int64))]
embeding________________________________________
[array([[-0.33141667, -0.194623  ,  0.13752429,  0.48839447],
       [-0.33676255, -0.3219394 , -0.14066543, -0.00224128],
       [-0.3653014 ,  0.92616415,  1.4802293 , -1.6142133 ],
       [ 0.        ,  0.        ,  0.        ,  0.        ]],
      dtype=float32)]
备注：
从上面的测试结果可以看出不在vocabulary里的数据'A'在经过categorical_column_with_vocabulary_list操作时映射为默认值-1,
而默认值-1在embeding column时映射为0向量,这是一个很有用的特性,
可以用-1来填充一个不定长的ID序列,这样可以得到定长的序列,
然后经过embedding column之后,填充的-1值不影响原来的结果。
"""

print(">>>>>>>>>>>>>>>>>>>>>>>>>> test_shared_embedding_column_with_hash_bucket <<<<<<<<<<<<<<<<<<<<<<<<<<<<")
"""
shared_embedding_columns
有时候在同一个网络模型中，有多个特征可能需要共享相同的embeding映射空间，比如用户历史行为序列中的商品ID和候选商品ID，这时候可以用到
tf.feature_column.shared_embedding_columns。

tf.feature_column.shared_embedding_columns(
    categorical_columns,
    dimension,
    combiner='mean',
    initializer=None,
    shared_embedding_collection_name=None,
    ckpt_to_load_from=None,
    tensor_name_in_ckpt=None,
    max_norm=None,
    trainable=True
)
•	categorical_columns 为需要共享embeding映射空间的类别特征列表
•	其他参数与embedding column类似
"""


def test_shared_embedding_column_with_hash_bucket():
    color_data = {'color': [[2, 2], [5, 5], [0, -1], [0, 0]], # 4行样本 shape=[4,2]
                  'color2': [[2], [5], [-1], [0]]}  # 4行样本  shape=[4,1]
    builder = _LazyBuilder(color_data)

    #categorical_column1
    color_column = feature_column.categorical_column_with_hash_bucket('color', 7, dtype=tf.int32)

    # tensor1
    color_column_tensor = color_column._get_sparse_tensors(builder)

    #categorical_column2
    color_column2 = feature_column.categorical_column_with_hash_bucket('color2', 7, dtype=tf.int32)
    # tensor2
    color_column_tensor2 = color_column2._get_sparse_tensors(builder)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print('not use input_layer' + '_' * 40)
        print(session.run([color_column_tensor.id_tensor]))
        print(session.run([color_column_tensor2.id_tensor]))
    print('not use input_layer' + '_' * 40)

	# shared_embedding_columns
    color_column_embed = feature_column.shared_embedding_columns([color_column2, color_column], 3, combiner='sum')
    print(type(color_column_embed))
    print((color_column_embed))

    color_dense_tensor = feature_column.input_layer(color_data, color_column_embed)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print('use input_layer' + '_' * 40)
        print(session.run(color_dense_tensor))

test_shared_embedding_column_with_hash_bucket()
"""
需要注意的是，tf.feature_column.shared_embedding_columns的返回值是一个与参数categorical_columns维数相同的列表。
"""
"""
>>>>>>>>>>>>>>>>>>>>>>>>>> test_shared_embedding_column_with_hash_bucket <<<<<<<<<<<<<<<<<<<<<<<<<<<<
输入：
[2, 2], 
[5, 5], 
[0, -1], 
[0, 0]

not use input_layer________________________________________
[SparseTensorValue(indices=array(
      [[0, 0], 2  hash(0-6) -> 5
       [0, 1], 2  hash(0-6) -> 5
       [1, 0], 5  hash(0-6) -> 1
       [1, 1], 5  hash(0-6) -> 1
       [2, 0], 0  hash(0-6) -> 2
       [3, 0], 0  hash(0-6) -> 2
       [3, 1]] 0  hash(0-6) -> 2  , dtype=int64), values=array([5, 5, 1, 1, 2, 2, 2], dtype=int64), dense_shape=array([4, 2], dtype=int64))]

输入：
[2], 
[5], 
[-1], 
[0]
输出：
[SparseTensorValue(indices=array(
      [[0, 0],  2  hash(0-6) -> 5
       [1, 0],  5  hash(0-6) -> 1
       [3, 0]], 0  hash(0-6) -> 2   dtype=int64), values=array([5, 1, 2], dtype=int64), dense_shape=array([4, 1], dtype=int64))]


>>>>>>>>>>>>>>>>>>>>>>>>>> test_shared_embedding_column_with_hash_bucket <<<<<<<<<<<<<<<<<<<<<<<<<<<<
not use input_layer________________________________________
<class 'list'>
[_SharedEmbeddingColumn(categorical_column=_HashedCategoricalColumn(key='color2', hash_bucket_size=7, dtype=tf.int32), 
dimension=3, combiner='sum', initializer=<tensorflow.python.ops.init_ops.TruncatedNormal object at 0x000001953EF33940>, 
shared_embedding_collection_name='color_color2_shared_embedding', ckpt_to_load_from=None, tensor_name_in_ckpt=None,
 max_norm=None, trainable=True), 
 _SharedEmbeddingColumn(categorical_column=_HashedCategoricalColumn(key='color', hash_bucket_size=7, dtype=tf.int32), 
 dimension=3, combiner='sum', initializer=<tensorflow.python.ops.init_ops.TruncatedNormal object at 0x000001953EF33940>, 
 shared_embedding_collection_name='color_color2_shared_embedding', ckpt_to_load_from=None, tensor_name_in_ckpt=None, 
 max_norm=None, trainable=True)]
 
use input_layer________________________________________

[[ 0.2906018   0.23723301 -0.40129796  0.5812036   0.47446603 -0.8025959 ]
 [-0.25720465  0.214194   -0.3002099  -0.5144093   0.428388   -0.6004198 ]
 [ 0.          0.          0.         -0.10063636  0.19022052  0.33866954]
 [-0.10063636  0.19022052  0.33866954 -0.20127271  0.38044104  0.6773391 ]]
 
 备注：
 ‘需要注意的是,tf.feature_column.shared_embedding_columns的返回值是一个与参数categorical_columns维数相同的列表。
"""

"""
crossed_column 和 shared_embedding_column需要在加强学习一下 
dimension是干什么用的？？？？
def shared_embedding_columns(
    categorical_columns, dimension, combiner='mean', initializer=None,
    shared_embedding_collection_name=None, ckpt_to_load_from=None,
    tensor_name_in_ckpt=None, max_norm=None, trainable=True):
    
List of dense columns that convert from sparse, categorical input.
  This is similar to `embedding_column`, except that that it produces a list of
  embedding columns that share the same embedding weights.

  Use this when your inputs are sparse and of the same type (e.g. watched and
  impression video IDs that share the same vocabulary), and you want to convert
  them to a dense representation (e.g., to feed to a DNN).

  Inputs must be a list of categorical columns created by any of the
  `categorical_column_*` function. They must all be of the same type and have
  the same arguments except `key`. E.g. they can be
  categorical_column_with_vocabulary_file with the same vocabulary_file. Some or
  all columns could also be weighted_categorical_column.

  Here is an example embedding of two features for a DNNClassifier model:

  python
  watched_video_id = categorical_column_with_vocabulary_file(
      'watched_video_id', video_vocabulary_file, video_vocabulary_size)
  impression_video_id = categorical_column_with_vocabulary_file(
      'impression_video_id', video_vocabulary_file, video_vocabulary_size)
  columns = shared_embedding_columns(
      [watched_video_id, impression_video_id], dimension=10)

  estimator = tf.estimator.DNNClassifier(feature_columns=columns, ...)

  label_column = ...
  def input_fn():
    features = tf.parse_example(
        ..., features=make_parse_example_spec(columns + [label_column]))
    labels = features.pop(label_column.name)
    return features, labels

  estimator.train(input_fn=input_fn, steps=100)
  

  Here is an example using `shared_embedding_columns` with model_fn:

  python
  def model_fn(features, ...):
    watched_video_id = categorical_column_with_vocabulary_file(
        'watched_video_id', video_vocabulary_file, video_vocabulary_size)
    impression_video_id = categorical_column_with_vocabulary_file(
        'impression_video_id', video_vocabulary_file, video_vocabulary_size)
    columns = shared_embedding_columns(
        [watched_video_id, impression_video_id], dimension=10)
    dense_tensor = input_layer(features, columns)
    # Form DNN layers, calculate loss, and return EstimatorSpec.
 

  Args:
    categorical_columns: List of categorical columns created by a
      `categorical_column_with_*` function. These columns produce the sparse IDs
      that are inputs to the embedding lookup. All columns must be of the same
      type and have the same arguments except `key`. E.g. they can be
      categorical_column_with_vocabulary_file with the same vocabulary_file.
      Some or all columns could also be weighted_categorical_column.
    dimension: An integer specifying dimension of the embedding, must be > 0.
    combiner: A string specifying how to reduce if there are multiple entries
      in a single row. Currently 'mean', 'sqrtn' and 'sum' are supported, with
      'mean' the default. 'sqrtn' often achieves good accuracy, in particular
      with bag-of-words columns. Each of this can be thought as example level
      normalizations on the column. For more information, see
      `tf.embedding_lookup_sparse`.
    initializer: A variable initializer function to be used in embedding
      variable initialization. If not specified, defaults to
      `tf.truncated_normal_initializer` with mean `0.0` and standard deviation
      `1/sqrt(dimension)`.
    shared_embedding_collection_name: Optional name of the collection where
      shared embedding weights are added. If not given, a reasonable name will
      be chosen based on the names of `categorical_columns`. This is also used
      in `variable_scope` when creating shared embedding weights.
    ckpt_to_load_from: String representing checkpoint name/pattern from which to
      restore column weights. Required if `tensor_name_in_ckpt` is not `None`.
    tensor_name_in_ckpt: Name of the `Tensor` in `ckpt_to_load_from` from
      which to restore the column weights. Required if `ckpt_to_load_from` is
      not `None`.
    max_norm: If not `None`, embedding values are l2-normalized to this value.
    trainable: Whether or not the embedding is trainable. Default is True.

  Returns:
    A list of dense columns that converts from sparse input. The order of
    results follows the ordering of `categorical_columns`.

  Raises:
    ValueError: if `dimension` not > 0.
    ValueError: if any of the given `categorical_columns` is of different type
      or has different arguments than the others.
    ValueError: if exactly one of `ckpt_to_load_from` and `tensor_name_in_ckpt`
      is specified.
    ValueError: if `initializer` is specified and is not callable.
"""


"""
Weighted categorical column
有时候我们需要给一个类别特征赋予一定的权重，比如给用户行为序列按照行为发生的时间到某个特定时间的差来计算不同的权重，
这是可以用到weighted_categorical_column。
tf.feature_column.weighted_categorical_column(
    categorical_column,
    weight_feature_key,
    dtype=tf.float32
)

可以看到，相对于前面其他categorical_column来说多了weight这个tensor。weighted_categorical_column的一个用例就是，
weighted_categorical_column的结果传入给shared_embedding_columns可以对ID序列的embeding向量做加权融合。
"""
print(">>>>>>>>>>>>>>>>>>>>>>>>>> test_weighted_categorical_column <<<<<<<<<<<<<<<<<<<<<<<<<<<<")
def test_weighted_categorical_column():
    #源数据
    color_data = {'color': [['R'], ['G'], ['B'], ['A']],
                  'weight': [[1.0], [2.0], [4.0], [8.0]]}  # 4行样本
    #categorical_column
    color_column = feature_column.categorical_column_with_vocabulary_list(
        'color', ['R', 'G', 'B'], dtype=tf.string, default_value=-1
    )

    # weighted_categorical_column
    color_weight_categorical_column = feature_column.weighted_categorical_column(color_column, 'weight')
    builder = _LazyBuilder(color_data)

    with tf.Session() as session:
        id_tensor, weight = color_weight_categorical_column._get_sparse_tensors(builder)
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print('weighted categorical' + '-' * 40)
        print(session.run([id_tensor]))
        print('-' * 40)
        print(session.run([weight]))
test_weighted_categorical_column()

"""
>>>>>>>>>>>>>>>>>>>>>>>>>> test_weighted_categorical_column <<<<<<<<<<<<<<<<<<<<<<<<<<<<
weighted categorical----------------------------------------
[SparseTensorValue(indices=array(
      [[0, 0],R->0
       [1, 0],G->1
       [2, 0],B->2
       [3, 0]]A->-1 , dtype=int64), values=array([ 0,  1,  2, -1], dtype=int64), dense_shape=array([4, 1], dtype=int64))]
----------------------------------------
[SparseTensorValue(indices=array(
      [[0, 0],
       [1, 0],
       [2, 0],
       [3, 0]], dtype=int64), values=array([1., 2., 4., 8.], dtype=float32), dense_shape=array([4, 1], dtype=int64))]
"""


'''
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
构建分布式Tensorflow模型系列之CVR预估案例ESMM模型
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
'''

from tensorflow import feature_column as fc
# user field 用户历史浏览过的产品ID列表 映射
pids = fc.categorical_column_with_hash_bucket("behaviorPids", 10240, dtype=tf.int64)
# item field   productId表示当前的候选产品ID映射
pid = fc.categorical_column_with_hash_bucket("productId", 1000000, dtype=tf.int64)
pid_embed = fc.shared_embedding_columns([pids, pid], 100, combiner='sum', shared_embedding_collection_name="pid")

"""
那么如何实现weighted sum pooling操作呢？答案就是使用weighted_categorical_column函数。我们必须在构建样本时添加一个额外的权重特征,
权重特征表示行为序列中每个产品的权重,因此权重特征是一个与行为序列平行的列表（向量）,两者的维度必须相同。
另外,如果行为序列中有填充的默认值-1,那么权重特征中这些默认值对应的权重必须为0。代码示例如下：
"""
from tensorflow import feature_column as fc
# user field
pids = fc.categorical_column_with_hash_bucket("behaviorPids", 10240, dtype=tf.int64)
pids_weighted = fc.weighted_categorical_column(pids, "pidWeights")
# item field
pid = fc.categorical_column_with_hash_bucket("productId", 1000000, dtype=tf.int64)
pid_embed = fc.shared_embedding_columns([pids_weighted, pid], 100, combiner='sum', shared_embedding_collection_name="pid")


"""
模型函数
Base模型的其他组件就不过多介绍了,模型函数的代码如下：
"""

def my_model(features, labels, mode, params):
  net = fc.input_layer(features, params['feature_columns'])
  # Build the hidden layers, sized according to the 'hidden_units' param.
  for units in params['hidden_units']:
    net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
    if 'dropout_rate' in params and params['dropout_rate'] > 0.0:
      net = tf.layers.dropout(net, params['dropout_rate'], training=(mode == tf.estimator.ModeKeys.TRAIN))
  my_head = tf.contrib.estimator.binary_classification_head(thresholds=[0.5])
  # Compute logits (1 per class).
  logits = tf.layers.dense(net, my_head.logits_dimension, activation=None, name="my_model_output_logits")
  optimizer = tf.train.AdagradOptimizer(learning_rate=params['learning_rate'])

  def _train_op_fn(loss):
    return optimizer.minimize(loss, global_step=tf.train.get_global_step())

  return my_head.create_estimator_spec(
    features=features,
    mode=mode,
    labels=labels,
    logits=logits,
    train_op_fn=_train_op_fn
  )








