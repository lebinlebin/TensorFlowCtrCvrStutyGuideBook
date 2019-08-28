import tensorflow as tf
from tensorflow import feature_column
from tensorflow.python.feature_column.feature_column import _LazyBuilder

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


def shared_embedding_column_with_hash_bucket():
    color_data = {'color': [[2, 2], [5, 5], [0, -1], [0, 0]],  # 4行样本 shape=[4,2]
                  'color2': [[2], [5], [-1], [0]]}  # 4行样本  shape=[4,1]
    builder = _LazyBuilder(color_data)

    # categorical_column1
    color_column = feature_column.categorical_column_with_hash_bucket('color', 7, dtype=tf.int32)
    print(color_column)
    # tensor1
    color_column_tensor = color_column._get_sparse_tensors(builder)

    # categorical_column2
    color_column2 = feature_column.categorical_column_with_hash_bucket('color2', 7, dtype=tf.int32)
    print(color_column2)

    # tensor2
    color_column_tensor2 = color_column2._get_sparse_tensors(builder)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print('categorical_column_with_hash_bucket' + '_' * 40)
        print(session.run([color_column_tensor.id_tensor]))
        print(session.run([color_column_tensor2.id_tensor]))
    print('not use input_layer' + '_' * 40)


    color_column_embed = feature_column.shared_embedding_columns([color_column2, color_column], 3, combiner='sum')
    print(type(color_column_embed))
    print((color_column_embed))

    color_dense_tensor = feature_column.input_layer(color_data, color_column_embed)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print('shared_embedding_columns' + '_' * 40)
        print(session.run(color_dense_tensor))


shared_embedding_column_with_hash_bucket()
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
    # shared_embedding_columns
    # 这里是每个共享column中每个column映射为指定维度2的embedding向量 因此这里会输出一个 1x2的embedding向量
[[-1.3173009   0.645887   -2.6346018   1.291774  ]
 [ 0.0483053   0.1325178   0.09661061  0.2650356 ]
 [ 0.          0.          0.20036496  0.6643266 ]
 [ 0.20036496  0.6643266   0.40072992  1.3286532 ]]
    从输出可以看出，输出的每一行以此代表[color_column2, color_column]，
    即[-1.3173009   0.645887   -2.6346018   1.291774  ]中的前两列-1.3173009   0.645887 表示源数据[[2], [5], [-1], [0]]中的2，
    后两列-2.6346018   1.291774 表示源数据[[2, 2], [5, 5], [0, -1], [0, 0]]中的 [2, 2],这里由于combiner='sum'
    因此[2, 2]的embedding向量是  [2]的embedding向量的两倍。  
    -1.3173009x2 = -2.6346018
    0.645887x2 = 1.291774
"""