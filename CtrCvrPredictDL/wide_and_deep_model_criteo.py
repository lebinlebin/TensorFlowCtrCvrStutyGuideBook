# coding: utf-8
# 宽度深度模型/wide and deep model
# 在之前的代码里大家看到了如何用tensorflow自带的op来构建灵活的神经网络，这里用tf中的高级接口，用更简单的方式完成wide&deep模型。


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)
print("Using TensorFlow version %s\n" % (tf.__version__))

# 我们这里使用的是criteo数据集，X的部分包括13个连续值列和26个类别型值的列
CONTINUOUS_COLUMNS =  ["I"+str(i) for i in range(1,14)] # 1-13 inclusive
CATEGORICAL_COLUMNS = ["C"+str(i) for i in range(1,27)] # 1-26 inclusive
# 标签是clicked
LABEL_COLUMN = ["clicked"]

# 训练集由 label列 + 连续值列 + 离散值列 构成
TRAIN_DATA_COLUMNS = LABEL_COLUMN + CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS
#TEST_DATA_COLUMNS = CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS

# 特征列就是 连续值列+离散值列
FEATURE_COLUMNS = CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS

# 输出一些信息
print('Feature columns are: ', FEATURE_COLUMNS, '\n')

# 数据示例
sample = [ 0, 127, 1, 3, 1683, 19, 26, 17, 475, 0, 9, 0, 3,
           "05db9164", "8947f767", "11c9d79e", "52a787c8", "4cf72387", "fbad5c96", "18671b18", "0b153874",
           "a73ee510", "ceb10289", "77212bd7", "79507c6b", "7203f04e", "07d13a8f", "2c14c412", "49013ffe",
           "8efede7f", "bd17c3da", "f6a3e43b", "a458ea53", "35cd95c9", "ad3062eb", "c7dc6720", "3fdb382b", "010f6491", "49d68486"]

print('Columns and data as a dict: ', dict(zip(FEATURE_COLUMNS, sample)), '\n')





"""
输入文件解析
我们把数据送进`Reader`然后从文件里一次读一个batch
对`_input_fn()`函数做了特殊的封装处理，使得它更适合不同类型的文件读取
注意一下：这里的文件是直接通过tensorflow读取的，我们没有用pandas这种工具，也没有一次性把所有数据读入内存，
这样对于非常大规模的数据文件训练，是合理的。

关于input_fn函数
这个函数定义了我们怎么读取数据用于训练和测试。这里的返回结果是一个pair对，第一个元素是列名到具体取值的映射字典，第二个元素是label的序列。
抽象一下，大概是这么个东西 `map(column_name => [Tensor of values]) , [Tensor of labels])`
举个例子就长这样：
    { 
      'age':            [ 39, 50, 38, 53, 28, … ], 
      'marital_status': [ 'Married-civ-spouse', 'Never-married', 'Widowed', 'Widowed' … ],
       ...
      'gender':           ['Male', 'Female', 'Male', 'Male', 'Female',, … ], 
    } , 
    [ 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1]
"""


# ### High-level structure of input functions for CSV-style data
# 1. Queue file(s)
# 2. Read a batch of data from the next file
# 3. Create record defaults, generally 0 for continuous values, and "" for categorical. You can use named types if you prefer
# 4. Decode the CSV and restructure it to be appropriate for the graph's input format
#     * `zip()` column headers with the data
#     * `pop()` off the label column(s)
#     * Remove/pop any unneeded column(s)
#     * Run `tf.expand_dims()` on categorical columns
#     5. Return the pair: `(feature_dict, label_array)`


BATCH_SIZE = 2000

def generate_input_fn(filename, batch_size=BATCH_SIZE):
    def _input_fn():
        filename_queue = tf.train.string_input_producer([filename])
        reader = tf.TextLineReader()
        # 只读batch_size行
        key, value = reader.read_up_to(filename_queue, num_records=batch_size)
        
        # 1个int型的label, 13个连续值, 26个字符串类型
        cont_defaults = [ [0] for i in range(1,14) ]
        cate_defaults = [ [" "] for i in range(1,27) ]
        label_defaults = [ [0] ]
        column_headers = TRAIN_DATA_COLUMNS
        
        # 第一列数据是label
        record_defaults = label_defaults + cont_defaults + cate_defaults
        # 解析读出的csv数据
        # 我们要手动把数据和header去zip在一起
        columns = tf.decode_csv(value, record_defaults=record_defaults)
        
        # 最终是列名到数据张量的映射字典
        all_columns = dict(zip(column_headers, columns))
        
        # 弹出和保存label标签
        labels = all_columns.pop(LABEL_COLUMN[0])
        
        # 其余列就是特征
        features = all_columns 

        #todo 类别型的列我们要做一个类似one-hot的扩展操作
        for feature_name in CATEGORICAL_COLUMNS:
            features[feature_name] = tf.expand_dims(features[feature_name], -1)

        return features, labels

    return _input_fn

print('input function configured')



"""
特征工程：
构建特征列
这个部分我们来看一下用tensorflow的高级接口，如何方便地对特征进行处理
稀疏列/Sparse Columns
我们先构建稀疏列(针对类别型)
对于所有类别取值都清楚的我们用`sparse_column_with_keys()`处理
对于类别可能比较多，没办法枚举的可以试试用`sparse_column_with_hash_bucket()`处理这个映射
"""

# Sparse base columns.
# C1 = tf.contrib.layers.sparse_column_with_hash_bucket('C1', hash_bucket_size=1000)
# C2 = tf.contrib.layers.sparse_column_with_hash_bucket('C2', hash_bucket_size=1000)
# C3 = tf.contrib.layers.sparse_column_with_hash_bucket('C3', hash_bucket_size=1000)
# ...
# Cn = tf.contrib.layers.sparse_column_with_hash_bucket('Cn', hash_bucket_size=1000)
# wide_columns = [C1, C2, C3, ... , Cn]

wide_columns = []
for name in CATEGORICAL_COLUMNS:
    wide_columns.append(tf.contrib.layers.sparse_column_with_hash_bucket(name, hash_bucket_size=1000))

print('Wide/Sparse columns configured')


# #### 连续值列/Continuous columns
# 通过`real_valued_column()`设定连续值列


# Continuous base columns.
# I1 = tf.contrib.layers.real_valued_column("I1")
# I2 = tf.contrib.layers.real_valued_column("I2")
# I3 = tf.contrib.layers.real_valued_column("I3")
# ...
# In = tf.contrib.layers.real_valued_column("In")
# deep_columns = [I1, I2, I3, ... , In]

deep_columns = []
for name in CONTINUOUS_COLUMNS:
    deep_columns.append(tf.contrib.layers.real_valued_column(name))

print('deep/continuous columns configured')



"""
特征交叉
因为这是一份做过脱敏处理的数据，所以我们做下面的2个操作
分桶/bucketizing 对连续值离散化和分桶
生成交叉特征/feature crossing  对2列或者多列去构建交叉组合特征(注意只有离散的特征才能交叉，所以如果连续值特征要用这个处理，要先离散化)
"""
# age_buckets = tf.contrib.layers.bucketized_column(age,
#             boundaries=[ 18, 25, 30, 35, 40, 45, 50, 55, 60, 65 ])
# education_occupation = tf.contrib.layers.crossed_column([education, occupation],
#                                                         hash_bucket_size=int(1e4))
# age_race_occupation = tf.contrib.layers.crossed_column([age_buckets, race, occupation],
#                                                        hash_bucket_size=int(1e6))
# country_occupation = tf.contrib.layers.crossed_column([native_country, occupation],
#                                                       hash_bucket_size=int(1e4))

print('Transformations complete')


# ### Group feature columns into 2 objects
"""
The wide columns are the sparse, categorical columns that we specified, as well as our hashed, bucket, and feature crossed columns. 
The deep columns are composed of embedded categorical columns along with the continuous real-valued columns. 
**Column embeddings** transform a sparse, categorical tensor into a low-dimensional and dense real-valued vector. 
The embedding values are also trained along with the rest of the model. For more information about embeddings, 
see the TensorFlow tutorial on [Vector Representations Words](https://www.tensorflow.org/tutorials/word2vec/), 
or [Word Embedding](https://en.wikipedia.org/wiki/Word_embedding) on Wikipedia.
The higher the dimension of the embedding is, the more degrees of freedom the model will have to learn the representations of the features. 
We are starting with an 8-dimension embedding for simplicity, but later you can come back and increase the dimensionality if you wish.
"""



# Wide columns and deep columns.
# wide_columns = [gender, race, native_country,
#       education, occupation, workclass,
#       marital_status, relationship,
#       age_buckets, education_occupation,
#       age_race_occupation, country_occupation]

# deep_columns = [
#   tf.contrib.layers.embedding_column(workclass, dimension=8),
#   tf.contrib.layers.embedding_column(education, dimension=8),
#   tf.contrib.layers.embedding_column(marital_status, dimension=8),
#   tf.contrib.layers.embedding_column(gender, dimension=8),
#   tf.contrib.layers.embedding_column(relationship, dimension=8),
#   tf.contrib.layers.embedding_column(race, dimension=8),
#   tf.contrib.layers.embedding_column(native_country, dimension=8),
#   tf.contrib.layers.embedding_column(occupation, dimension=8),
#   age,
#   education_num,
#   capital_gain,
#   capital_loss,
#   hours_per_week,
# ]

# Embeddings for wide columns into deep columns
for col in wide_columns:
    deep_columns.append(tf.contrib.layers.embedding_column(col,dimension=8))
print('wide and deep columns configured')

"""
构建模型
你可以根据实际情况构建“宽模型”、“深模型”、“深度宽度模型”
* **Wide**: 相当于逻辑回归
* **Deep**: 相当于多层感知器
* **Wide & Deep**: 组合两种结构
这里有2个参数`hidden_units` 或者 `dnn_hidden_units`可以指定隐层的节点个数，比如`[12, 20, 15]`构建3层神经元个数分别为12、20、15的隐层。
"""

def create_model_dir(model_type):
    # 返回类似这样的结果 models/model_WIDE_AND_DEEP_1493043407
    return './models/model_' + model_type + '_' + str(int(time.time()))

# 指定模型文件夹
def get_model(model_type, model_dir):
    print("Model directory = %s" % model_dir)
    
    # 对checkpoint去做设定
    runconfig = tf.contrib.learn.RunConfig(
        save_checkpoints_secs=None,
        save_checkpoints_steps = 100,
    )
    
    m = None
    
    # 宽模型
    if model_type == 'WIDE':
        m = tf.contrib.learn.LinearClassifier(
            model_dir=model_dir, 
            feature_columns=wide_columns)

    # 深度模型
    if model_type == 'DEEP':
        m = tf.contrib.learn.DNNClassifier(
            model_dir=model_dir,
            feature_columns=deep_columns,
            hidden_units=[100, 50, 25])

    # 宽度深度模型
    if model_type == 'WIDE_AND_DEEP':
        m = tf.contrib.learn.DNNLinearCombinedClassifier(
            model_dir=model_dir,
            linear_feature_columns=wide_columns,
            dnn_feature_columns=deep_columns,
            dnn_hidden_units=[100, 70, 50, 25],
            config=runconfig)
        
    print('estimator built')
    
    return m
    

MODEL_TYPE = 'WIDE_AND_DEEP'
model_dir = create_model_dir(model_type=MODEL_TYPE)
m = get_model(model_type=MODEL_TYPE, model_dir=model_dir)

# 评估
from tensorflow.contrib.learn.python.learn import evaluable
isinstance(m, evaluable.Evaluable)

"""
拟合与模型训练
执行`fit()`函数训练模型，可以试试不同的`train_steps`和`BATCH_SIZE`参数，会影响速度和结果
训练文件与测试文件
"""

train_file = "./data/criteo_data/criteo_train.txt"
eval_file  = "./data/criteo_data/criteo_test.txt"

# This can be found with
# wc -l train.csv
train_sample_size = 2000000
train_steps = train_sample_size/BATCH_SIZE*20

m.fit(input_fn=generate_input_fn(train_file, BATCH_SIZE), steps=train_steps)

print('fit done')


"""
评估模型准确率
"""

eval_sample_size = 500000 # this can be found with a 'wc -l eval.csv'
eval_steps = eval_sample_size/BATCH_SIZE

results = m.evaluate(input_fn=generate_input_fn(eval_file),steps=eval_steps)
print('evaluate done')

print('Accuracy: %s' % results['accuracy'])
print(results)

"""
进行预估
"""

def pred_fn():
    sample = [ 0, 127, 1, 3, 1683, 19, 26, 17, 475, 0, 9, 0, 3,
               "05db9164", "8947f767", "11c9d79e", "52a787c8", "4cf72387", "fbad5c96", "18671b18", "0b153874",
               "a73ee510", "ceb10289", "77212bd7", "79507c6b", "7203f04e", "07d13a8f", "2c14c412", "49013ffe",
               "8efede7f", "bd17c3da", "f6a3e43b", "a458ea53", "35cd95c9", "ad3062eb", "c7dc6720", "3fdb382b",
               "010f6491", "49d68486"]
    sample_dict = dict(zip(FEATURE_COLUMNS, sample))

    #OneHot
    for feature_name in CATEGORICAL_COLUMNS:
        sample_dict[feature_name] = tf.expand_dims(sample_dict[feature_name], -1)
        
    for feature_name in CONTINUOUS_COLUMNS:
        sample_dict[feature_name] = tf.constant(sample_dict[feature_name], dtype=tf.int32)
    print(sample_dict)

    return sample_dict

m.predict(input_fn=pred_fn)