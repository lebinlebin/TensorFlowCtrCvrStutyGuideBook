from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import _pickle as pkl
import numpy as np
from sklearn.metrics import roc_auc_score
# import progressbar
import tensorflow as tf
from scipy.sparse import coo_matrix

# 读取数据，统计基本的信息，field等
DTYPE = tf.float32
FIELD_SIZES = [0] * 26  # 新建一个长度为26的数组，初始化为0
"""
:other	0
0:0	1
0:1	2
0:2	3
0:3	4
0:4	5
0:5	6
0:6	7
1:other	8
1:00	9
1:01	10
1:02	11
1:03	12
1:04	13
1:05	14
"""
with open('./data/featindex.txt') as fin:
    for line in fin:
        line = line.strip().split(':')
        if len(line) > 1:
            f = int(line[0]) - 1
            FIELD_SIZES[f] += 1
print('field sizes:', FIELD_SIZES)
FIELD_OFFSETS = [sum(FIELD_SIZES[:i]) for i in range(len(FIELD_SIZES))]
INPUT_DIM = sum(FIELD_SIZES)
OUTPUT_DIM = 1
STDDEV = 1e-3
MINVAL = -1e-3
MAXVAL = 1e-3


# 读取libsvm格式数据成稀疏矩阵形式
# 0 5:1 9:1 140858:1 445908:1 446177:1 446293:1 449140:1 490778:1 491626:1 491634:1 491641:1 491645:1 491648:1 491668:1 491700:1 491708:1
def read_data(file_name):
    X = []
    D = []
    y = []
    with open(file_name) as fin:
        for line in fin:
            fields = line.strip().split()
            # print(len(fields))  输出17
            y_i = int(fields[0])
            X_i = [int(x.split(':')[0]) for x in fields[1:]]
            D_i = [int(x.split(':')[1]) for x in fields[1:]]
            y.append(y_i)
            X.append(X_i)
            D.append(D_i)
    y = np.reshape(np.array(y), [-1])
    X = libsvm_2_coo(zip(X, D), (len(X), INPUT_DIM)).tocsr()
    return X, y

# 数据乱序
def shuffle(data):
    X, y = data
    ind = np.arange(X.shape[0])
    for i in range(7):
        np.random.shuffle(ind)
    return X[ind], y[ind]

# 工具函数，libsvm格式转成coo稀疏存储格式
def libsvm_2_coo(libsvm_data, shape):
    coo_rows = []
    coo_cols = []
    coo_data = []
    n = 0
    for x, d in libsvm_data:
        coo_rows.extend([n] * len(x))
        coo_cols.extend(x)
        coo_data.extend(d)
        n += 1
    coo_rows = np.array(coo_rows)
    coo_cols = np.array(coo_cols)
    coo_data = np.array(coo_data)
    return coo_matrix((coo_data, (coo_rows, coo_cols)), shape=shape)


#  csr转成输入格式
def csr_2_input(csr_mat):
    if not isinstance(csr_mat, list):
        coo_mat = csr_mat.tocoo()
        indices = np.vstack((coo_mat.row, coo_mat.col)).transpose()
        values = csr_mat.data
        shape = csr_mat.shape
        return indices, values, shape
    else:
        inputs = []
        for csr_i in csr_mat:
            inputs.append(csr_2_input(csr_i))
        return inputs

# 数据切片
def slice(csr_data, start=0, size=-1):
    if not isinstance(csr_data[0], list):
        if size == -1 or start + size >= csr_data[0].shape[0]:
            slc_data = csr_data[0][start:]
            slc_labels = csr_data[1][start:]
        else:
            slc_data = csr_data[0][start:start + size]
            slc_labels = csr_data[1][start:start + size]
    else:
        if size == -1 or start + size >= csr_data[0][0].shape[0]:
            slc_data = []
            for d_i in csr_data[0]:
                slc_data.append(d_i[start:])
            slc_labels = csr_data[1][start:]
        else:
            slc_data = []
            for d_i in csr_data[0]:
                slc_data.append(d_i[start:start + size])
            slc_labels = csr_data[1][start:start + size]
    return csr_2_input(slc_data), slc_labels



# 数据切分
def split_data(data, skip_empty=True):
    fields = []
    for i in range(len(FIELD_OFFSETS) - 1):
        start_ind = FIELD_OFFSETS[i]
        end_ind = FIELD_OFFSETS[i + 1]
        if skip_empty and start_ind == end_ind:
            continue
        field_i = data[0][:, start_ind:end_ind]
        fields.append(field_i)
    fields.append(data[0][:, FIELD_OFFSETS[-1]:])
    return fields, data[1]



# 在tensorflow中初始化各种参数变量
def init_var_map(init_vars, init_path=None):
    if init_path is not None:
        load_var_map = pkl.load(open(init_path, 'rb'))
        print('load variable map from', init_path, load_var_map.keys())
    var_map = {}
    for var_name, var_shape, init_method, dtype in init_vars:
        if init_method == 'zero':
            var_map[var_name] = tf.Variable(tf.zeros(var_shape, dtype=dtype), name=var_name, dtype=dtype)
        elif init_method == 'one':
            var_map[var_name] = tf.Variable(tf.ones(var_shape, dtype=dtype), name=var_name, dtype=dtype)
        elif init_method == 'normal':
            var_map[var_name] = tf.Variable(tf.random_normal(var_shape, mean=0.0, stddev=STDDEV, dtype=dtype),
                                            name=var_name, dtype=dtype)
        elif init_method == 'tnormal':
            var_map[var_name] = tf.Variable(tf.truncated_normal(var_shape, mean=0.0, stddev=STDDEV, dtype=dtype),
                                            name=var_name, dtype=dtype)
        elif init_method == 'uniform':
            var_map[var_name] = tf.Variable(tf.random_uniform(var_shape, minval=MINVAL, maxval=MAXVAL, dtype=dtype),
                                            name=var_name, dtype=dtype)
        elif init_method == 'xavier':
            maxval = np.sqrt(6. / np.sum(var_shape))
            minval = -maxval
            var_map[var_name] = tf.Variable(tf.random_uniform(var_shape, minval=minval, maxval=maxval, dtype=dtype),
                                            name=var_name, dtype=dtype)
        elif isinstance(init_method, int) or isinstance(init_method, float):
            var_map[var_name] = tf.Variable(tf.ones(var_shape, dtype=dtype) * init_method, name=var_name, dtype=dtype)
        elif init_method in load_var_map:
            if load_var_map[init_method].shape == tuple(var_shape):
                var_map[var_name] = tf.Variable(load_var_map[init_method], name=var_name, dtype=dtype)
            else:
                print('BadParam: init method', init_method, 'shape', var_shape, load_var_map[init_method].shape)
        else:
            print('BadParam: init method', init_method)
    return var_map

# 不同的激活函数选择
def activate(weights, activation_function):
    if activation_function == 'sigmoid':
        return tf.nn.sigmoid(weights)
    elif activation_function == 'softmax':
        return tf.nn.softmax(weights)
    elif activation_function == 'relu':
        return tf.nn.relu(weights)
    elif activation_function == 'tanh':
        return tf.nn.tanh(weights)
    elif activation_function == 'elu':
        return tf.nn.elu(weights)
    elif activation_function == 'none':
        return weights
    else:
        return weights

# 不同的优化器选择
def get_optimizer(opt_algo, learning_rate, loss):
    if opt_algo == 'adaldeta':
        return tf.train.AdadeltaOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'adagrad':
        return tf.train.AdagradOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'adam':
        return tf.train.AdamOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'ftrl':
        return tf.train.FtrlOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'gd':
        return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'padagrad':
        return tf.train.ProximalAdagradOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'pgd':
        return tf.train.ProximalGradientDescentOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'rmsprop':
        return tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
    else:
        return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)




# 工具函数
# 提示：tf.slice(input_, begin, size, name=None)：按照指定的下标范围抽取连续区域的子集
#   tf.gather(params, indices, validate_indices=None, name=None)：按照指定的下标集合从axis=0中抽取子集，适合抽取不连续区域的子集
def gather_2d(params, indices):
    shape = tf.shape(params)
    flat = tf.reshape(params, [-1])
    flat_idx = indices[:, 0] * shape[1] + indices[:, 1]
    flat_idx = tf.reshape(flat_idx, [-1])
    return tf.gather(flat, flat_idx)

def gather_3d(params, indices):
    shape = tf.shape(params)
    flat = tf.reshape(params, [-1])
    flat_idx = indices[:, 0] * shape[1] * shape[2] + indices[:, 1] * shape[2] + indices[:, 2]
    flat_idx = tf.reshape(flat_idx, [-1])
    return tf.gather(flat, flat_idx)


def gather_4d(params, indices):
    shape = tf.shape(params)
    flat = tf.reshape(params, [-1])
    flat_idx = indices[:, 0] * shape[1] * shape[2] * shape[3] + indices[:, 1] * shape[2] * shape[3] + indices[:, 2] * \
               shape[3] + indices[:, 3]
    flat_idx = tf.reshape(flat_idx, [-1])
    return tf.gather(flat, flat_idx)


# 池化2d
def max_pool_2d(params, k):
    _, indices = tf.nn.top_k(params, k, sorted=False)
    shape = tf.shape(indices)
    r1 = tf.reshape(tf.range(shape[0]), [-1, 1])
    r1 = tf.tile(r1, [1, k])
    r1 = tf.reshape(r1, [-1, 1])
    indices = tf.concat([r1, tf.reshape(indices, [-1, 1])], 1)
    return tf.reshape(gather_2d(params, indices), [-1, k])



# 池化3d
def max_pool_3d(params, k):
    _, indices = tf.nn.top_k(params, k, sorted=False)
    shape = tf.shape(indices)
    r1 = tf.reshape(tf.range(shape[0]), [-1, 1])
    r2 = tf.reshape(tf.range(shape[1]), [-1, 1])
    r1 = tf.tile(r1, [1, k * shape[1]])
    r2 = tf.tile(r2, [1, k])
    r1 = tf.reshape(r1, [-1, 1])
    r2 = tf.tile(tf.reshape(r2, [-1, 1]), [shape[0], 1])
    indices = tf.concat([r1, r2, tf.reshape(indices, [-1, 1])], 1)
    return tf.reshape(gather_3d(params, indices), [-1, shape[1], k])


# 池化4d
def max_pool_4d(params, k):
    _, indices = tf.nn.top_k(params, k, sorted=False)
    shape = tf.shape(indices)
    r1 = tf.reshape(tf.range(shape[0]), [-1, 1])
    r2 = tf.reshape(tf.range(shape[1]), [-1, 1])
    r3 = tf.reshape(tf.range(shape[2]), [-1, 1])
    r1 = tf.tile(r1, [1, shape[1] * shape[2] * k])
    r2 = tf.tile(r2, [1, shape[2] * k])
    r3 = tf.tile(r3, [1, k])
    r1 = tf.reshape(r1, [-1, 1])
    r2 = tf.tile(tf.reshape(r2, [-1, 1]), [shape[0], 1])
    r3 = tf.tile(tf.reshape(r3, [-1, 1]), [shape[0] * shape[1], 1])
    indices = tf.concat([r1, r2, r3, tf.reshape(indices, [-1, 1])], 1)
    return tf.reshape(gather_4d(params, indices), [-1, shape[1], shape[2], k])

#定义不同的模型
# 定义基类模型
dtype = DTYPE

class Model:
    def __init__(self):
        self.sess = None
        self.X = None
        self.y = None
        self.layer_keeps = None
        self.vars = None
        self.keep_prob_train = None
        self.keep_prob_test = None

    # run model
    def run(self, fetches, X=None, y=None, mode='train'):
        # 通过feed_dict传入数据
        feed_dict = {}
        if type(self.X) is list:
            for i in range(len(X)):
                feed_dict[self.X[i]] = X[i]
        else:
            feed_dict[self.X] = X
        if y is not None:
            feed_dict[self.y] = y
        if self.layer_keeps is not None:
            if mode == 'train':
                feed_dict[self.layer_keeps] = self.keep_prob_train
            elif mode == 'test':
                feed_dict[self.layer_keeps] = self.keep_prob_test
        # 通过session.run去执行op
        return self.sess.run(fetches, feed_dict)

    # 模型参数持久化
    def dump(self, model_path):
        var_map = {}
        for name, var in self.vars.iteritems():
            var_map[name] = self.run(var)
        pkl.dump(var_map, open(model_path, 'wb'))
        print('model dumped at', model_path)


# ### CCPM
# FM只能学习特征的二阶组合，但CNN能学习更高阶的组合，可学习的阶数和卷积的视野相关。
# embedding层：e1, e2…en是某特定用户被展示的一系列广告。如果在预测广告是否会点击时不考虑历史展示广告的点击情况，则n=1。同时embedding矩阵的具体值是随着模型训练学出来的。Embedding矩阵为S，向量维度为d。
# 卷积层：卷积参数W有d*w个，即对于矩阵S，上图每一列对应一个参数不共享的一维卷积，其视野为w，卷积共有d个，每个输出向量维度为(n+w-1)，输出矩阵维度d*(n+w-1)。因为对于ctr预估而言，矩阵S每一列都对应特定的描述维度，所以需要分别处理，得到的输出矩阵的每一列就都是描述广告特定方面的特征。
# Pooling层：flexible p-max pooling。
# L是模型总卷积层数，n是输入序列长度，pi就是第i层的pooling参数。这样最后一层卷积层都是输出3个最大的元素，长度固定方便后面接全连接层。同时这个指数型的参数，一开始改变比较小，几乎都是n，后面就减少得比较快。这样可以防止在模型浅层的时候就损失太多信息，众所周知深度模型在前面几层最好不要做得太简单，容易损失很多信息。文章还提到p-max pooling输出的几个最大的元素是保序的，可输入时的顺序一致，这点对于保留序列信息是重要的。
# 激活层：tanh
# 最后，
# Fij是指低i层的第j个feature map。感觉是不同输入通道的卷积参数也不共享，对应输出是所有输入通道卷积的输出的求和。

class CCPM(Model):
    def __init__(self, field_sizes=None, embed_size=10, filter_sizes=None, layer_acts=None, drop_out=None,
                 init_path=None, opt_algo='gd', learning_rate=1e-2, random_seed=None):
        Model.__init__(self)
        init_vars = []
        num_inputs = len(field_sizes)
        for i in range(num_inputs):
            init_vars.append(('embed_%d' % i, [field_sizes[i], embed_size], 'xavier', dtype))
        init_vars.append(('f1', [embed_size, filter_sizes[0], 1, 2], 'xavier', dtype))
        init_vars.append(('f2', [embed_size, filter_sizes[1], 2, 2], 'xavier', dtype))
        init_vars.append(('w1', [2 * 3 * embed_size, 1], 'xavier', dtype))
        init_vars.append(('b1', [1], 'zero', dtype))

        self.graph = tf.Graph()
        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            self.X = [tf.sparse_placeholder(dtype) for i in range(num_inputs)]
            self.y = tf.placeholder(dtype)
            self.keep_prob_train = 1 - np.array(drop_out)
            self.keep_prob_test = np.ones_like(drop_out)
            self.layer_keeps = tf.placeholder(dtype)
            self.vars = init_var_map(init_vars, init_path)
            w0 = [self.vars['embed_%d' % i] for i in range(num_inputs)]
            xw = tf.concat([tf.sparse_tensor_dense_matmul(self.X[i], w0[i]) for i in range(num_inputs)], 1)
            l = xw

            l = tf.transpose(tf.reshape(l, [-1, num_inputs, embed_size, 1]), [0, 2, 1, 3])
            f1 = self.vars['f1']
            l = tf.nn.conv2d(l, f1, [1, 1, 1, 1], 'SAME')
            l = tf.transpose(
                max_pool_4d(
                    tf.transpose(l, [0, 1, 3, 2]),
                    int(num_inputs / 2)),
                [0, 1, 3, 2])
            f2 = self.vars['f2']
            l = tf.nn.conv2d(l, f2, [1, 1, 1, 1], 'SAME')
            l = tf.transpose(
                max_pool_4d(
                    tf.transpose(l, [0, 1, 3, 2]), 3),
                [0, 1, 3, 2])
            l = tf.nn.dropout(
                activate(
                    tf.reshape(l, [-1, embed_size * 3 * 2]),
                    layer_acts[0]),
                self.layer_keeps[0])
            w1 = self.vars['w1']
            b1 = self.vars['b1']
            l = tf.matmul(l, w1) + b1

            l = tf.squeeze(l)
            self.y_prob = tf.sigmoid(l)

            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=l, labels=self.y))
            self.optimizer = get_optimizer(opt_algo, learning_rate, self.loss)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            tf.global_variables_initializer().run(session=self.sess)