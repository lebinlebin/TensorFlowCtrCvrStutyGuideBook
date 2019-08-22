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



"""
FNN
FNN的考虑是模型的capacity可以进一步提升，以对更复杂的场景建模。
FNN可以视作FM + MLP = LR + MF + MLP
"""


class FNN(Model):
    def __init__(self, field_sizes=None, embed_size=10, layer_sizes=None, layer_acts=None, drop_out=None,
                 embed_l2=None, layer_l2=None, init_path=None, opt_algo='gd', learning_rate=1e-2, random_seed=None):
        Model.__init__(self)
        init_vars = []
        num_inputs = len(field_sizes)
        # 每个field都要初始化一下
        for i in range(num_inputs):
            init_vars.append(('embed_%d' % i, [field_sizes[i], embed_size], 'xavier', dtype))
        node_in = num_inputs * embed_size
        for i in range(len(layer_sizes)):
            init_vars.append(('w%d' % i, [node_in, layer_sizes[i]], 'xavier', dtype))
            init_vars.append(('b%d' % i, [layer_sizes[i]], 'zero', dtype))
            node_in = layer_sizes[i]
        self.graph = tf.Graph()
        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            self.X = [tf.sparse_placeholder(dtype) for i in range(num_inputs)]
            self.y = tf.placeholder(dtype)
            # 训练阶段会加上dropout  test阶段不会用dropput
            self.keep_prob_train = 1 - np.array(drop_out)
            self.keep_prob_test = np.ones_like(drop_out)
            self.layer_keeps = tf.placeholder(dtype)
            self.vars = init_var_map(init_vars, init_path)
            w0 = [self.vars['embed_%d' % i] for i in range(num_inputs)]
            xw = tf.concat([tf.sparse_tensor_dense_matmul(self.X[i], w0[i]) for i in range(num_inputs)], 1)
            l = xw

            for i in range(len(layer_sizes)):
                wi = self.vars['w%d' % i]
                bi = self.vars['b%d' % i]
                print(l.shape, wi.shape, bi.shape)
                l = tf.nn.dropout(
                    activate(
                        tf.matmul(l, wi) + bi,
                        layer_acts[i]),
                    self.layer_keeps[i])

            l = tf.squeeze(l)
            self.y_prob = tf.sigmoid(l)

            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=l, labels=self.y))
            if layer_l2 is not None:
                self.loss += embed_l2 * tf.nn.l2_loss(xw)
                for i in range(len(layer_sizes)):
                    wi = self.vars['w%d' % i]
                    self.loss += layer_l2[i] * tf.nn.l2_loss(wi)
            self.optimizer = get_optimizer(opt_algo, learning_rate, self.loss)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            tf.global_variables_initializer().run(session=self.sess)





import numpy as np
from sklearn.metrics import roc_auc_score
import progressbar

train_file = './data/train.txt'
test_file = './data/test.txt'

input_dim = INPUT_DIM
# train_data = pkl.load(open('./data/train.pkl', 'rb'))
# test_data = pkl.load(open('./data/test.pkl', 'rb'))
train_data = read_data(train_file)
test_data = read_data(test_file)
train_data = shuffle(train_data)



if train_data[1].ndim > 1:
    print('label must be 1-dim')
    exit(0)
print('read finish')
print('train data size:', train_data[0].shape)
print('test data size:', test_data[0].shape)

train_size = train_data[0].shape[0]
test_size = test_data[0].shape[0]
num_feas = len(FIELD_SIZES)

min_round = 1
num_round = 200
early_stop_round = 5
batch_size = 1024

field_sizes = FIELD_SIZES
field_offsets = FIELD_OFFSETS

train_data = split_data(train_data)
test_data = split_data(test_data)
tmp = []
for x in field_sizes:
    if x > 0:
        tmp.append(x)
field_sizes = tmp
print('remove empty fields', field_sizes)

fnn_params = {
    'field_sizes': field_sizes,
    'embed_size': 10,
    'layer_sizes': [500, 1],
    'layer_acts': ['relu', None],
    'drop_out': [0, 0],
    'opt_algo': 'gd',
    'learning_rate': 0.1,
    'embed_l2': 0,
    'layer_l2': [0, 0],
    'random_seed': 0
}
print(fnn_params)
model = FNN(**fnn_params)


def train(model):
    history_score = []
    for i in range(num_round):
        fetches = [model.optimizer, model.loss]
        if batch_size > 0:
            ls = []
            bar = progressbar.ProgressBar()
            print('[%d]\ttraining...' % i)
            for j in bar(range(int(train_size / batch_size + 1))):
                X_i, y_i = slice(train_data, j * batch_size, batch_size)
                _, l = model.run(fetches, X_i, y_i)
                ls.append(l)
        elif batch_size == -1:
            X_i, y_i = slice(train_data)
            _, l = model.run(fetches, X_i, y_i)
            ls = [l]
        train_preds = []
        print('[%d]\tevaluating...' % i)
        bar = progressbar.ProgressBar()
        for j in bar(range(int(train_size / 10000 + 1))):
            X_i, _ = slice(train_data, j * 10000, 10000)
            preds = model.run(model.y_prob, X_i, mode='test')
            train_preds.extend(preds)
        test_preds = []
        bar = progressbar.ProgressBar()
        for j in bar(range(int(test_size / 10000 + 1))):
            X_i, _ = slice(test_data, j * 10000, 10000)
            preds = model.run(model.y_prob, X_i, mode='test')
            test_preds.extend(preds)
        train_score = roc_auc_score(train_data[1], train_preds)
        test_score = roc_auc_score(test_data[1], test_preds)
        print('[%d]\tloss (with l2 norm):%f\ttrain-auc: %f\teval-auc: %f' % (i, np.mean(ls), train_score, test_score))
        history_score.append(test_score)
        if i > min_round and i > early_stop_round:
            if np.argmax(history_score) == i - early_stop_round and history_score[-1] - history_score[
                -1 * early_stop_round] < 1e-5:
                print('early stop\nbest iteration:\n[%d]\teval-auc: %f' % (
                    np.argmax(history_score), np.max(history_score)))
                break


train(model)