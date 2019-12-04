import tensorflow as tf
"""
自定义一个估算器（分类器）Estimator的完整流程
结构概览
从表面看，我们的Estimator应该具有DNNClassifier一样的功能
•	创建的时候接收一些参数，如feature_columns、hidden_units、n_classes等
•	具有train()、evaluate()、predict()三个方法用来训练、评价、预测
如上所说，我们使用 tf.estimator.Estimator()方法来生成自定义Estimator，它的语法格式是
tf.estimator.Estimator(
    model_fn, #模型函数
    model_dir=None, #存储目录
    config=None, #设置参数对象
    params=None, #超参数，将传递给model_fn使用
    warm_start_from=None #热启动目录路径
)
模型函数model_fn是唯一没有默认值的参数，它也是自定义Estimator最关键的部分，
包含了最核心的算法。model_fn需要一个能够进行运算的函数，它的样子应该长成这样
my_model(
  features, #输入的特征数据
  labels, #输入的标签数据
  mode, #train、evaluate或predict
  params #超参数，对应上面Estimator传来的参数
)
"""
#自定义模型函数
def my_model_fn(features,labels,mode,params):
    #输入层,feature_columns对应Classifier(feature_columns=...)
    net = tf.feature_column.input_layer(features, params['feature_columns'])
    # with tf.Session() as session:
    #     session.run(tf.global_variables_initializer())
    #     session.run(tf.tables_initializer())
    #     print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@数据输入到模型中的格式@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    #     print(session.run(net))
    #隐藏层,hidden_units对应Classifier(unit=[10,10])，2个各含10节点的隐藏层
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
    
    #输出层，n_classes对应3种鸢尾花
    logits = tf.layers.dense(net, params['n_classes'], activation=None)
    """
    训练train、评价evaluate和预测predict
    前面我们知道，自定义的估算分类器必须能够用来执行my_classifier.train()、my_classifier.evaluate()、my_classifier.predict()三个方法。
    但实际上，它们都是model_fn这一个函数的分身！
    上面出现的model_fn语法：
    my_model(
      features, #输入的特征数据
      labels, #输入的标签数据
      mode, #train、evaluate或predict
      params #超参数，对应上面Estimator传来的参数
    )
    注意第三个参数mode，如果它等于"TRAIN"我们就执行训练：
    #示意代码
    my_model(..,..,"TRAIN",...)
    如果是“EVAL”就执行评价，“PREDICT”就执行预测。
    我们修改my_model代码来实现这三个功能:

    """
    #预测
    predicted_classes = tf.argmax(logits, 1) #预测的结果中最大值即种类
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis], #拼成列表[[3],[2]]格式
            'probabilities': tf.nn.softmax(logits), #把[-1.3,2.6,-0.9]规则化到0~1范围,表示可能性
            'logits': logits,#[-1.3,2.6,-0.9]
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)


    #损失函数
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    
    #训练
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdagradOptimizer(learning_rate=0.1) #用它优化损失函数，达到损失最少精度最高
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())  #执行优化！
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)      
    
    #评价
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op') #计算精度
    metrics = {'accuracy': accuracy} #返回格式
    tf.summary.scalar('accuracy', accuracy[1]) #仅为了后面图表统计使用
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

import os
import pandas as pd

FUTURES = ['SepalLength', 'SepalWidth','PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

dir_path = os.path.dirname(os.path.realpath(__file__))
train_path=os.path.join(dir_path,'iris_training.csv')
test_path=os.path.join(dir_path,'iris_test.csv')

train = pd.read_csv(train_path, names=FUTURES, header=0)
train_x, train_y = train, train.pop('Species')

test = pd.read_csv(test_path, names=FUTURES, header=0)
test_x, test_y = test, test.pop('Species')

feature_columns = []
for key in train_x.keys():
    feature_columns.append(tf.feature_column.numeric_column(key=key))

"""
模型的恢复与保存设置
修改创建估算分类器的代码设置model_dir模型保存与自动恢复,并设定日志打印
tf.logging.set_verbosity(tf.logging.INFO)
models_path=os.path.join(dir_path,'mymodels/')
#创建自定义分类器
classifier = tf.estimator.Estimator(
        model_fn=my_model_fn,
        model_dir=models_path,
        params={
            'feature_columns': feature_columns,
            'hidden_units': [10, 10],
            'n_classes': 3,
        })

"""
tf.logging.set_verbosity(tf.logging.INFO)
models_path=os.path.join(dir_path,'models/')

#创建自定义分类器
classifier = tf.estimator.Estimator(
        model_fn=my_model_fn,
        model_dir=models_path,
        params={
            'feature_columns': feature_columns,
            'hidden_units': [10, 10],
            'n_classes': 3,
        })


#针对训练的喂食函数
batch_size=100
def train_input_fn(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.shuffle(1000).repeat().batch(batch_size) #每次随机调整数据顺序
    return dataset.make_one_shot_iterator().get_next()

#开始训练
classifier.train(
    input_fn=lambda:train_input_fn(train_x, train_y, 100), steps=1000)

#针对测试的喂食函数
def eval_input_fn(features, labels, batch_size):
    features=dict(features)
    inputs=(features,labels)
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    dataset = dataset.batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()

#评估我们训练出来的模型质量
eval_result = classifier.evaluate(
    input_fn=lambda:eval_input_fn(test_x, test_y, batch_size))

print(eval_result)


"""
应用模型
回到我们最初要解决的问题，也就是当女朋友把她的鸢尾花测量数据交给我们的时候，
我们可以让训练出来的人工智能模型来自动对这朵花进行分类，由于精度超过90%，
所以我们有超过九成的把握可以认为这个分类就是正确的。
"""
#支持100次循环对新数据进行分类预测
for i in range(0,100):
    print('\nPlease enter features: SepalLength,SepalWidth,PetalLength,PetalWidth')
    a,b,c,d = map(float, input().split(',')) #捕获用户输入的数字
    predict_x = {
        'SepalLength': [a],
        'SepalWidth': [b],
        'PetalLength': [c],
        'PetalWidth': [d],
    }

    #进行预测
    predictions = classifier.predict(
        input_fn=lambda:eval_input_fn(predict_x,
                                      labels=[0,],#这个输入是啥意思？？？eval_input_fn为甚么要传入labels列表。eval和predict不同,也是有监督，要传入labels；但是predict是不需要labels的。这里传入一个列表形式的labels，只为调用eval_input_fn不报错
                                      batch_size=batch_size))

    #预测结果是数组，尽管实际我们只有一个
    for pred_dict in predictions:
        class_id = pred_dict['class_ids'][0] #由于输出为 [[2],[1],[0],[2],...]形式，因此要再取一个pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]
        print(SPECIES[class_id],100 * probability)