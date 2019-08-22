import os
import pandas as pd
import tensorflow as tf

FUTURES = ['SepalLength', 'SepalWidth','PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

#格式化数据文件的目录地址
dir_path = os.path.dirname(os.path.realpath(__file__))
train_path=os.path.join(dir_path,'iris_training.csv')
test_path=os.path.join(dir_path,'iris_test.csv')

#载入训练数据
train = pd.read_csv(train_path, names=FUTURES, header=0)
train_x, train_y = train, train.pop('Species')

#载入测试数据
test = pd.read_csv(test_path, names=FUTURES, header=0)
test_x, test_y = test, test.pop('Species')

#设定特征值的名称
feature_columns = []
for key in train_x.keys():
    feature_columns.append(tf.feature_column.numeric_column(key=key))

#选定评估器：深层神经网络分类器  
classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[10, 10],
    n_classes=3)

#针对训练的喂食函数
def train_input_fn(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.shuffle(1000).repeat().batch(batch_size) #每次随机调整数据顺序
    return dataset

#设定仅输出警告提示，可改为INFO
tf.logging.set_verbosity(tf.logging.WARN)

#开始训练模型！
batch_size=100
classifier.train(input_fn=lambda:train_input_fn(train_x, train_y,batch_size),steps=1000)

#针对测试的喂食函数
def eval_input_fn(features, labels, batch_size):
    features=dict(features)
    inputs=(features,labels)
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    dataset = dataset.batch(batch_size)
    return dataset

#评估我们训练出来的模型质量
eval_result = classifier.evaluate(
    input_fn=lambda:eval_input_fn(test_x, test_y,batch_size))

print(eval_result)

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
                                            labels=[0],
                                            batch_size=batch_size))

    #预测结果是数组，尽管实际我们只有一个
    for pred_dict in predictions:
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]
        print(SPECIES[class_id],100 * probability)
