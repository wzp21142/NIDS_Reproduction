import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.python.ops import resources
from tensorflow.contrib
dataset = pd.read_csv('./data/train.csv').values

####################
# 读入数据进行数据处理
# 最终 X.shape 为 [?,784]
# y.shape 为 [?]，y 必须为数字标签，非 onehot标签。
#
# X 需要归一化处理
####################
# paremeters
num_steps = 5000  # 迭代次数
batch_size = 1000  # 每个处理批次的大小
num_classes = 10  # 类别的数目
num_features = 784  # 特征的数目
num_trees = 50  # 森林里树的个数
max_nodes = 800  # 每棵数的最大节点数目

# 定义 X 和 y
X = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.int32, shape=[None])
x = tf.compat.v1.contri
# 定义随机森林的内部参数
hparams = tensor_forest.ForestHParams(num_classes=num_classes, num_features=num_features, num_trees=num_trees,
                                      max_nodes=max_nodes).fill()

# 创建随机森林运算图
forest_graph = tensor_forest.RandomForestGraphs(hparams)

# 定义训练运算图和 loss
train_op = forest_graph.training_graph(X, y)
loss_op = forest_graph.training_loss(X, y)

# 计算准确率
infer_op, _, _ = forest_graph.inference_graph(X)
correct_prediction = tf.equal(tf.argmax(infer_op, 1), tf.cast(y, tf.int64))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 定义图的初始化内容
init_var = tf.group(tf.global_variables_initializer(), resources.initialize_resources(resources.shared_resources()))

# 初始化
sess = tf.Session()
sess.run(init_var)


# 定义批次为随机 size 个训练数据
def next_batch(X_data, y_data, size):
    rand = np.arange(y_data.shape[0])
    np.random.shuffle(rand)
    return X_data[rand[0: size]], y_data[rand[0: size]]


# 训练过程
for i in range(1, num_steps + 1):
    batch = next_batch(X_train, y_train, batch_size)
    _, cost = sess.run([train_op, loss_op], feed_dict={X: batch[0], y: batch[1]})

    if i % 100 == 0 or i == 1:
        train_acc = sess.run(accuracy_op, feed_dict={X: batch[0], y: batch[1]})
        val_acc = sess.run(accuracy_op, feed_dict={X: X_val, y: y_val})
        print('Step %i, Loss: %f, Train_Acc: %s, Test_acc: %s' % (i, cost, train_acc, val_acc))

print("Test Accuracy:", sess.run(accuracy_op, feed_dict={X: X_test y: y_test}))