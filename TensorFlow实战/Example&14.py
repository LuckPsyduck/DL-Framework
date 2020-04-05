# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

max_step = 1000  # 最大迭代次数
learning_rate = 0.001   # 学习率
dropout = 0.9   # dropout时随机保留神经元的比例

# 样本数据存储的路径
data_dir = "./MNIST_data"  
log_dir = "./log"    # 输出日志保存的路径

mnist = input_data.read_data_sets(data_dir,one_hot=True)

sess = tf.InteractiveSession()

#创建输入数据的占位符，分别创建特征数据x，标签数据y_ 
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
    
"""
使用tf.summary.image保存图像信息 
特征数据其实就是图像的像素数据拉升成一个1*784的向量，
现在如果想在tensorboard上还原出输入的特征数据对应的图片，
就需要将拉升的向量转变成28 * 28 * 1的原始像素了，
于是可以用tf.reshape()直接重新调整特征数据的维度：
将输入的数据转换成[28 * 28 * 1]的shape，
存储成另一个tensor，命名为image_shaped_input。
使用tf.summary.image将图片数据汇总给tensorbord。
tf.summary.image（）第三个是最多展示的张数
"""
with tf.name_scope('input_reshape'):
#-1代表的含义是不用我们自己指定这一维的大小，函数会自动计算，但列表中只能存在一个-1
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shaped_input, 10)
 
def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

"""
使训练中的参数可见
"""
def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
      # 计算参数的均值，并使用tf.summary.scaler记录
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)

      # 计算参数的标准差
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      # 使用tf.summary.scaler记录记录下标准差，最大值，最小值
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      # 用直方图记录参数的分布
      tf.summary.histogram('histogram', var)
      
      """
    创建第一层隐藏层
    创建一个构建隐藏层的方法,输入的参数有：
    input_tensor：特征数据
    input_dim：输入数据的维度大小
    output_dim：输出数据的维度大小(=隐层神经元个数）
    layer_name：命名空间
    act=tf.nn.relu：激活函数（默认是relu)
      """
def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer.
    It does a matrix multiply, bias add, and then uses relu to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # 设置命名空间
    with tf.name_scope(layer_name):
      # 调用之前的方法初始化权重w，并且调用参数信息的记录方法，记录w的信息
      with tf.name_scope('weights'):
        weights = weight_variable([input_dim, output_dim])
        variable_summaries(weights)
      # 调用之前的方法初始化权重b，并且调用参数信息的记录方法，记录b的信息
      with tf.name_scope('biases'):
        biases = bias_variable([output_dim])
        variable_summaries(biases)
      # 执行wx+b的线性计算，并且用直方图记录下来
      with tf.name_scope('linear_compute'):
        preactivate = tf.matmul(input_tensor, weights) + biases
        tf.summary.histogram('linear', preactivate)
      # 将线性输出经过激励函数，并将输出也用直方图记录下来
      activations = act(preactivate, name='activation')
      tf.summary.histogram('activations', activations)

      # 返回激励层的最终输出
      return activations
  
# 调用隐层创建函数创建一个隐藏层：输入的维度是特征的维度784，
#神经元个数是500，也就是输出的维度 
hidden1 = nn_layer(x, 784, 500, 'layer1')

"""
创建一个dropout层，,随机关闭掉hidden1的一些神经元，并记录keep_prob
"""
with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    tf.summary.scalar('dropout_keep_probability', keep_prob)
    dropped = tf.nn.dropout(hidden1, keep_prob)
    
    """
    创建一个输出层，输入的维度是上一层的输出:500,
    输出的维度是分类的类别种类：10，激活函数设置为全等映射identity.
    （暂且先别使用softmax,会放在之后的损失函数中一起计算）
    """ 
y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)   

"""
使用tf.nn.softmax_cross_entropy_with_logits来计算softmax
并计算交叉熵损失,并且求均值作为最终的损失值
"""
with tf.name_scope('loss'):
    # 计算交叉熵损失（每个样本都会有一个损失）
    diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
    with tf.name_scope('total'):
      # 计算所有样本交叉熵损失的均值
      cross_entropy = tf.reduce_mean(diff)
tf.summary.scalar('loss', cross_entropy)

"""
    使用AdamOptimizer优化器训练模型，最小化交叉熵损失
"""
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(
        cross_entropy)

"""
计算准确率,并用tf.summary.scalar记录准确率
"""
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      # 分别将预测和真实的标签中取出最大值的索引，弱相同则返回1(true),不同则返回0(false)
      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
      # 求均值即为准确率
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

"""
"""
# summaries合并
merged = tf.summary.merge_all()
# 写到指定的磁盘路径中
train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
test_writer = tf.summary.FileWriter(log_dir + '/test')

# 运行初始化所有变量
tf.global_variables_initializer().run()

"""
现在我们要获取之后要喂人的数据.
如果是train==true，就从mnist.train中获取一个batch样本
，并且设置dropout值；
如果是不是train==false,则获取minist.test的测试数据，
并且设置keep_prob为1，即保留所有神经元开启
"""
def feed_dict(train):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if train:
      xs, ys = mnist.train.next_batch(100)
      k = dropout
    else:
      xs, ys = mnist.test.images, mnist.test.labels
      k = 1.0
    return {x: xs, y_: ys, keep_prob: k}

"""
开始训练模型。
每隔10步，就进行一次merge, 并打印一次测试数据集的准确率，
然后将测试数据集的各种summary信息写进日志中。
每隔100步，记录原信息
其他每一步时都记录下训练集的summary信息并写到日志中
"""
for i in range(max_step):
    if i % 10 == 0:  # 记录测试集的summary与accuracy
      summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
      test_writer.add_summary(summary, i)
      print('Accuracy at step %s: %s' % (i, acc))
    else:  # 记录训练集的summary
      if i % 100 == 99:  # Record execution stats
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary, _ = sess.run([merged, train_step],
                              feed_dict=feed_dict(True),
                              options=run_options,
                              run_metadata=run_metadata)
        train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
        train_writer.add_summary(summary, i)
        print('Adding run metadata for', i)
      else:  # Record a summary
        summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
        train_writer.add_summary(summary, i)
        
train_writer.close()
test_writer.close()