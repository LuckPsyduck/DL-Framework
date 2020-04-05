# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#MNIST数据集相关的常数
INPUT_NODE = 784 #输入层的结点数，图片像素
OUTPUT_NODE = 10 #输出层的结点数，0-9

#配置神经网络的参数
LAYER1_NODE = 500 #隐藏层结点数
BATCH_SIZE = 100 #一个训练batch中的训练数据个数

LEARNING_RATE_BASE = 0.8 #基础的学习率
LEARNIGN_RATE_DECAY = 0.99 #学习率的衰减
REGULARIZATION_RATE = 0.0001 #描述模型复杂度的正则化项在损失函数中的系数
TRAINING_STEPS = 30000 #训练轮数
MOVING_AVERAGE_DECAY = 0.99 #滑动平均衰减

#一个辅助函数，给定神经网络的输入和所有参数，计算神经网络的前向传播算法
def inference(input_tensor,avg_class,weights1,biases1,weights2,biases2):
    #如果当时没有提供滑动平均类时，直接使用参数当前的值
    if avg_class == None:
        #计算隐藏层的前向传播结果
        layer1 = tf.nn.relu(tf.matmul(input_tensor,weights1)+biases1)
        return tf.matmul(layer1,weights2)+biases2 
    else:
        #首先使用avg_class.average函数来计算得出变量的滑动平均值
        layer1 = tf.nn.relu(tf.matmul(input_tensor,avg_class.average(weights1))
        +avg_class.average(biases1))
        return tf.matmul(layer1,avg_class.average(weights2))+avg_class.average(biases2)
                            
    
#训练模型的过程  
def train(mnist):
    x = tf.placeholder(tf.float32,[None,INPUT_NODE],name = 'x-input')
    y_ = tf.placeholder(tf.float32,[None,OUTPUT_NODE],name = 'y-input')
    
    #生成隐藏层的参数 
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE,LAYER1_NODE],stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1,shape=[LAYER1_NODE]))
    #生成输出层的参数
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE,OUTPUT_NODE],stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1,shape=[OUTPUT_NODE]))
    
    #计算在当下参数下神经网络前行传播的结果，无平滑平均类
    y = inference(x,None,weights1,biases1,weights2,biases2)
    
    #定义存储训练轮数的变量，为不可训练的变量
    globe_step = tf.Variable(0,trainable=False)
    
    #给定滑动平均衰减率和训练轮数的变量，初始化滑动平均类
    #给定训练轮数的变量可以加快训练早期变量的更新速度
    variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY,globe_step)
    
    #在所有代表神经网络参数的变量上使用滑动平均
    variables_average_op = variable_averages.apply(tf.trainable_variables())
    
    #计算使用滑动平均之后的前向传播结果
    average_y = inference(x,variable_averages,weights1,biases1,weights2,biases2)
    
    #计算使用交叉熵作为刻画预测值和真实值之间的差距的损失函数
    """
    tf.argmax 是一个非常有用的函数，
    它能给出某个tensor对象在某一维上的其数据最大值所在的索引值。
    由于标签向量是由0,1组成，因此最大值1所在的索引位置就是类别标签，
    比如tf.argmax(y,1)返回的是模型对于任一输入x预测到的标签值，
    而 tf.argmax(y_,1) 代表正确的标签，
    """
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=y,labels=tf.argmax (y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    
    #计算L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    #计算模型的正则化损失，一般只计算神经网络边上的权重的正则化损失，而不使用偏置项
    regularization = regularizer(weights1) + regularizer(weights2)
    #总损失等于交叉熵和正则化损失的和
    loss = cross_entropy_mean + regularization
    
    #设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(
            LEARNING_RATE_BASE,globe_step,
            mnist.train.num_examples/BATCH_SIZE,
            LEARNIGN_RATE_DECAY)
    #梯度下降法优化算法来优化损失函数
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(
                 loss,global_step=globe_step)
    
    #在训练神经网络模型时，每过一遍数据需要通过反向传播算法更新神经网络的参数
    #又要更新每个参数的滑动平均值
    with tf.control_dependencies([train_step,variables_average_op]):
         train_op = tf.no_op(name = 'train')
     
    #判断两个张量的每一位维是否相等，如果相等则返回TRrue，否则返回False
    correct_prediction = tf.equal(tf.arg_max(average_y,1),tf.argmax(y_,1))
    #将结果变成数
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    
    #初始化回话，并开始驯练
    with tf.Session() as sess:
         tf.initialize_all_variables().run() 
         #准备验证数据
         validate_feed = {x:mnist.validation.images,y_:mnist.validation.labels}          
        
        #z准备测试数据
         test_feed = {x:mnist.test.images,y_:mnist.test.labels}
        
        #迭代的训练神经网络
         for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy,feed_dict = validate_feed)
                print("Afer %d training steps,validation accuracy "
                      "using average model is %g" %(i,validate_acc))
            #产生这一轮使用的一个batch的训练数据，并进行训练
            xs,ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op,feed_dict = {x:xs,y_:ys})
         
        #在驯练结束后，在测试数据集上检测神经网路模型的最终正确李
         test_acc = sess.run(accuracy,feed_dict = test_feed)
         print("Afer %d training steps,test accuracy"
               " using average model is %g" %(TRAINING_STEPS,test_acc))
         """
         validate_acc = sess.run(accuracy,feed_dict = validate_feed)
         test_acc = sess.run(accuracy,feed_dict = test_feed)
         print("After %d training,validation accuracy using average"
              "model is %g,test accuracy using average model is %g"
              %(i,validate_acc,test_acc))                 
         """
def main(argv=None):
    mnist = input_data.read_data_sets("/tmp/data",one_hot = True)
    train(mnist)

if __name__ =='__main__':#如果直接.py则执行下面的语句，如果导入，则不执行下面的语句
    tf.app.run()#main函数入口
    

    