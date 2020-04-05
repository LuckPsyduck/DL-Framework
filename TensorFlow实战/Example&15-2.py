# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import os 
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8 #基础的学习率
LEARNIGN_RATE_DECAY = 0.99 #学习率的衰减
REGULARIZATION_RATE = 0.0001 #描述模型复杂度的正则化项在损失函数中的系数
TRAINING_STEPS = 30000 #训练轮数
MOVING_AVERAGE_DECAY = 0.99 #滑动平均衰减

#MODEL_SAVE_PATH = "save/model.ckpt"
#MODEL_NAME = "model.ckpt"

def train(mnist):
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32,[None,mnist_inference.INPUT_NODE],name = 'x-input')
        y_ = tf.placeholder(tf.float32,[None,mnist_inference.OUTPUT_NODE],name = 'y-input')
    
    #计算L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    
    y = mnist_inference.inference(x,regularizer)
    global_step = tf.Variable(0,trainable = False)
    
    
    #给定滑动平均衰减率和训练轮数的变量，初始化滑动平均类
    #给定训练轮数的变量可以加快训练早期变量的更新速度
    with tf.name_scope('moving_average'):
        
        variable_averages = tf.train.ExponentialMovingAverage(
                MOVING_AVERAGE_DECAY,global_step)
        
        #在所有代表神经网络参数的变量上使用滑动平均
        variables_average_op = variable_averages.apply(tf.trainable_variables())
    

    #计算使用交叉熵作为刻画预测值和真实值之间的差距的损失函数
    """
    tf.argmax 是一个非常有用的函数，
    它能给出某个tensor对象在某一维上的其数据最大值所在的索引值。
    由于标签向量是由0,1组成，因此最大值1所在的索引位置就是类别标签，
    比如tf.argmax(y,1)返回的是模型对于任一输入x预测到的标签值，
    而 tf.argmax(y_,1) 代表正确的标签，
    """
    with tf.name_scope('loss_function'):
        
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=y,labels=tf.argmax (y_, 1))
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
    
        #总损失等于交叉熵和正则化损失的和
        loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    
     #设置指数衰减的学习率
    with tf.name_scope("train_step"):  
        learning_rate = tf.train.exponential_decay(
                LEARNING_RATE_BASE,global_step,
                mnist.train.num_examples/BATCH_SIZE,
                LEARNIGN_RATE_DECAY)
        #梯度下降法优化算法来优化损失函数
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(
                     loss,global_step=global_step)
    
    #在训练神经网络模型时，每过一遍数据需要通过反向传播算法更新神经网络的参数
    #又要更新每个参数的滑动平均值
        with tf.control_dependencies([train_step,variables_average_op]):
             train_op = tf.no_op(name = 'train')
     
        saver = tf.train.Saver()
        #初始化回话，并开始驯练
        with tf.Session() as sess:
             tf.initialize_all_variables().run() 
             
            #迭代的训练神经网络
             for i in range(TRAINING_STEPS):
                 xs,ys = mnist.train.next_batch(BATCH_SIZE)
                 _, loss_value,step = sess.run([train_op,loss,global_step],
                                               feed_dict = {x:xs,y_:ys})
                 if i % 1000 == 0:
                    print("Afer %d training steps,loss_on_training "
                          "batch is %g" %(step,loss_value))
                    saver.save(sess,os.path.join("save/model.ckpt"),
                                                global_step = global_step)
    writer = tf.summary.FileWriter("./log",tf.get_default_graph())
    writer.close()
    
def main(argv = None):
    mnist  = input_data.read_data_sets("/tmp/data",one_hot = True)
    train(mnist)
    
if __name__ == '__main__':
    tf.app.run()

    
    