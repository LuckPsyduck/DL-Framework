# -*- coding: utf-8 -*-

import tensorflow as tf

#获取一层神经网络边上的权重，并将这个权重的L2正则化损失加入名称为‘losses’的集合中
def get_weight(shape,a):
    var = tf.Variable(tf.random_normal(shape),dtype=tf.float32)
    tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(a)(var))
    return var

x = tf.placeholder(tf.float32,shape = (None,2))
y_ = tf.placeholder(tf.float32,shape = (None,1))
batch_size = 8

#定义每一层网络节点的个数
layer_dimension = [2,10,10,10,1]

n_layers = len(layer_dimension)

#这个变量维护前向算法传播时最深层的节点，开始时为输入层
cur_layer = x

in_dimension = layer_dimension[0]#当前层的结点数

#通过一个循环来生成5层全连接的神经网络结构
for i in range(1,n_layers):
    out_dimension = layer_dimension[i]#下一层的结点数
    #生成当前层中权重的变量
    weight = get_weight([in_dimension,out_dimension],0.001)
    bias = tf.Variable(tf.constant(0.1,shape = [out_dimension]))
    #使用ReLU激活函数
    cur_layer = tf.nn.relu(tf.matmul(cur_layer,weight)+bias)
    
    in_dimension = layer_dimension[i]#g更新
    
mse_loss = tf.reduce_mean(tf.square(y_-cur_layer))
tf.add_to_collection('losses',mse_loss)

loss = tf.add_n(tf.get_collection('losses'))
    
    
    
    
    