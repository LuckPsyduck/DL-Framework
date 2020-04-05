# -*- coding: utf-8 -*-

import tensorflow as tf

#定义一个变量用于计算滑动平均，这个变量的初始值为0
v1 = tf.Variable(0,dtype=tf.float32)
#模拟神经网络中的迭代的轮数，可以用于动态控制衰减率
step = tf.Variable(0,trainable=False)
#定义一个滑动平均的类，初始化衰减率(0.99)和控制衰减率的变量step
ema = tf.train.ExponentialMovingAverage(0.99,step)

maintain_averages_op = ema.apply([v1])

with tf.Session() as sess:
    #初始化所有的变量
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    #通过ema.average(v1)获取滑动平均之后变量的取值，在初始化之后变量V1的值和v1的滑动平均都为0
    print(sess.run([v1,ema.average(v1)]))#[0.0,0.0]
    
    sess.run(tf.assign(v1,5))#更新v1的值到5
    
    #更新v1的滑动平均值，衰减率为min{0.99,(1+step)/(10+step)=0.1}=0.1
    #所以v1的滑动平均会被更新为0.1*0+0.9*5=4.5
    sess.run(maintain_averages_op)
    print(sess.run([v1,ema.average(v1)]))#[5.0,4.5]
    
    sess.run(tf.assign(step,10000))
    sess.run(tf.assign(v1,10))
    
     #更新v1的滑动平均值，衰减率为min{0.99,(1+step)/(10+step)=0.999}=0.99
    #所以v1的滑动平均会被更新为0.99*4.5+0.01*10=4.555
    sess.run(maintain_averages_op)
    print(sess.run([v1,ema.average(v1)]))#[10.0,4.554998]
    
    sess.run(maintain_averages_op)
    print(sess.run([v1,ema.average(v1)]))
    
    
    