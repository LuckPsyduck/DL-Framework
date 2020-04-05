# -*- coding: utf-8 -*-
import tensorflow as tf

#with tf.name_scope("hello") as name_scope:
#    arr1 = tf.get_variable("arr1", shape=[2,10],dtype=tf.float32)
#
#    print (name_scope)
#    print (arr1.name)
#    print ("scope_name:%s " % tf.get_variable_scope().original_name_scope)
#    """   
#    tf.name_scope() 返回的是 一个string,”hello/”
#    在name_scope使用 get_variable() 中定义的 variable 的 name 并没有 “hello/”前缀
#    tf.get_variable_scope()的original_name_scope 是空
#   name_scope对 get_variable()创建的变量 的名字不会有任何影响,
#   而创建的op会被加上前缀.
#    """
    
with tf.variable_scope("hello") as variable_scope:
    arr1 = tf.get_variable("arr1", shape=[2, 10], dtype=tf.float32)

    print (variable_scope)
    print (variable_scope.name) #打印出变量空间名字
    print (arr1.name)
    print (tf.get_variable_scope().original_name_scope)
    #tf.get_variable_scope() 获取的就是variable_scope

    with tf.variable_scope("xixi") as v_scope2:
        print (tf.get_variable_scope().original_name_scope)
        #tf.get_variable_scope() 获取的就是v _scope2
        
        """  
    tf.variable_scope() 返回的是一个 VariableScope 对象
    variable_scope使用 get_variable 定义的variable 的name加上了”hello/”前缀
    tf.get_variable_scope()的original_name_scope 是 嵌套后的scope name
        """
#with tf.name_scope("name1"):
#    with tf.variable_scope("var1"):
#        w = tf.get_variable("w",shape=[2])
#        res = tf.add(w,[3])
#
#print (w.name)
#print (res.name)
#
#"""
#    variable scope和name scope都会给op的name加上前缀
#        这实际上是因为 创建 variable_scope 时内部会创建一个同名的 name_scope
#
#"""   



#with tf.name_scope('hidden') as scope:
#  a = tf.constant(5, name='alpha')
#  W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0), name='weights')
#  b = tf.Variable(tf.zeros([1]), name='biases')
#  print (a.name)
#  print (W.name)
#  print (b.name)
#  """
#  name_scope 是给op_name加前缀, 
#  variable_scope是给get_variable()创建的变量的名字加前缀
#  """
#
#def test(name=None):
#    with tf.variable_scope(name, default_name="scope") as scope:
#        w = tf.get_variable("w", shape=[2, 10])
#
#test()
#test()
#ws = tf.trainable_variables()
#for w in ws:
#    print(w.name)
#    
#  """
#  可以看出，如果只是使用default_name这个属性来创建variable_scope
#  的时候，会处理命名冲突
#  """


#with tf.name_scope("hehe"):
#    w1 = tf.Variable(1.0)
#    with tf.name_scope(None):
#        w2 = tf.Variable(2.0)
#print(w1.name)
#print(w2.name)
"""
tf.name_scope(None) 有清除name scope的作用
"""
