# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 19:42:18 2019

@author: david
"""
import tensorflow as tf
import numpy as np
import os
from os import listdir
from os.path import isfile, join

class Model():
    
    def __init__(self, input_dim, lr, batch_size = 1,step=0):
        self.input_dim = input_dim #(width / time component, num_features / chanells)
        self.output_dim = 5
        self.lr = lr
        self.batch_size = batch_size
        self.step = step
        pass
    
    def build_model(self, w1=100, n1 = 100, s1=10, w2 = 10, n2 = 10, s2 = 1, n_lstm1 = 20):
        tf.compat.v1.disable_eager_execution()        
        with tf.device("/gpu:0"):
            
            self.Input1 = tf.compat.v1.placeholder(tf.float32, (None,)+self.input_dim, name="input1")
            self.Input2 = tf.compat.v1.placeholder(tf.float32, (None,)+self.input_dim, name="input2")
            self.Input3 = tf.compat.v1.placeholder(tf.float32, (None,)+self.input_dim, name="input3")
            self.Input4 = tf.compat.v1.placeholder(tf.float32, (None,)+self.input_dim, name="input4")
            self.Input5 = tf.compat.v1.placeholder(tf.float32, (None,)+self.input_dim, name="input5")
            
            Inputs = [self.Input1,self.Input2,self.Input3,self.Input4,self.Input5]
            
            self.Label = tf.compat.v1.placeholder(tf.float32, (None, self.output_dim), name="output")
                    
            #apply convolutions
            filters1 = []
            strides1 = []
            for i in range(5):
                #width, in_chanells, num output features
                filters1.append( tf.Variable(tf.zeros(np.array([w1,self.input_dim[1],n1])), name = "filter1"+str(i)))
                strides1.append( s1 )
            
            nets = []
            for i in range(5):
                nets.append( tf.nn.conv1d(Inputs[i], filters1[i], strides1[i], padding="SAME") )
            
            #apply convolutions
            filters2 = []
            strides2 = []
            n0_shape =  nets[0].get_shape()
            f2 = int(n0_shape[1] / 1)
            for i in range(5):
                filters2.append( tf.Variable(tf.zeros(np.array([w2,f2,n2])), name = "filter2"+str(i)))
                
                strides2.append( s2 )
            
            #print((nets[0].get_shape()))
            
            #print((filters1[0].get_shape()))
            for i in range(5):
                nets[i] = tf.nn.conv1d(nets[i], filters2[i], strides2[i], padding="SAME")
                
            #concantemate
            net = tf.concat(nets, 1)
            
            #apply lstm
            #print(net.get_shape())
            net_unstacked = tf.unstack(net,axis=1)
            rnn_cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(n_lstm1, name="LSTM_cell")
            outputs, states = tf.compat.v1.nn.static_rnn(rnn_cell, net_unstacked, dtype=tf.float32)
            outputs_st =tf.stack(outputs,axis=1)
            
            print(outputs_st.get_shape())
            #apply fully connected
            W = tf.Variable(tf.zeros(np.array([n_lstm1, self.output_dim])),name="W")
            b = tf.Variable(tf.zeros(np.array([self.output_dim])), name="b")
            self.output = tf.nn.relu(tf.matmul(outputs[-1], W) + b)
            
            #define loss and loss        
            self.loss = tf.math.reduce_mean(tf.compat.v2.losses.mse(self.output, self.Label))
            
            self.init_op = tf.compat.v1.global_variables_initializer()
            
            
            #self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = self.lr).minimize(self.loss)
            self.optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = self.lr).minimize(self.loss)
            
            #tf.compat.v1.get_default_graph().finalize()
            
    
    def build_model_1(self,
                      w1=100,
                      n1 = 2000,
                      s1=10,
                      w2 = 10,
                      n2 = 1000,
                      s2 = 1,
                      w3 = 10,
                      n3 = 500,
                      s3=1,
                      w1_out=1000,
                      w2_out=100,
                      w3_out=50):
        tf.compat.v1.disable_eager_execution()
        with tf.device("/gpu:0"):
            
            self.Input1 = tf.compat.v1.placeholder(tf.float32, (None,)+self.input_dim, name="input1")
            self.Input2 = tf.compat.v1.placeholder(tf.float32, (None,)+self.input_dim, name="input2")
            self.Input3 = tf.compat.v1.placeholder(tf.float32, (None,)+self.input_dim, name="input3")
            self.Input4 = tf.compat.v1.placeholder(tf.float32, (None,)+self.input_dim, name="input4")
            self.Input5 = tf.compat.v1.placeholder(tf.float32, (None,)+self.input_dim, name="input5")
            
            Inputs = [self.Input1,self.Input2,self.Input3,self.Input4,self.Input5]
            
            self.Label = tf.compat.v1.placeholder(tf.float32, (None, self.output_dim), name="output")
            
            #apply convolutions
            filters1 = []
            strides1 = []
            for i in range(5):
                #width, in_chanells, num output features
                filters1.append( tf.Variable(tf.zeros(np.array([w1,self.input_dim[1],n1])), name = "filter1"+str(i)))
                strides1.append( s1 )
            
            nets = []
            for i in range(5):
                nets.append( tf.nn.relu(tf.nn.conv1d(Inputs[i], filters1[i], strides1[i], padding="SAME", name="conv1") ))
            
            #apply convolutions
            filters2 = []
            strides2 = []
            n0_shape =  nets[0].get_shape()
            f2 = int(n0_shape[1] / 1)
            for i in range(5):
                filters2.append( tf.Variable(tf.zeros(np.array([w2,f2,n2])), name = "filter2"+str(i)))                
                strides2.append( s2 )
            
            for i in range(5):
                nets[i] = tf.nn.relu(tf.nn.conv1d(nets[i], filters2[i], strides2[i], padding="SAME", name="conv2"))
                
            #apply convolutions
            filters3 = []
            strides3 = []
            f3 = int( nets[0].get_shape()[1] )
            for i in range(5):
                filters3.append( tf.Variable(tf.zeros(np.array([w3, f3, n3])), name = "filter3"+str(i)))
                strides3.append(s3)
                
            for i in range(5):
                nets[i] = tf.nn.relu(tf.nn.conv1d(nets[i], filters3[i], strides3[i], padding="SAME", name="conv3"))
            
            #concantemate
            net = tf.concat(nets, 1)
            print("net shape",net.get_shape())
            
            #flatten
            net = tf.compat.v1.reshape(net, tf.compat.v2.convert_to_tensor((-1,net.get_shape()[1]*net.get_shape()[2])))
            print("net shape",net.get_shape())
            
            #fully connected
            W1 = tf.Variable(tf.zeros(np.array([net.get_shape()[1], w1_out])), name = "W1")
            b1 = tf.Variable(tf.zeros(np.array([w1_out])), name = "b1")
            net = tf.add( tf.matmul(net,W1) , b1)
            net = tf.nn.relu(net)
            
            W2 = tf.Variable(tf.zeros(np.array([w1_out, w2_out])), name = "W2")
            b2 = tf.Variable(tf.zeros(np.array([w2_out])), name = "b2")
            net = tf.add( tf.matmul(net,W2) , b2)
            net = tf.nn.relu(net)
            
            W3 = tf.Variable(tf.zeros(np.array([w2_out, w3_out])), name = "W3")
            b3 = tf.Variable(tf.zeros(np.array([w3_out])), name = "b3")
            net = tf.add( tf.matmul(net,W3) , b3)
            net = tf.nn.relu(net)
            
            W4 = tf.Variable(tf.zeros(np.array([w3_out, self.output_dim])), name = "W4")
            b4 = tf.Variable(tf.zeros(np.array([self.output_dim])), name = "b4")
            net = tf.add( tf.matmul(net,W4) , b4)
            
            self.output = net
            
            #define loss and loss        
            self.loss = tf.math.reduce_mean(tf.compat.v2.losses.mse(self.output, self.Label))
            
            self.init_op = tf.compat.v1.global_variables_initializer()
            
            self.optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = self.lr).minimize(self.loss)
        
        
    def initialize_variables_and_sess(self):
        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
        self.sess.run(self.init_op)
        
    def predict(self, inputs):
        if len(inputs[0].shape) == 2:
            new_shape = (1,inputs[0].shape[0],inputs[0].shape[1])
        elif len(inputs[0].shape) == 3:
            new_shape = inputs[0].shape
        
        fd = {
            self.Input1 : np.reshape(inputs[0], new_shape),
            self.Input2 : np.reshape(inputs[1], new_shape),
            self.Input3 : np.reshape(inputs[2], new_shape),
            self.Input4 : np.reshape(inputs[3], new_shape),
            self.Input5 : np.reshape(inputs[4], new_shape)
        }        
        
        output = self.sess.run(self.output, feed_dict = fd)
        return output
    
    def update_model(self, inputs, labels):
        if len(inputs[0].shape) == 2:
            new_shape = (1,inputs[0].shape[0],inputs[0].shape[1])
        elif len(inputs[0].shape) == 3:
            new_shape = inputs[0].shape
            
        fd = {
            self.Input1 : np.reshape(inputs[0], new_shape),
            self.Input2 : np.reshape(inputs[1], new_shape),
            self.Input3 : np.reshape(inputs[2], new_shape),
            self.Input4 : np.reshape(inputs[3], new_shape),
            self.Input5 : np.reshape(inputs[4], new_shape),
            self.Label : np.reshape(np.array(labels), (self.batch_size,-1))
        }
        
        _, loss = self.sess.run([self.optimizer, self.loss], feed_dict = fd)
        return loss
    
    def close_session(self):
        self.sess.close()
    
    def initialze_saver(self):
        self.saver = tf.compat.v1.train.Saver()
    
    def maybe_make_ckpt_dir(self, directory = "../crypto_results/checkpoint"):
        if not os.path.isdir(directory):
            os.mkdir(directory)
    
    def save_model(self, directory="../crypto_results/checkpoint"):
        self.maybe_make_ckpt_dir(directory)
        filename = directory + "/" + "crypto_prediction_model"
        self.saver.save(self.sess, filename, global_step = self.step)
        
    def get_latest_checkpoint(self, directory = "../crypto_results/checkpoint"):
        mypath = directory
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        onlymeta = [f for f in onlyfiles if f[-4:] == "meta"]
        maxi = -1
        for f in onlymeta:
            num = int(f[:-5].split("-")[1])
            if num > maxi:
                maxi = num
                filename = f
        
        self.step = maxi
        self.latest_metafile = filename
        
    def restore_latest_session(self, directory="../crypto_results/checkpoint"):
        self.sess = tf.compat.v1.Session()
        self.sess.run(self.init_op)
        self.get_latest_checkpoint(directory)
        print("restoring from model " + self.latest_metafile)
        filename = directory + "/" + self.latest_metafile
        self.saver = tf.compat.v1.train.import_meta_graph(filename)
        ckpt = tf.compat.v2.train.latest_checkpoint(directory)
        self.saver.restore(self.sess, ckpt)
        
        
    def restore_latest_session1(self, directory="../crypto_results/checkpoint"):
        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
        self.sess.run(self.init_op)
        print("restoring model")
        self.get_latest_checkpoint(directory)
        self.initialze_saver()        
        ckpt = tf.compat.v2.train.latest_checkpoint(directory)
        self.saver.restore(self.sess, ckpt)
        
        
    
        
        
        