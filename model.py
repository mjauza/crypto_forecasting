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
    
    def build_model(self, w1=10, n1 = 1000, s1=1, w2 = 100, n2 = 500, s2 = 1, n_lstm1 = 100):
        tf.compat.v1.disable_eager_execution()        
        with tf.device("/gpu:0"):
            
            self.Input1 = tf.compat.v1.placeholder(tf.float32, (None,)+self.input_dim, name="input1")
            self.Input2 = tf.compat.v1.placeholder(tf.float32, (None,)+self.input_dim, name="input2")
            self.Input3 = tf.compat.v1.placeholder(tf.float32, (None,)+self.input_dim, name="input3")
            self.Input4 = tf.compat.v1.placeholder(tf.float32, (None,)+self.input_dim, name="input4")
            self.Input5 = tf.compat.v1.placeholder(tf.float32, (None,)+self.input_dim, name="input5")
            
            Inputs = [self.Input1,self.Input2,self.Input3,self.Input4,self.Input5]
            
            self.Label = tf.compat.v1.placeholder(tf.float32, (None, self.output_dim), name="output")
            
            initializer = tf.compat.v2.random_normal_initializer()
            
            #apply convolutions
            filters1 = []
            strides1 = []
            for i in range(5):
                #width, in_chanells, num output features
                filters1.append( tf.Variable(initializer(shape = np.array([w1,self.input_dim[1],n1])), name = "filter1"+str(i)))
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
                filters2.append( tf.Variable(initializer(shape = np.array([w2,f2,n2])), name = "filter2"+str(i)))                
                strides2.append( s2 )
            
            #print((filters1[0].get_shape()))
            for i in range(5):
                nets[i] = tf.nn.conv1d(nets[i], filters2[i], strides2[i], padding="SAME")
            
            w3 = 50
            n3 = 500
            s3 = 1
            n1_shape = nets[0].get_shape()
            f3 = int(n1_shape[1])
            filters3 = []
            strides3 = []
            for i in range(5):
                filters3.append(tf.Variable(initializer(shape = np.array([w3,f3,n3])), name = "filter3"+str(i)))
                strides3.append(s3)
                
            for i in range(5):
                nets[i] = tf.nn.conv1d(nets[i], filters3[i], strides3[i], padding = "SAME")
            
            #concantemate
            net = tf.concat(nets, 1)
            
            #apply lstm
            
            net_unstacked = tf.unstack(net,axis=1)
            rnn_cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(n_lstm1, name="LSTM_cell")
            outputs1, states1 = tf.compat.v1.nn.static_rnn(rnn_cell, net_unstacked, dtype=tf.float32)
            
            
            #apply lstm
            n_lstm2 = 100
            rnn_cell2 = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(n_lstm2, name = "LSTM_cell2")
            outputs2, states2 = tf.compat.v1.nn.static_rnn(rnn_cell2, outputs1, dtype = tf.float32)
            
            #apply lstm
            n_lstm3 = 100
            rnn_cell3 = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(n_lstm3, name = "LSTM_cell3")
            outputs3, states3 = tf.compat.v1.nn.static_rnn(rnn_cell3, outputs2, dtype = tf.float32)
            
            #outputs_st =tf.stack(outputs3,axis=1)
            net = tf.compat.v1.concat(outputs3, 1)
            print("net.get_shape() = ",net.get_shape())
            
            in_shape = net.get_shape()[1]
            #apply fully connected
            W1_n = 500
            W1 = tf.Variable(initializer(shape = np.array([in_shape, W1_n])), name = "W1")
            b1 = tf.Variable(initializer(shape = np.array([W1_n])), name="b1")
            
            net = tf.nn.relu(tf.add(tf.matmul(net,W1), b1))
            
            W2_n = 500
            W2 = tf.Variable(initializer(shape = np.array([W1_n, W2_n])), name = "W2")
            b2 = tf.Variable(initializer(shape = np.array([W2_n])), name="b2")
            
            net = tf.nn.relu(tf.add(tf.matmul(net, W2),b2))
                       
            W3 = tf.Variable(initializer(shape = np.array([W2_n, self.output_dim])),name="W3")
            b3 = tf.Variable(initializer(shape = np.array([self.output_dim])), name="b3")
            self.output = tf.add(tf.matmul(net, W3), b3)
            
            #define loss and loss        
            self.loss = tf.math.reduce_mean(tf.compat.v2.losses.mse(self.output, self.Label))
            
            self.init_op = tf.compat.v1.global_variables_initializer()
                        
            self.optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = self.lr).minimize(self.loss)
            
            
    
    def build_model_1(self,
                      w1=100,
                      n1 = 1000,
                      s1=10,
                      w2 = 10,
                      n2 = 1000,
                      s2 = 1,
                      w3 = 10,
                      n3 = 500,
                      s3=1,
                      w1_out=500,
                      w2_out=500,
                      w3_out=500):
        tf.compat.v1.disable_eager_execution()
        with tf.device("/gpu:0"):
            
            self.Input1 = tf.compat.v1.placeholder(tf.float32, (None,)+self.input_dim, name="input1")
            self.Input2 = tf.compat.v1.placeholder(tf.float32, (None,)+self.input_dim, name="input2")
            self.Input3 = tf.compat.v1.placeholder(tf.float32, (None,)+self.input_dim, name="input3")
            self.Input4 = tf.compat.v1.placeholder(tf.float32, (None,)+self.input_dim, name="input4")
            self.Input5 = tf.compat.v1.placeholder(tf.float32, (None,)+self.input_dim, name="input5")
            
            Inputs = [self.Input1,self.Input2,self.Input3,self.Input4,self.Input5]
            
            self.Label = tf.compat.v1.placeholder(tf.float32, (None, self.output_dim), name="output")
            
            initializer = tf.compat.v2.random_normal_initializer()
            
            #apply convolutions
            filters1 = []
            strides1 = []
            for i in range(5):
                #width, in_chanells, num output features
                filters1.append( tf.Variable(initializer(shape = np.array([w1,self.input_dim[1],n1])), name = "filter1"+str(i)))
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
                filters2.append( tf.Variable(initializer(shape = np.array([w2,f2,n2])), name = "filter2"+str(i)))                
                strides2.append( s2 )
            
            for i in range(5):
                nets[i] = tf.nn.relu(tf.nn.conv1d(nets[i], filters2[i], strides2[i], padding="SAME", name="conv2"))
                
            #apply convolutions
            filters3 = []
            strides3 = []
            f3 = int( nets[0].get_shape()[1] )
            for i in range(5):
                filters3.append( tf.Variable(initializer(shape = np.array([w3, f3, n3])), name = "filter3"+str(i)))
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
            W1 = tf.Variable(initializer(shape = np.array([net.get_shape()[1], w1_out])), name = "W1")
            b1 = tf.Variable(initializer(shape = np.array([w1_out])), name = "b1")
            net = tf.add( tf.matmul(net,W1) , b1)
            net = tf.nn.relu(net)
            
            W2 = tf.Variable(initializer(shape = np.array([w1_out, w2_out])), name = "W2")
            b2 = tf.Variable(initializer(shape = np.array([w2_out])), name = "b2")
            net = tf.add( tf.matmul(net,W2) , b2)
            net = tf.nn.relu(net)
            
            W3 = tf.Variable(initializer(shape = np.array([w2_out, w3_out])), name = "W3")
            b3 = tf.Variable(initializer(shape = np.array([w3_out])), name = "b3")
            net = tf.add( tf.matmul(net,W3) , b3)
            net = tf.nn.relu(net)
            
            W4 = tf.Variable(initializer(shape = np.array([w3_out, self.output_dim])), name = "W4")
            b4 = tf.Variable(initializer(shape = np.array([self.output_dim])), name = "b4")
            net = tf.add( tf.matmul(net,W4) , b4)
            
            self.output = net
            
            #define loss and loss        
            self.loss = tf.math.reduce_mean(tf.compat.v2.losses.mse(self.output, self.Label))
            
            self.init_op = tf.compat.v1.global_variables_initializer()
            
            self.optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = self.lr).minimize(self.loss)
            
    
    def build_model_2(self):
        
        tf.compat.v1.disable_eager_execution()
        with tf.device("/gpu:0"):
            
            #deifne inouts and labels
            self.Input1 = tf.compat.v1.placeholder(tf.float32, (None,)+self.input_dim, name="input1")
            self.Input2 = tf.compat.v1.placeholder(tf.float32, (None,)+self.input_dim, name="input2")
            self.Input3 = tf.compat.v1.placeholder(tf.float32, (None,)+self.input_dim, name="input3")
            self.Input4 = tf.compat.v1.placeholder(tf.float32, (None,)+self.input_dim, name="input4")
            self.Input5 = tf.compat.v1.placeholder(tf.float32, (None,)+self.input_dim, name="input5")
            
            Inputs = [self.Input1,self.Input2,self.Input3,self.Input4,self.Input5]
            
            self.Label = tf.compat.v1.placeholder(tf.float32, (None, self.output_dim), name="output")
            
            #apply lstm for each input
            n_lstm1 = 1000
            #lstm_cells = []
            #for i in range(5):
            #    lstm_cells.append(tf.compat.v1.nn.rnn_cell.BasicLSTMCell(n_lstm1, name="LSTM1_"+str(i)))
            
            lstm_cell1 = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(n_lstm1, name="LSTM1")
            initializer = tf.compat.v2.random_normal_initializer()
            inputs_unstacked = []
            for i in range(5):
                #input_unstacked = tf.unstack(Inputs[i],axis=1)
                new_shape = (-1, Inputs[i].get_shape()[1]*Inputs[i].get_shape()[2])
                input_unstacked = tf.reshape(Inputs[i], tf.compat.v2.convert_to_tensor(new_shape))
                inputs_unstacked.append(input_unstacked)
                
            #print(inputs_unstacked)
            #print(len(inputs_unstacked))
            #print(len(inputs_unstacked[0]))
            #print("inputs_unstacked.get_shape() = ",inputs_unstacked[0].get_shape())
            
            outputs, states = tf.compat.v1.nn.static_rnn(lstm_cell1, inputs_unstacked, dtype=tf.float32)
                
            #apply lstm
            n_lstm2 = 500
            #lstm_cells2 = []
            #for i in range(5):
            #    lstm_cells2.append(tf.compat.v1.nn.rnn_cell.basicLSTMCell(n_lstm2, name="LSTM2_"+str(i)))
            
            lstm_cell2 = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(n_lstm2, name="LSTM2")            
            outputs2, states2 = tf.compat.v1.nn.static_rnn(lstm_cell2, outputs, dtype=tf.float32)
            
            #apply lstm
            n_lstm3 = 500
            lstm_cell3 = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(n_lstm3, name="LSTM3")
            outputs3, states3 = tf.compat.v1.nn.static_rnn(lstm_cell3, outputs2, dtype=tf.float32)
            
            #concatatnet
            net = tf.compat.v1.concat(outputs3, 1)
            
            #apply fully connceted
            n1 = 500
            W1 = tf.Variable(initializer(shape = np.array([n_lstm2*5, n1])) , name="W1")
            b1 = tf.Variable(initializer(shape = np.array([n1])), name="b1")
            net = tf.add(tf.matmul(net,W1),b1)
            net = tf.nn.relu(net)
            
            n2 = 500
            W2 = tf.Variable(initializer(shape = np.array([n1,n2])), name="W2")
            b2 = tf.Variable(initializer(shape = np.array([n2])), name="b2")
            net = tf.add(tf.matmul(net,W2), b2)
            net = tf.nn.relu(net)
            
            W3 = tf.Variable(initializer(shape = np.array([n2,self.output_dim])), name="W3")
            b3 = tf.Variable(initializer(shape = np.array([self.output_dim])), name="b3")
            net = tf.add(tf.matmul(net,W3), b3)
            
            self.output = net
            
            #define loss and loss        
            self.loss = tf.math.reduce_mean(tf.compat.v2.losses.mse(self.output, self.Label))
            
            self.init_op = tf.compat.v1.global_variables_initializer()
            
            self.optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = self.lr).minimize(self.loss)
                
    def build_model_3(self):
        tf.compat.v1.disable_eager_execution()
        with tf.device("/gpu:0"):
            
            #deifne inouts and labels
            self.Input1 = tf.compat.v1.placeholder(tf.float32, (None,)+self.input_dim, name="input1")
            self.Input2 = tf.compat.v1.placeholder(tf.float32, (None,)+self.input_dim, name="input2")
            self.Input3 = tf.compat.v1.placeholder(tf.float32, (None,)+self.input_dim, name="input3")
            self.Input4 = tf.compat.v1.placeholder(tf.float32, (None,)+self.input_dim, name="input4")
            self.Input5 = tf.compat.v1.placeholder(tf.float32, (None,)+self.input_dim, name="input5")
            
            Inputs = [self.Input1,self.Input2,self.Input3,self.Input4,self.Input5]
            
            self.Label = tf.compat.v1.placeholder(tf.float32, (None, self.output_dim), name="output")
            
            #flaten inputs
            inputs_flatten = []
            d1 = Inputs[0].get_shape()[1]*Inputs[0].get_shape()[2]
            print(Inputs[0].get_shape())
            new_shape = tf.compat.v1.convert_to_tensor((-1, d1))            
            for i in range(5):
                inputs_flatten.append(tf.reshape(Inputs[i], new_shape))
            
            nets = []
            
            W1_list = []
            b1_list = []
            n1 = 100
            print(new_shape.get_shape())
            initializer = tf.compat.v2.random_normal_initializer()
            
            for i in range(5):
                W1_list.append(tf.Variable(initializer(shape = np.array( (d1, n1) )), name="W1_"+str(i)))
                b1_list.append(tf.Variable(initializer(shape = np.array([n1])), name="b1_"+str(i)))
            
            for i in range(5):                
                nets.append( tf.nn.relu( tf.add(tf.matmul(inputs_flatten[i], W1_list[i]), b1_list[i]) ) )
            
            n2 = 50
            W2_list = []
            b2_list = []
            for i in range(5):
                W2_list.append(tf.Variable(initializer(shape = np.array( (n1, n2) )), name="W2_"+str(i)))
                b2_list.append(tf.Variable(initializer(shape = np.array([n2])), name="b2_"+str(i)))
            
            for i in range(5):
                nets[i] = tf.nn.relu( tf.add(tf.matmul(nets[i], W2_list[i]), b2_list[i]) )
            
            n3 = 1
            W3_list = []
            b3_list = []
            for i in range(5):
                W3_list.append(tf.Variable(initializer(shape = np.array( (n2, n3) )), name="W3_"+str(i)))
                b3_list.append(tf.Variable(initializer(shape = np.array([n3])), name="b3_"+str(i)))
            
            for i in range(5):
                nets[i] = tf.add(tf.matmul(nets[i], W3_list[i]), b3_list[i]) 
                
            net = tf.compat.v1.concat(nets,1)
            print("shape net = ",net.get_shape())
            
            self.output = net
            
            #define loss and loss  
            mse = tf.compat.v2.losses.mse(self.output, self.Label)
            print("mse shape = ",mse.get_shape())
            self.loss = tf.math.reduce_mean(mse)
            print("loss shape = ",self.loss.get_shape())
            
            self.init_op = tf.compat.v1.global_variables_initializer()
            
            self.optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = self.lr).minimize(self.loss)
    
    def build_model_4(self):
        
        tf.compat.v1.disable_eager_execution()
        with tf.device("/gpu:0"):
            
            #deifne inouts and labels
            self.Input1 = tf.compat.v1.placeholder(tf.float32, (None,)+self.input_dim, name="input1")
            self.Input2 = tf.compat.v1.placeholder(tf.float32, (None,)+self.input_dim, name="input2")
            self.Input3 = tf.compat.v1.placeholder(tf.float32, (None,)+self.input_dim, name="input3")
            self.Input4 = tf.compat.v1.placeholder(tf.float32, (None,)+self.input_dim, name="input4")
            self.Input5 = tf.compat.v1.placeholder(tf.float32, (None,)+self.input_dim, name="input5")
            
            Inputs = [self.Input1,self.Input2,self.Input3,self.Input4,self.Input5]
            
            self.Label = tf.compat.v1.placeholder(tf.float32, (None, self.output_dim), name="output")
            
            #apply lstm for each input
            n_lstm1 = 1000
            
            lstm_cell1 = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(n_lstm1, name="LSTM1")
            initializer = tf.compat.v2.random_normal_initializer()
            inputs_unstacked = []
            for i in range(5):
                new_shape = (-1, Inputs[i].get_shape()[1]*Inputs[i].get_shape()[2])
                input_unstacked = tf.reshape(Inputs[i], tf.compat.v2.convert_to_tensor(new_shape))
                inputs_unstacked.append(input_unstacked)
                
            
            outputs, states = tf.compat.v1.nn.static_rnn(lstm_cell1, inputs_unstacked, dtype=tf.float32)
                
            #apply lstm
            n_lstm2 = 500
            
            lstm_cell2 = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(n_lstm2, name="LSTM2")            
            outputs2, states2 = tf.compat.v1.nn.static_rnn(lstm_cell2, outputs, dtype=tf.float32)
            
            #apply lstm
            n_lstm3 = 500
            lstm_cell3 = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(n_lstm3, name="LSTM3")
            outputs3, states3 = tf.compat.v1.nn.static_rnn(lstm_cell3, outputs2, dtype=tf.float32)
            
            #apply lstm
            n_lstm4 = 500
            lstm_cell4 = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(n_lstm4, name = "LSTM4")
            outputs4, states4 = tf.compat.v1.nn.static_rnn(lstm_cell4, outputs3, dtype = tf.float32)
            
            #concatatnet
            net = tf.compat.v1.concat(outputs4, 1)
            
            #apply fully connceted
            n1 = 500
            W1 = tf.Variable(initializer(shape = np.array([n_lstm2*5, n1])) , name="W1")
            b1 = tf.Variable(initializer(shape = np.array([n1])), name="b1")
            net = tf.add(tf.matmul(net,W1),b1)
            net = tf.nn.relu(net)
            
            n2 = 500
            W2 = tf.Variable(initializer(shape = np.array([n1,n2])), name="W2")
            b2 = tf.Variable(initializer(shape = np.array([n2])), name="b2")
            net = tf.add(tf.matmul(net,W2), b2)
            net = tf.nn.relu(net)
            
            W3 = tf.Variable(initializer(shape = np.array([n2,self.output_dim])), name="W3")
            b3 = tf.Variable(initializer(shape = np.array([self.output_dim])), name="b3")
            net = tf.add(tf.matmul(net,W3), b3)
            
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
        
        
    
        
        
        