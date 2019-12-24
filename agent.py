# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 17:47:02 2019

@author: david
"""

import numpy as np
from data import Data
from model import Model
import matplotlib.pyplot as plt


class Agent():
    def __init__(self, 
                 time_width = 10000,
                 time_step=1000 ,
                 lr = 0.001,
                 n1=100,s1=10,
                 n2 = 10,
                 s2 = 1,
                 n_lstm1 = 20,
                 test_ratio = 0.2,
                 batch_size = 10,
                 restore = False):
        self.time_width = time_width
        self.lr = lr
        self.n1 = n1
        self.s1 = s1
        self.n2 = n2
        self.s2 = s2
        self.n_lstm1 = 20        
        self.test_ratio = test_ratio
        self.time_step = time_step
        self.batch_size = batch_size
        
        #define data and generators
        self.data = Data()
        self.data.load_pickle_data()
        self.define_generators()
        
        #define model
        self.input_dim = (self.time_width, 5)
        self.model = Model(self.input_dim,self.lr,batch_size=self.batch_size)
        self.model.build_model(n1=self.n1,
                               s1=self.s1,
                               n2 = self.n2,
                               s2 = self.s2,
                               n_lstm1 = self.n_lstm1)
        
        
        if not restore:            
            self.model.initialize_variables_and_sess()
            self.model.initialze_saver()
        else:
            self.model.restore_latest_session()
        
        
        self.symbol_list = ["BCH", "ETH", "LTC", "XBT", "XRP"]
        
        
        
    def generator(self,min_index, max_index, time_width, time_step):
        """
        This function is a train generator which yields training sequences
        ARGS:
        max_index  : max index of dependent variables = labels
        time_width : time width of input sequence 
        """
        y_index = time_width
        x_start_index = min_index
        x_end_index = x_start_index + time_width
        
        while y_index < max_index:
            x = self.data.combined_data.iloc[x_start_index:x_end_index, :].copy()
            
            x1 = x[['High_BCH', 'Low_BCH', 'Close_BCH', 'Volume_BCH', 'Transactions_BCH']].values
            x2 = x[['High_ETH', 'Low_ETH', 'Close_ETH','Volume_ETH', 'Transactions_ETH']].values
            x3 = x[['High', 'Low', 'Close','Volume', 'Transactions']].values
            x4 = x[['High_XBT', 'Low_XBT','Close_XBT', 'Volume_XBT', 'Transactions_XBT']].values
            x5 = x[['High_XRP','Low_XRP', 'Close_XRP', 'Volume_XRP', 'Transactions_XRP']].values
            y = self.data.combined_data.iloc[y_index,:].copy()
            y = y[['Open_BCH', 'Open_ETH', 'Open', 'Open_XBT', 'Open_XRP']].values
            t = self.data.combined_data.iloc[y_index,:].copy()
            t = t[["Time"]].values
            yield [x1,x2,x3,x4,x5], y, t
            x_start_index += time_step
            x_end_index += time_step
            y_index += time_step
            
            if y_index >= max_index:
                print("reseting generator")
                y_index = time_width
                x_start_index = 0
                x_end_index = x_start_index + time_width
    
    def define_generators(self):
        d = self.data.combined_data.shape[0]
        n = round(d*(1-self.test_ratio))
        self.train_gen = self.generator(min_index=0, max_index=n, time_width=self.time_width, time_step=self.time_step)
        self.num_steps_train = round( round((n - self.time_width) / self.time_step) / self.batch_size)
        self.test_gen = self.generator(min_index=n-self.time_width, max_index=d, time_width=self.time_width, time_step=self.time_step)
        self.num_steps_test = round( round((d - self.time_width - n) / self.time_step) / self.batch_size)
    
    
    def run_generator_n_times(self, gen , n):
        input1_batch = []
        input2_batch = []
        input3_batch = []
        input4_batch = []
        input5_batch = []
        y_batch = []
        for i in range(n):
            inputs, y, t = next(self.train_gen)
            #print("shape of inputs[0]", inputs[0].shape)
            input1_batch.append(inputs[0].tolist())
            input2_batch.append(inputs[1].tolist())
            input3_batch.append(inputs[2].tolist())
            input4_batch.append(inputs[3].tolist())
            input5_batch.append(inputs[4].tolist())
            y_batch.append(y.tolist())
        
        inputs_list = [
                np.array(input1_batch),
                np.array(input2_batch),
                np.array(input3_batch),
                np.array(input4_batch),
                np.array(input5_batch)
        ]
        #print("inputs_list[0].shape = ",inputs_list[0].shape)
        return inputs_list, np.array(y_batch)
            
    
    def train_model(self, num_epochs=10):
        
        for i in range(num_epochs):
            losses = []
            for s in range(self.num_steps_train):
                #print("step {} of {}".format(s,self.num_steps_train))
                
                inputs , y = self.run_generator_n_times(self.train_gen, self.batch_size)
                
                loss = self.model.update_model(inputs, y)
                losses.append(loss)
            avg_loss = np.mean(np.array(losses))
            print("Epoch {} :  Average loss = {}".format(i,avg_loss))
            self.model.step += 1
        
        self.model.save_model()
    
    def test_model(self):
        y_list = []
        y_pred_list = []
        y_errors = []
        for s in range(self.num_steps_test):
            inputs, y, t = next(self.test_gen)
            y_pred = self.model.predict(inputs)
            y_error = np.mean( (y - y_pred)**2 )
            y_errors.append(y_error)
            y_list.append(y.tolist())
            y_pred_list.append(y_pred[0].tolist())
        
        self.plot_results(y_list,y_pred_list)
        return y_errors
    
    def plot_results(self, y_list,y_pred_list):
        y_np = np.array(y_list)
        y_pred_np = np.array(y_pred_list)
        print(y_np.shape)
        print(y_pred_np.shape)
        for i in range(y_np.shape[1]):
            y = y_np[:,i]
            y_pred = y_pred_np[:,i]
            plt.figure()
            plt.plot(y, label="real")
            plt.plot(y_pred, label="predicted")
            plt.legend()
            plt.title(self.symbol_list[i])
            filename = "figures/"+self.symbol_list[i]+"_test_resuls.jpg"
            plt.savefig(filename)
            plt.close()
    
if __name__ == "__main__":
    agent = Agent(time_width = 1000,
                  lr = 0.001,
                  n1=100,
                  s1=10,
                  n2 = 10,
                  s2 = 1,
                  n_lstm1 = 20,
                  test_ratio = 0.2,
                  batch_size=10,
                  restore = False)
    agent.train_model(num_epochs=300)
    y_errors = agent.test_model()
    avg_error = np.mean(np.array(y_errors))
    print("Avergae error = {}".format(avg_error))
    agent.model.close_session()
            
            
        
            

            
            