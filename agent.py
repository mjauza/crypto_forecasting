# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 17:47:02 2019

@author: david
"""

import numpy as np
import pandas as pd
from data import Data
from model import Model
import matplotlib.pyplot as plt
import logging
from datetime import datetime
from utils import Utils
from sklearn.preprocessing import StandardScaler

class Agent():
    def __init__(self, 
                 time_width = 10000,
                 time_step=1 ,
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
        
        d = self.data.combined_data.shape[0]
        n = round(d*(1-self.test_ratio))
        
        self.scale_data(n)
        self.define_generators()
        
        #define model
        self.input_dim = (self.time_width, 6)
        self.model = Model(self.input_dim,self.lr,batch_size=self.batch_size)
        #self.model.build_model(n1=self.n1,
        #                       s1=self.s1,
        #                       n2 = self.n2,
        #                       s2 = self.s2,
        #                       n_lstm1 = self.n_lstm1)
        
        #self.model.build_model_1()
        self.model.build_model_3()
        
        if not restore:            
            self.model.initialize_variables_and_sess()
            self.model.initialze_saver()
        else:
            #self.model.restore_latest_session()
            self.model.restore_latest_session1()
        
        self.utils = Utils()
        
        self.symbol_list = ["BCH", "ETH", "LTC", "XBT", "XRP"]
        
    
    def scale_data(self,max_train_index):
        X_train = self.data.combined_data.iloc[:max_train_index,:]
        X_train.drop("Time", 1, inplace=True)
        
        X_test = self.data.combined_data.iloc[max_train_index:, :]
        X_test.drop("Time", 1, inplace=True)
        
        self.scaler = StandardScaler()
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        
        self.scaled_data = pd.concat([X_train_scaled, X_test_scaled])
        self.scaled_data["Time"] = self.data.combined_data["Time"]
        
        self.pred_columns = ['Open_BCH', 'Open_ETH', 'Open', 'Open_XBT', 'Open_XRP']
        self.columns_scalar = X_train.columns
        
        
    def generator(self,min_index, max_index, time_width, time_step):
        """
        This function is a train generator which yields training sequences
        ARGS:
        max_index  : max index of dependent variables = labels
        time_width : time width of input sequence 
        """
        x_start_index = min_index
        x_end_index = x_start_index + time_width
        y_index = x_end_index - 1
        
        
        while y_index < max_index:
            #x = self.data.combined_data.iloc[x_start_index:x_end_index, :].copy()            
            x = self.scaled_data.iloc[x_start_index:x_end_index, :].copy()
            
            #y = self.data.combined_data.iloc[y_index,:].copy()
            y = self.scaled_data.iloc[y_index,:].copy()
            y_for_scaler = self.scaled_data.iloc[y_index:y_index+1,:].copy()
            y_original = self.scaler.inverse_transform(y_for_scaler.drop("Time",1))
            y_original = pd.DataFrame(y_original, columns=self.columns_scalar)
            #t = self.data.combined_data.iloc[y_index,:].copy()
            t = self.scaled_data.iloc[y_index,:].copy()
            
            x1 = x[['Open_BCH','High_BCH', 'Low_BCH', 'Close_BCH', 'Volume_BCH', 'Transactions_BCH']].values
            x2 = x[['Open_ETH','High_ETH', 'Low_ETH', 'Close_ETH','Volume_ETH', 'Transactions_ETH']].values
            x3 = x[['Open','High', 'Low', 'Close','Volume', 'Transactions']].values
            x4 = x[['Open_XBT','High_XBT', 'Low_XBT','Close_XBT', 'Volume_XBT', 'Transactions_XBT']].values
            x5 = x[['Open_XRP','High_XRP','Low_XRP', 'Close_XRP', 'Volume_XRP', 'Transactions_XRP']].values
            
            y = y[['Open_BCH', 'Open_ETH', 'Open', 'Open_XBT', 'Open_XRP']].values
            y_original = y_original[['Open_BCH', 'Open_ETH', 'Open', 'Open_XBT', 'Open_XRP']].values            
            t = t[["Time"]].values
            
            yield [x1,x2,x3,x4,x5], y, t , y_original[0,:]
            x_start_index += time_step
            x_end_index += time_step
            y_index += time_step
            
            if y_index >= max_index:
                print("reseting generator")                
                x_start_index = min_index
                x_end_index = x_start_index + time_width
                y_index = x_end_index - 1
    
    def define_generators(self):
        d = self.data.combined_data.shape[0]
        n = round(d*(1-self.test_ratio))
        
        self.train_gen = self.generator(min_index=0, max_index=n, time_width=self.time_width, time_step=self.time_step)
        self.num_steps_train = round( round((n - self.time_width) / self.time_step) / self.batch_size)
        
        self.test_gen = self.generator(min_index=n-self.time_width, max_index=d, time_width=self.time_width, time_step=self.time_step)
        self.num_steps_test = round((d - self.time_width - n) / self.time_step)
        
        self.train_gen_eval = self.generator(min_index=0, max_index=n, time_width=self.time_width, time_step=self.time_step)
        self.num_steps_train_eval = round((n - self.time_width) / self.time_step)
        
        
    
    def run_generator_n_times(self, gen , n):
        input1_batch = []
        input2_batch = []
        input3_batch = []
        input4_batch = []
        input5_batch = []
        y_batch = []
        for i in range(n):
            inputs, y, t, _ = next(self.train_gen)
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
            
    def define_point_for_scaler(self,y):
        row = pd.DataFrame(np.zeros((1,self.scaled_data.shape[1]-1)), columns=self.columns_scalar)
        row[self.pred_columns] = y
        return row
    
    def get_elements_scaled_array(self,arr, col_names):
        df  = pd.DataFrame(arr, columns=self.columns_scalar)
        return df[col_names]
    
    def unscale_y(self, y):
        #define point for scaler
        row = self.define_point_for_scaler(y)
        unscaled_row = self.scaler.inverse_transform(row)
        y = self.get_elements_scaled_array(unscaled_row, self.pred_columns)
        return y.values
    
    def train_model(self, num_epochs=10):
        
        f_date = datetime.now()
        fname = str(datetime(f_date.year, f_date.month, f_date.day, f_date.hour, f_date.minute)) + "_train.log"
        fname = fname.replace(":","-")
        path = "../crypto_results/log/"+fname
        
        self.train_logger = self.utils.get_logger("train_logger", path)
        
        
        for i in range(num_epochs):
            losses = []
            for s in range(self.num_steps_train):
                #print("step {} of {}".format(s,self.num_steps_train))
                
                inputs , y = self.run_generator_n_times(self.train_gen, self.batch_size)
                
                loss = self.model.update_model(inputs, y)
                losses.append(loss)
            avg_loss = np.mean(np.array(losses))
            str0 = "Epoch {} :  Average loss = {}".format(i,avg_loss)
            self.train_logger.info(str0)
            print(str0)
            self.model.step += 1
        
        self.model.save_model()
    
    def test_model(self):
        
        f_date = datetime.now()
        fname = str(datetime(f_date.year, f_date.month, f_date.day, f_date.hour, f_date.minute)) + "_test.log"
        fname = fname.replace(":","-")
        path = "../crypto_results/log/"+fname
        
        self.test_logger = self.utils.get_logger("test_logger", path)
        
        
        y_list = []
        y_pred_list = []
        y_errors_scaled = []
        for s in range(self.num_steps_test):
            inputs, y, t, y_original = next(self.test_gen)
            y_pred = self.model.predict(inputs)
            
            y_pred_unscaled = self.unscale_y(y_pred)
            
            y_error_scaled = np.mean( (y - y_pred)**2 )
            y_errors_scaled.append(y_error_scaled)
            y_list.append(y_original.tolist())
            y_pred_list.append(y_pred_unscaled[0].tolist())
        
        self.plot_results(y_list,y_pred_list)
        avg_error = np.mean(np.array(y_errors_scaled))
        str0 = "Averge error = {}".format(avg_error)
        self.test_logger.info(str0)
        return y_errors_scaled
    
    def evaluate_model(self, gen, num_steps):        
        y_list = []
        y_pred_list = []
        y_errors = []
        for s in range(num_steps):
            inputs, y, t, y_original = next(gen)
            y_pred = self.model.predict(inputs)
            
            y_pred_unscaled = self.unscale_y(y_pred)
            
            y_error = np.mean( (y - y_pred)**2 )
            y_errors.append(y_error)
            y_list.append(y_original.tolist())
            y_pred_list.append(y_pred_unscaled[0].tolist())
        
        self.plot_results(y_list,y_pred_list,ending="_train_results")
        
        
    
    def plot_results(self, y_list,y_pred_list, directory="../crypto_results/figures/", ending="_test_resuls"):
        y_np = np.array(y_list)
        y_pred_np = np.array(y_pred_list)
        print(y_np.shape)
        print(y_pred_np.shape)
        for i in range(y_np.shape[1]):
            y = y_np[:,i]
            y_pred = y_pred_np[:,i]
            print("All y_pred the same ",np.all(y_pred[3]==y_pred))
            plt.figure()
            plt.plot(y, label="real")
            plt.plot(y_pred, label="predicted")
            plt.legend()
            plt.title(self.symbol_list[i])
            filename = directory+self.symbol_list[i]+ending+".jpg"
            plt.savefig(filename)
            plt.close()
    
if __name__ == "__main__":
    agent = Agent(time_width = 100,
                  time_step=100,
                  lr = 0.001,
                  n1=100,
                  s1=10,
                  n2 = 10,
                  s2 = 1,
                  n_lstm1 = 20,
                  test_ratio = 0.2,
                  batch_size=512,
                  restore = False)
    agent.train_model(num_epochs=3)
    y_errors = agent.test_model()
    avg_error = np.mean(np.array(y_errors))
    print("Avergae error = {}".format(avg_error))
    
    agent.evaluate_model(agent.train_gen_eval, agent.num_steps_train_eval)
    
    agent.model.close_session()
    logging.shutdown()
            
            
        
            

            
            