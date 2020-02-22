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
                 restore = False,
                 restore_dir = "../crypto_results/checkpoint"):
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
        
        #self.model.build_model()
        #self.model.build_model_1()
        self.model.build_model_4()
        
        if not restore:            
            self.model.initialize_variables_and_sess()
            self.model.initialze_saver()
        else:
            #self.model.restore_latest_session()
            self.model.restore_latest_session1(restore_dir)
        
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
        y_index = x_end_index
        
        
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
                y_index = x_end_index
    
    def define_generators(self):
        d = self.data.combined_data.shape[0]
        n = round(d*(1-self.test_ratio))
        
        self.train_gen = self.generator(min_index=0, max_index=n, time_width=self.time_width, time_step=self.time_step)
        self.num_steps_train = round( round((n - self.time_width) / self.time_step) / self.batch_size)
        
        self.test_gen = self.generator(min_index=n-self.time_width, max_index=d, time_width=self.time_width, time_step=self.time_step)
        self.num_steps_test = round((d - self.time_width - n) / self.time_step)
        
        self.train_gen_eval = self.generator(min_index=0, max_index=n, time_width=self.time_width, time_step=self.time_step)
        self.num_steps_train_eval = round((n - self.time_width) / self.time_step)
        
    def get_file_in_arrf_format(self,filename, mode="train"):
        f = open(filename, "w")
        if mode == "train":
            gen = self.train_gen_eval
            num_steps = self.num_steps_train_eval
        elif mode=="test":
            gen = self.test_gen
            num_steps = self.num_steps_test
        else:
            f.close()
            raise Exception("Wrong mode")
        f.write("@RELATION cryptoTimeSeries\n")
        
        a1 = ['Open_BCH','High_BCH', 'Low_BCH', 'Close_BCH', 'Volume_BCH', 'Transactions_BCH']
        a2 =  ['Open_ETH','High_ETH', 'Low_ETH', 'Close_ETH','Volume_ETH', 'Transactions_ETH']
        a3 = ['Open','High', 'Low', 'Close','Volume', 'Transactions']
        a4 = ['Open_XBT','High_XBT', 'Low_XBT','Close_XBT', 'Volume_XBT', 'Transactions_XBT']
        a5 = ['Open_XRP','High_XRP','Low_XRP', 'Close_XRP', 'Volume_XRP', 'Transactions_XRP']
        atributes = [a1,a2,a3,a4,a5]
        for a in atributes:
            for i in range(len(a)):
                a_name = a[i]
                f.write("@ATTRIBUTE " + a_name + " timeseries\n")
        
        f.write("@ATTRIBUTE target timeseries\n")
        
        
        f.write("@DATA\n")
        for s in range(num_steps):
            inputs, y, t, y_original = next(gen)
            
            for x in inputs:
                for i in range(x.shape[1]):
                    f.write(str(x[:, i].tolist()) + ",") 
                    
            
            f.write(str(y.tolist()))
            f.write("\n")
        
        f.close()
    
    def get_file_in_arrf_format_1(self,filename, mode="train",max_num_steps=None):
        f = open(filename, "w")
        if mode == "train":
            gen = self.train_gen_eval
            num_steps = self.num_steps_train_eval
        elif mode=="test":
            gen = self.test_gen
            num_steps = self.num_steps_test
        else:
            f.close()
            raise Exception("Wrong mode")
            
        if (max_num_steps is not None) & (num_steps > max_num_steps):
            num_steps = max_num_steps
            
        print("using number of steps =", num_steps)
        
        f.write("@RELATION cryptoTimeSeries\n")
        
        a1 = ['Open_BCH','High_BCH', 'Low_BCH', 'Close_BCH', 'Volume_BCH', 'Transactions_BCH']
        a2 =  ['Open_ETH','High_ETH', 'Low_ETH', 'Close_ETH','Volume_ETH', 'Transactions_ETH']
        a3 = ['Open','High', 'Low', 'Close','Volume', 'Transactions']
        a4 = ['Open_XBT','High_XBT', 'Low_XBT','Close_XBT', 'Volume_XBT', 'Transactions_XBT']
        a5 = ['Open_XRP','High_XRP','Low_XRP', 'Close_XRP', 'Volume_XRP', 'Transactions_XRP']
        atributes = [a1,a2,a3,a4,a5]
        number_of_desc_att = 0
        for a in atributes:
            for i in range(len(a)):
                a_name = a[i]
                for j in range(self.time_width):
                    f.write("@ATTRIBUTE " + a_name + "_" +str(j) + " numeric\n")
                    number_of_desc_att += 1
        
        for i in range(5):
            name = self.pred_columns[i]
            f.write("@ATTRIBUTE " + name + " numeric\n")
            
                
        f.write("@DATA\n")
        for s in range(num_steps):
            inputs, y, t, y_original = next(gen)
            
            for x in inputs:
                for i in range(x.shape[1]):
                    for x_i in x[:, i].tolist():
                        f.write(str(x_i) + ",") 
            
            y_list = y.tolist()
            for j in range(5): 
                f.write(str(y_list[j]))
                if j != 4:
                    f.write(",")
            
            f.write("\n")
        
        f.close()
        print(number_of_desc_att)
            
            
    
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
        y_errors = []
        y_scaled_list = []
        y_scaled_pred_list = []
        up_moves_pred_list = []
        up_moves_real_list = []
        for s in range(self.num_steps_test):
            inputs, y, t, y_original = next(self.test_gen)
            y_pred = self.model.predict(inputs)
                        
            y_pred_unscaled = self.unscale_y(y_pred)
            
            if s == 0:
                y_prev = y_original
                y_pred_prev = y_pred_unscaled
            
            
            y_error_scaled = np.mean( (y - y_pred)**2 )
            y_error = np.mean((y_original - y_pred_unscaled)**2)
            
            y_errors_scaled.append(y_error_scaled)
            y_errors.append(y_error)
            
            y_list.append(y_original.tolist())
            y_pred_list.append(y_pred_unscaled[0].tolist())
            
            y_scaled_list.append(y.tolist())
            y_scaled_pred_list.append(y_pred[0].tolist())
            
            if s > 0:
                up_pred = (y_pred_unscaled[0] - y_pred_prev[0]) > 0
                up_real = (y_original - y_prev) > 0
                up_moves_pred_list.append(up_pred.tolist())
                up_moves_real_list.append(up_real.tolist())
                y_prev = y_original
                y_pred_prev = y_pred_unscaled
        
        self.plot_results(y_list,y_pred_list)
        self.plot_results(y_scaled_list, y_scaled_pred_list, ending = "_scaled_test_results")
        
        #avg_scaled_error = np.mean(np.array(y_errors_scaled))
        #avg_error = np.mean(np.array(y_errors))
        #accuracy = np.mean(np.array(up_moves_pred_list) == np.array(up_moves_real_list))
        
        str_1 = "Doing test evaluation"
        #str0 = "Averge MSE scaled error = {}".format(avg_scaled_error)
        #str1 = "Average MSE error = {}".format(avg_error)
        #str2 = "Accuracy = {}".format(accuracy)
        
        print(str_1)
        #print(str0)
        #print(str1)
        #print(str2)
        
        self.test_logger.info(str_1)
        #self.test_logger.info(str0)
        #self.test_logger.info(str1)
        #self.test_logger.info(str2)
        
        self.calculate_acc(up_moves_pred_list, up_moves_real_list, self.test_logger)
        self.calculate_rmse(y_list, y_pred_list, self.test_logger)
        
        return y_errors_scaled
    
    def evaluate_model(self, gen, num_steps):        
        y_list = []
        y_pred_list = []
        y_scaled_list = []
        y_scaled_pred_list = []
        y_scaled_errors = []
        y_errors = []
        up_moves_pred_list = []
        up_moves_real_list = []
        
        for s in range(num_steps):
            inputs, y, t, y_original = next(gen)
            y_pred = self.model.predict(inputs)            
            y_pred_unscaled = self.unscale_y(y_pred)
            
            if s == 0:
                y_prev = y_original
                y_pred_prev = y_pred_unscaled
            
            
            y_scaled_error = np.mean( (y - y_pred)**2 )
            y_scaled_errors.append(y_scaled_error)
            
            y_error = np.mean( (y_original - y_pred_unscaled)**2 )
            y_errors.append(y_error)
            
            y_list.append(y_original.tolist())
            y_pred_list.append(y_pred_unscaled[0].tolist())
            
            y_scaled_list.append(y.tolist())
            y_scaled_pred_list.append(y_pred[0].tolist())
            
            if s > 0:
                up_pred = (y_pred_unscaled[0] - y_pred_prev[0]) > 0
                up_real = (y_original - y_prev) > 0
                up_moves_pred_list.append(up_pred.tolist())
                up_moves_real_list.append(up_real.tolist())
                y_prev = y_original
                y_pred_prev = y_pred_unscaled
        
        #avg_error = np.mean(np.array(y_errors))
        #accuracy = np.mean( np.array(up_moves_pred_list) == np.array(up_moves_real_list) )
        
        str0 = "Doing training evaluation"
        #str1 = "Average MSE error = {}".format(avg_error)
        #str2 = "Accuracy = {}".format(accuracy)
        
        self.test_logger.info(str0)
        #self.test_logger.info(str1)
        #self.test_logger.info(str2)
        
        print(str0)
        #print(str1)
        #print(str2)
        
        self.calculate_acc(up_moves_pred_list, up_moves_real_list, self.test_logger)
        self.calculate_rmse(y_list, y_pred_list, self.test_logger)
        
        self.plot_results(y_list,y_pred_list,ending="_train_results")
        self.plot_results(y_scaled_list, y_scaled_pred_list, ending = "_scaled_train_results")
    
    def evaluate_clus_results(self, csv_filename):
        
        f_date = datetime.now()
        fname = str(datetime(f_date.year, f_date.month, f_date.day, f_date.hour, f_date.minute)) + "_clus_test.log"
        fname = fname.replace(":","-")
        path = "../crypto_results/log/"+fname
        
        self.clus_test_logger = self.utils.get_logger("clus_test_logger", path)
        
        
        df = pd.read_csv(csv_filename, header=0)
        cols = df.columns.values.tolist()
        new_cols = [c.strip() for c in cols]
        df.columns = new_cols
        
        num_steps = df.shape[0]

        original_model_names = ["Original-p-Open_BCH", "Original-p-Open_ETH", "Original-p-Open","Original-p-Open_XBT","Original-p-Open_XRP"]
        pruned_model_names = ["Pruned-p-Open_BCH","Pruned-p-Open_ETH","Pruned-p-Open","Pruned-p-Open_XBT","Pruned-p-Open_XRP"]
        
        df_original_model = df[original_model_names]
        df_pruned_model = df[pruned_model_names]
        
        del df
        
        y_list = []
        y_original_model_pred_list = []
        y_pruned_model_pred_list = []
        
        up_moves_pred_orig_model_list = []
        up_moves_pred_prune_model_list = []
        up_moves_real_list = []
        
        for i in range(num_steps):
            inputs, y, t, y_original = next(self.test_gen)
            y_pred_original_model = df_original_model.iloc[i,:].values
            y_pred_pruned_model = df_pruned_model.iloc[i,:].values
            y_pred_orig_model_unscaled = self.unscale_y(y_pred_original_model)
            y_pred_prune_model_unscaled = self.unscale_y(y_pred_pruned_model)
            
            if i == 0:
                y_prev = y_original
                y_pred_original_model_prev = y_pred_orig_model_unscaled
                y_pred_pruned_model_prev = y_pred_prune_model_unscaled
            
            y_list.append(y_original.tolist())
            y_original_model_pred_list.append(y_pred_orig_model_unscaled.tolist()[0])
            y_pruned_model_pred_list.append(y_pred_prune_model_unscaled.tolist()[0])
            
            if i > 0:
                up_pred_original_model = (y_pred_orig_model_unscaled - y_pred_original_model_prev) > 0
                up_pred_pruned_model = (y_pred_prune_model_unscaled - y_pred_pruned_model_prev) > 0
                up_real = (y_original - y_prev) > 0
                
                
                up_moves_pred_orig_model_list.append(up_pred_original_model.tolist()[0])
                up_moves_pred_prune_model_list.append(up_pred_pruned_model.tolist()[0])
                up_moves_real_list.append(up_real.tolist())
                
                y_prev = y_original
                y_pred_original_model_prev = y_pred_orig_model_unscaled
                y_pred_pruned_model_prev = y_pred_prune_model_unscaled
        
        
        self.clus_test_logger.info("Doing clus test evaluation")        
        
        self.clus_test_logger.info("Doing original model evaluation")
        self.calculate_acc(up_moves_pred_orig_model_list, up_moves_real_list, self.clus_test_logger)
        self.calculate_rmse(y_list, y_original_model_pred_list, self.clus_test_logger)
        
        self.clus_test_logger.info("Doing pruned model evaluation")
        self.calculate_acc(up_moves_pred_prune_model_list, up_moves_real_list, self.clus_test_logger)
        self.calculate_rmse(y_list, y_pruned_model_pred_list, self.clus_test_logger)
                       
        self.plot_results(y_list,y_original_model_pred_list,ending="_clus_test_results_original_model")
        self.plot_results(y_list,y_pruned_model_pred_list,ending="_clus_test_results_pruned_model")
        
    
    def get_column_names_forest(self, number):
        
        
        name1 = "Forest with {} trees-p-Open_BCH".format(number)
        name2 = "Forest with {} trees-p-Open_ETH".format(number)
        name3 = "Forest with {} trees-p-Open".format(number)
        name4 = "Forest with {} trees-p-Open_XBT".format(number)
        name5 = "Forest with {} trees-p-Open_XRP".format(number)
        
        return [name1, name2, name3, name4, name5]
        
    
    def get_evaluate_one_clus_result(self,df_results, num_steps, description, gen):
        y_list = []
        y_model_pred_list = []
        #y_pruned_model_pred_list = []
        
        up_moves_pred_list = []
        #up_moves_pred_prune_model_list = []
        up_moves_real_list = []
        
        for i in range(num_steps):
            inputs, y, t, y_original = next(gen)
            y_pred_model = df_results.iloc[i,:].values            
            y_pred_model_unscaled = self.unscale_y(y_pred_model)
            
            if i == 0:
                y_prev = y_original
                y_pred_model_prev = y_pred_model_unscaled
                
            
            y_list.append(y_original.tolist())
            y_model_pred_list.append(y_pred_model_unscaled.tolist()[0])
            
            if i > 0:
                up_pred_model = (y_pred_model_unscaled - y_pred_model_prev) > 0
                up_real = (y_original - y_prev) > 0
                
                
                up_moves_pred_list.append(up_pred_model.tolist()[0])
                up_moves_real_list.append(up_real.tolist())
                
                y_prev = y_original
                y_pred_model_prev = y_pred_model_unscaled
        
        self.clus_test_logger.info("Doing clus test evaluation")        
        
        self.clus_test_logger.info("Doing " + description + " model evaluation")
        self.calculate_acc(up_moves_pred_list, up_moves_real_list, self.clus_test_logger)
        self.calculate_rmse(y_list, y_model_pred_list, self.clus_test_logger)
        
                       
        self.plot_results(y_list,y_model_pred_list,ending="_clus_test_results_" + description + "_model")
        
        
        
    def evaluate_clus_forest_results(self, csv_filename, forest_num_list = [10,30,50,100]):
        
        f_date = datetime.now()
        fname = str(datetime(f_date.year, f_date.month, f_date.day, f_date.hour, f_date.minute)) + "_clus_test.log"
        fname = fname.replace(":","-")
        path = "../crypto_results/log/"+fname
        
        self.clus_test_logger = self.utils.get_logger("clus_test_logger", path)
        
        
        df = pd.read_csv(csv_filename, header=0)
        cols = df.columns.values.tolist()
        new_cols = [c.strip() for c in cols]
        df.columns = new_cols
        print(df.columns)
        
        num_steps = df.shape[0]

        #original_model_names = ["Original-p-Open_BCH", "Original-p-Open_ETH", "Original-p-Open","Original-p-Open_XBT","Original-p-Open_XRP"]
        #pruned_model_names = ["Pruned-p-Open_BCH","Pruned-p-Open_ETH","Pruned-p-Open","Pruned-p-Open_XBT","Pruned-p-Open_XRP"]
        
        #forest_10_names = self.get_column_names_forest(10)
        #forest_30_names = self.get_column_names_forest(30)
        #forest_50_names = self.get_column_names_forest(50)
        #forest_100_names = self.get_column_names_forest(100)
        
        
        
        for i in range(len(forest_num_list)):
            d = self.data.combined_data.shape[0]
            n = round(d*(1-self.test_ratio))
            test_gen = self.generator(min_index=n-self.time_width, max_index=d, time_width=self.time_width, time_step=self.time_step)
            
            forest_number = forest_num_list[i]
            forest_name = self.get_column_names_forest(forest_number)
            df_forest = df[forest_name]
            desc_forest = "random_forest_{}".format(forest_number)
            print("description = " + desc_forest)
            self.get_evaluate_one_clus_result(df_forest, df.shape[0], desc_forest, test_gen)
        
        #df_original_model = df[original_model_names]
        #df_pruned_model = df[pruned_model_names]
        
        
        
        
        
        
        
    
    def plot_results(self, y_list,y_pred_list, directory="../crypto_results/figures/", ending="_test_results"):
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
    
    def calculate_acc(self,up_moves_pred_list, up_moves_real_list, logger):
        up_moves_pred_np = np.array(up_moves_pred_list)
        up_moves_real_np = np.array(up_moves_real_list)
        print("up_moves_real_np.shape = ",up_moves_real_np.shape)
        print("up_moves_pred_np.shape = ",up_moves_pred_np.shape)
        for i in range(up_moves_pred_np.shape[1]):
            symbol = self.symbol_list[i]
            acc = np.mean(up_moves_pred_np[:,i] == up_moves_real_np[:,i])
            str0 = "Accuracy for " + symbol + " = " + str(acc)
            logger.info(str0)
            print(str0)
    
    def calculate_rmse(self, y_list, y_pred_list, logger):
        y_np = np.array(y_list)
        y_pred_np = np.array(y_pred_list)
        print("y_np.shape = ",y_np.shape)
        print("y_pred_np.shape = ",y_pred_np.shape)
        for i in range(y_np.shape[1]):
            symbol = self.symbol_list[i]
            mse = np.mean( (y_np[:,i] -  y_pred_np[:,i])**2 )
            rmse = np.sqrt(mse)
            str0 = "Symbol : " + symbol
            str1 = "MSE = {}".format(mse)
            str2 = "RMSE = {}".format(rmse)
            logger.info(str0)
            logger.info(str1)
            logger.info(str2)
            print(str0)
            print(str1)
            print(str2)
    
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
                  restore = True,
                  restore_dir="../crypto_results/checkpoint")
    #agent.train_model(num_epochs=30)
    #y_errors = agent.test_model()
    #avg_error = np.mean(np.array(y_errors))
    #print("Avergae error = {}".format(avg_error))
    
    #agent.evaluate_model(agent.train_gen_eval, agent.num_steps_train_eval)
    
    #agent.get_file_in_arrf_format(filename="test_data.arff", mode="test")
    #agent.get_file_in_arrf_format_1(filename="test_data_1.arff", mode="test",max_num_steps=500)
    #agent.get_file_in_arrf_format_1(filename="cryptoTimeSeries1.arff", mode="train")
    
    csv_filename = "../crypto_data/cryptoTimeSeries1_test_predictions.csv"
    #agent.evaluate_clus_results(csv_filename)
    agent.evaluate_clus_forest_results(csv_filename, forest_num_list = [200,300,500,1000])
    
    agent.model.close_session()
    logging.shutdown()
            
            
        
            

            
            