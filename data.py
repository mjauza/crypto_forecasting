# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import pyreadr
import os
import pickle

class Data():
    
    def __init__(self):
        pass
    
    def read_rdata(self,filename):
        data = pyreadr.read_r(filename)
        return data
    
    def check_time(self, dataframes):
        ts_sets = []
        lengths = []
        for df in dataframes:
            s = set(df["Time"])
            ts_sets.append(s)
            lengths.append(len(s))
        
        the_same = all([x == ts_sets[0] for x in ts_sets])
        len_same = all([x == lengths[0] for x in lengths])
        print("The same sets : ", the_same)
        print("The same lengths : ", len_same)
    
    def combine_rdata(self, directory):        
        onlyfiles = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]    
        r_files = [f for f in onlyfiles if f.split(".")[-1] == "RDATA"]
        print(r_files)
        dataframes = []
        symbols = []
        for f in r_files:
            filename = os.path.join(directory, f)
            data = self.read_rdata(filename)
            s = f.split("_")[1]
            dataframes.append(data["OHLC"])
            symbols.append(s)        
        
        combined_data = dataframes[0]
        for i in range(1,len(dataframes)):
            print(combined_data.columns)
            if i == 1:
                s1 = symbols[0]
                s2 = symbols[1]
                combined_data = combined_data.merge(dataframes[i], suffixes=("_"+s1,"_"+s2), left_on="Time", right_on="Time")
            else:
                left = "Time"
                right = "Time"
                si = symbols[i]
                combined_data = combined_data.merge(dataframes[i], suffixes = ("","_"+si), left_on=left, right_on=right)
        
        self.combined_data = combined_data
    
    def load_pickle_data(self, pkl_file="combined_data.pkl"):
        with open(pkl_file, "rb") as input:
            self.combined_data = pickle.load(input)


data = Data()
#filename = "OHLC_BCH_EUR.RDATA"
directory = "C:/Users/david/OneDrive/Dokumenti/crypto_data"
#data_var = data.read_rdata(filename)
#data.combine_rdata(directory)
data.load_pickle_data()

