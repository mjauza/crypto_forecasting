# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 16:38:56 2019

@author: david
"""

import logging

class Utils():
    
    def __init__(self):
        pass
    
    def get_logger(self, name, log_file, level=logging.INFO):
        formater = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        
        handler = logging.FileHandler(log_file)
        handler.setFormatter(formater)
        
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.addHandler(handler)
        
        return logger