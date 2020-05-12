# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 14:32:42 2019

@author: chihe
"""

import numpy as np
import pandas as pd

if __name__ is '__main__':
    
    data_test=pd.read_csv("C:\\Users\\chihe\\Documents\\Course 2019 2020\\1. Foundations of Machine Learning\\Project\\Database\\interactions_test.csv")
    data_train=pd.read_csv("C:\\Users\\chihe\\Documents\\Course 2019 2020\\1. Foundations of Machine Learning\\Project\\Database\\interactions_train.csv")
    data_cross=pd.read_csv("C:\\Users\\chihe\\Documents\\Course 2019 2020\\1. Foundations of Machine Learning\\Project\\Database\\interactions_validation.csv")
    data=pd.concat([data_test, data_train, data_cross])
    
    users=data['u']
    id_recipe=data['i']
    rating=data['rating']
    
    users=np.asarray(users, dtype=int)
    id_recipe=np.asarray(id_recipe, dtype=int)
    rating=np.asarray(rating, dtype=int)
    
    del data_test, data_train, data_cross
    
    longueur=max(users)+1
    largeur=max(id_recipe)+1
    diag=len(users)
    
    matrix=np.zeros((longueur, largeur), dtype=np.uint8)
    i=0
    
    while(i<diag):
        m=users[i]
        n=id_recipe[i]
        matrix[m][n]=rating[i]
        i+=1
    
    #np.savetxt("matrix.txt", matrix, delimiter=',')











