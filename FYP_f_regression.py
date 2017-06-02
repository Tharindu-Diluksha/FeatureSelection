#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 14:47:42 2017
f_regression for fyp for feature selection
@author: tdiluksha
"""
import pandas as pd
import numpy as np
from sklearn.feature_selection import f_regression

#data file path
provided_file_path = "/media/tdiluksha/626047056046E001/Semester 07/FYP/Dev"
data_path = provided_file_path+'/dengue_data_20_MOH.csv'


df = pd.read_csv(data_path, index_col=[0,1,5]) #indexing moh.id,moh.name,week

# fill missing values
df.fillna(method='ffill', inplace=True) #forward fill method

# removing unwanted columns in features
df.drop('lat', axis=1, inplace=True)
df.drop('lon', axis=1, inplace=True)
df.drop('population', axis=1, inplace=True) #why population


# separate moh areas
mohId = 273
moh = df.loc[mohId] # this one include the totalcases use for y parameter in f_regression

# removing features for  f_regression
df.drop('cases',axis=1,inplace=True)
df.drop('mean.temp',axis=1,inplace=True)
df.drop('min.temp',axis=1,inplace=True)
df.drop('max.temp',axis=1,inplace=True)
df.drop('precipitation',axis=1,inplace=True)
df.drop('mobility.value',axis=1,inplace=True)
df.drop('mean.ndvi',axis=1,inplace=True)

for i in range (1,3):
    #drop 1 and 2 weeks laggin data features since 3 weeks are needed
    df.drop('cases.lag.'+str(i),axis=1,inplace=True)
    df.drop('mean.temp.lag.'+str(i),axis=1,inplace=True)
    df.drop('min.temp.lag.'+str(i),axis=1,inplace=True)
    df.drop('max.temp.lag.'+str(i),axis=1,inplace=True)
    df.drop('precipitation.lag.'+str(i),axis=1,inplace=True)
    df.drop('mob.lag.'+str(i),axis=1,inplace=True)
    df.drop('ndvi.lag.'+str(i),axis=1,inplace=True)
    
f_reg_moh = df.loc[mohId]
#print(df.columns)
result = f_regression(f_reg_moh,moh.cases) #calculating F-values and P-values

#write results to a csv file
import csv

count = 0
count_list = []
fval_list = []
pval_list = []
features_list = []

for fval in result[0]: # apped f_values to a list
    fval_list.append(fval)

for pval in result[1]: # append p_values , count and features to a list
    count_list.append(count+1)
    pval_list.append(pval)
    features_list.append(df.columns[count])
    count+=1
     
with open(provided_file_path+'/FeatureSelection/f_result_'+str(mohId)+'.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:
    wr = csv.writer(myfile)
    wr.writerow(("count", "f_val", "p_val","features"))
    wr.writerows(zip(count_list,fval_list,pval_list,features_list))
    
print("Finished Successfully: MohId: "+str(mohId))