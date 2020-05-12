#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 17:00:37 2019

@author: alexandreduval
"""

# Import libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Import datasets 
inter_train = pd.read_csv('interactions_train.csv')
inter_test = pd.read_csv('interactions_test.csv')
validation = pd.read_csv('interactions_validation.csv')
raw_inter = pd.read_csv('RAW_interactions.csv')
raw_recipes = pd.read_csv('RAW_recipes.csv')
pp_recipes = pd.read_csv('PP_recipes.csv')



### First study interaction datasets (x3)


# Display basic info about interactions datasets
print ("Rows     : " ,inter_train.shape[0]) # 698901 rows
print ("Columns  : " ,inter_train.shape[1]) #  6 col. 
inter_train.info() # Date is an object, rating a float and others and int. 
inter_train.describe() # More than 50% of the ratings are 5, mean = 4.57
print ("\nFeatures : \n" ,inter_train.columns.tolist()) # ['user_id', 'recipe_id', 'date', 'rating', 'u', 'i']
print ("\nMissing values :  ", inter_train.isnull().sum().values.sum()) # 0 missing values 
inter_train_bis = inter_train[inter_train["rating"].notnull()] # No rating is missing 
print ("\nUnique values :  \n",inter_train.nunique()) # 25076 users id and 160901 id. 6 ratings. 
print(inter_train['rating'].value_counts()) # 518568 of ratings are 5.0 ; 16957 ratings = 0 ==> Probably all recipes non rated by users. 
inter_train[inter_train['rating']== 0] # 16957 have a 0 rating

# Alternative way 
unique, counts = np.unique(inter_train['rating'], return_counts=True)
print('training set: ', dict(zip(unique, counts)))
 
# Some stats about user-recipe relation
print(inter_train['i'].value_counts()) # The most rated recipe has 1091 ratings. 
inter_train[inter_train['i']==99787]
print(inter_train['u'].value_counts()) # One user rated 6437 recipes (highest number)
inter_train[inter_train['u']==94]


## Way to structure data 
# Group by users
inter_train.groupby(['u'])
# Group dataset by users and date of recipe rating
inter_train.groupby(['u','date'])





####################### UNDERSTAND HOW IS SPLIT THE DATA (from raw data to rest)



### Check the relation of users & recipes of the training, test and validation (same or new ones from a set to another) 


# All users in the training set are in the test set 
count = []
users_test = []
users_train = []
for item in inter_test['u']:
    users_test.append(item)
for item in inter_train['u']:
    users_train.append(item)
set(users_test).issubset(users_train)   
    

# No recipe of the test set appears in the training set 
recipe_test = []
recipe_train =[]
for item in inter_test['i']:
    recipe_test.append(item)
for item in inter_train['i']: 
    recipe_train.append(item)
set(recipe_test).issubset(recipe_train) # False, it is not a subset
set(recipe_test).isdisjoint(recipe_train) # True, it is disjoint
set(recipe_test).intersection(recipe_train) # No intersection


# Validation and train now
validation.nunique() # Each user appears once. There are 6K recipe! 
recipe_val=[]
for item in validation['i']:
    recipe_val.append(item)

user_val=[]
for item in validation['u']:
    user_val.append(item)
set(user_val).issubset(users_train) # included


# Conclusion: we predict ratings of recipes that has never been seen before. Slight bias! 
# We would have prefered if ratings were taken at random into the test set. 
# We should therefore split the data ourselves 


# Problem spotted: most of the recipes has been rated less than 5 times.
# They won't add much values to our model and we may delete them at the beginning. 
print(inter_test['i'].value_counts()) # One recipe has been rated 4 times. 
inter_train[inter_train['i']==99787]
print(inter_test['u'].value_counts()) # One user has rated 1 recipe. 
inter_train[inter_train['u']==94]







### Study the difference of dimension between raw recipe and pp_recipe
# Compare them and draw differences


## Some basic info 
pp_recipes.describe()  # stats 
raw_recipes.describe()
pp_recipes.info() # only numbers 
raw_recipes.info() # mix between integers and objects


## Get an overview of all column name (to use them more easily afterwards)
pp_recipes.columns # col names 
raw_recipes.columns # Some columns do not appear anymore!
# Minutes it takes, submission date, tags, description, contributor_id, number of steps/ingredients...
# The rest was tokenised: name, ingredients, steps, technique, calorie level


## Get an overview of some column content. Everything has been tokenised. Only numbers! 
pp_recipes.head(1).T
pp_recipes.ingredient_tokens.head(10) # list of lists of numbers
raw_recipes.head(1).T 


## Investigate recipes. Big difference in number of recipes between the 2 datasets. 
pp_recipes.isnull().sum().values.sum() # 0 missing values 
pp_recipes['id'].value_counts() # 178265 recipes 
pp_recipes['i'].value_counts() # no pb, same number of values 
pp_recipes['i'].nunique() # Each recipe appears only once 

raw_recipes['id'].value_counts() # 231637 recipes. Much more than pp_recipes. 
raw_recipes['id'].nunique() # 231 637 as well. No dupplicate 
raw_recipes.isnull().sum().values.sum() # 4980 missing values 

# Identify missing values. 1 in name and the rest in description
bool_series = pd.isnull(raw_recipes["description"])
bool_series[bool_series == True].size # isolate the recipes that don't contain a description
raw_recipes[bool_series]  
missing_recipes = raw_recipes.id[bool_series].tolist() # recipe id of the recipes that contain a missing value 
set(missing_recipes).isdisjoint(pp_recipes.id)
a =set(missing_recipes).intersection(pp_recipes.id) # False. Recipes whose description is not included are not contained in the pp_recipes dataset
count =0
for el in a:
    count += 1
# 3954 elements whose description is missing are in the pp_recipes dataset. 
# As "description" is not used in pp_recipes, most of the missing recipes do not coincide with the missing values in the description column. 
missing_name = pd.isnull(raw_recipes.name)
missing_name[missing_name == True].size
val = raw_recipes.id[missing_name] # isolate the recipe_id containing a missing value for name
set(val).issubset(pp_recipes.id) # Not in pp_recipers. This observation was deleted. 



# Further investiation about missing values gave nothing. The whole data look nice! 
# The problem must be that these recipes don't have a rating. Check this hypothesis.

# For this, look at raw_interactions file. 
raw_inter.info()
raw_inter.describe()
raw_inter.columns
pd.isnull(raw_inter) # only 169 of reviews are missing, which does not explain the missing recipes! 
raw_inter.recipe_id.nunique()   # 231 637. As many as raw recipes. 



### NO IDEA WHERE MOST OF THESE DIFFERENCES IN DATASET DIMENSIONS ARISE 