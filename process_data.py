import pyspark
from pyspark import SparkContext
import os
import sys
import json
import time
import itertools
import random
import csv
import numpy as np
import pandas as pd
import xgboost
from xgboost import XGBRegressor
import math
from sklearn.linear_model import LinearRegression

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
sc=SparkContext('local[*]','task1')
sc.setLogLevel('ERROR')

tip_RDD=sc.textFile('tip.json').map(json.loads)
#tip_features=tip_RDD.map(lambda business:)
photo_RDD=sc.textFile('photo.json').map(json.loads)
#print(photo_RDD.collect()[:5])

def price_range(a,key):
    r=0
    if a:
        if key in a.keys():
            r=a[key]
    r=int(r)
    return r
def noise(a,key):
    r=0
    if a:
        if key in a.keys():
            if a[key]=='quiet':
                return 1
            elif a[key]=='average':
                return 2
            elif a[key]=='loud':
                return 3
            elif a[key]=='very_loud':
                return 4
    return 0
def attire(a,key):
    if a:
        if key in a.keys():
            if a[key]=='casual':
                return 1
            elif a[key]=='formal':
                return 2
            elif a[key]=='dressy':
                return 3
    return 0


business_RDD=sc.textFile('business.json').map(json.loads)
user_RDD=sc.textFile('user.json').map(json.loads)
#print(user_RDD)
#print(business_RDD.collect()[0])
business_features=business_RDD.map(lambda business:[(business['business_id']),(business['review_count'],business['stars'],price_range(business['attributes'],'RestaurantsPriceRange2'))]).collectAsMap()
#user_features=user_RDD.map(lambda user:[(user['user_id']),(user['review_count'],user['useful'],user['average_stars'])]).collectAsMap()
user_features=user_RDD.map(lambda user:[user['useful'],user['user_id']]).groupByKey().map(lambda x:(x[0],len(x[1]))).collect()
s=0
l=0
for i in user_features:
    s+=i[0]
    l+=i[1]
print(s/l)
x=business_RDD.map(lambda y:noise(y['attributes'],'NoiseLevel')).distinct().collect()

restaurant_attire=business_RDD.map(lambda x:(attire(x['attributes'],'RestaurantsAttire'),x['stars'])).groupByKey().map(lambda x:(x[0],sum(x[1])/len(x[1]))).collect()
print(restaurant_attire)
#print(business_RDD.collect()[:20])
#print(x)

#print(user_RDD.collect()[:10])

good_list=['good','great','excellent','friendly','best','professional','professionally','nice','polite','well','fantastic'
,'love','accessible','adoring','adoringly','advanced','yummy','fancy','amazing','comfort','comfortable','delicious','favorate']
bad_list=['bad','slow','slowest','messy','expensive','fail','delay','delayed','poor','poorly','sad','saddest']

def tip_comment(item):
    business=item['business_id']
    comment=item['text']
    rating=0
    for i in comment.split(' '):
        if i.lower().replace('!','').replace(',','').replace('.','') in good_list:
            rating+=1
        elif i.lower().replace('!','').replace(',','').replace('.','') in bad_list:
            rating-=1
    return (business,rating)

def photo_comment(item):
    business=item['business_id']
    comment=item['caption']
    rating=0
    for i in comment.split(' '):
        if i.lower().replace('!','').replace(',','').replace('.','') in good_list:
            rating+=1
        elif i.lower().replace('!','').replace(',','').replace('.','') in bad_list:
            rating-=1
    return (business,rating)
def photo_count(item):
    business=item['business_id']
    id=item['photo_id']
    return(business,id)
business_rating=tip_RDD.map(tip_comment).groupByKey().map(lambda user:(user[0],sum(user[1])))
photo_rating=photo_RDD.map(photo_comment).groupByKey().map(lambda business:(business[0],sum(business[1])))
photo_RDD=sc.textFile('photo.json').map(json.loads)
photo_count_dict=photo_RDD.map(photo_count).groupByKey().map(lambda business:(business[0],len(list(business[1])))).collectAsMap()
print(photo_count_dict)
#print(business_rating.collect()[:50])

def all_checks(d):
    return sum(d.values())

checkin_RDD=sc.textFile('checkin.json').map(json.loads)
total_checkin=checkin_RDD.map(lambda business:(business['business_id'],all_checks(business['time'])))
print(total_checkin.collect()[:5])
