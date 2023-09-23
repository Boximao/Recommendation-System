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
import ast
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def pearson_correlation(item1,item2):
    
    dict1=dict(business_user[item1])
    dict2=dict(business_user[item2])
    list1=set(dict1.keys())
    list2=set(dict2.keys())
    
    # get co-related users
    similar_items=list(list1&list2)
    
    if len(similar_items)==0:
        
        return 0
    else:
        numer=0
        denom1=0
        denom2=0
        s1=0
        s2=0
        for i in similar_items:
            s1+=float(dict1[i])
            s2+=float(dict2[i])
        avg1=s1/len(similar_items)
        avg2=s2/len(similar_items)
        #avg1=business_average[item1]
        #avg2=business_average[item2]
        for i in range(len(similar_items)):
            #num1=float(dict1[similar_items[i]])-business_average[item1]
            #num2=float(dict2[similar_items[i]])-business_average[item2]
            num1=float(dict1[similar_items[i]])-avg1
            num2=float(dict2[similar_items[i]])-avg2
            numer+=num1*num2
            denom1+=num1**2
            denom2+=num2**2
        if denom1==0 or denom2==0:
            if numer==0:
                return 1.0
            return 0
        else:
        #print(numer/(math.sqrt(denom1)*math.sqrt(denom2)))
            return min(numer/(math.sqrt(denom1)*math.sqrt(denom2)),1.0)

def item_prediction(line):
    user=line[0]
    business=line[1]
    
    
    if business_user.get(business,None)==None:
        # deal with cold start of item-based cf
        if user_average.get(business,None)==None:
            #print(2.5)
            #item_user_pred[string]=[3.0,0]
            return (3.0,0)

        else: 
            #print(user_average[user])
            #item_user_pred[string]=[0,0]
            return (user_average[user],0)
            
    #elif user_business.get(user,None)==None:
        
    #    return (business_average[business],0)
    else:
        user_rated=user_business[user]
        user_rated_dict=dict(user_rated)
        pearson_list=[]
        for key in user_rated_dict.keys():
            if key!=business:
                pearson=pearson_correlation(business,key)
                if pearson!=0:
                    pearson_list.append([business,key,pearson])
        pearson_list=list(filter(lambda x:x[2]>0.2,pearson_list))
        pearson_list.sort(key=lambda x:x[2],reverse=True)
        #
        #if len(pearson_list)>10:
        #    candidates=pearson_list[:10]
        #else: 
        candidates=pearson_list

        user_dict=dict(user_business[user])
        num=0
        denom=0
        for item in candidates:
            num+=float(user_dict[item[1]])*float(item[2])
            denom+=abs(float(item[2]))
            #print(user_dict[item[1]])
        if denom==0:
            prediction=0
        else:
            prediction=num/denom
        if len(candidates)>50:
            #item_user_pred[string]=[min(5,max(prediction,1)),len(candidates)]
            return (min(5,max(prediction,1)),len(candidates))
        else:
            prediction=len(candidates)*0.02*prediction+(50-len(candidates))*0.02*(user_average[user]+business_average[business])*0.5
            #item_user_pred[string]=[min(5,max(prediction,1)),len(candidates)]
            return (min(5,max(prediction,1)),len(candidates))

def price_range(a,key):
    r=0
    if a:
        if key in a.keys():
            r=a[key]
    r=int(r)
    return r

def noise(a,key):
    r=2.5
    if a:
        if key in a.keys():
            if a[key]=='quiet':
                r=1
            elif a[key]=='average':
                r= 2
            elif a[key]=='loud':
                r= 3
            elif a[key]=='very_loud':
                r= 4
    
    return r

def good_for_kids(a,key):
    r=0.5
    if a:
        if key in a.keys():
            if a[key]==True or a[key]=='True':
                r=1
            else:
                r=0
    return r

def reserve(a,key):
    r=0.5
    if a:
        if key in a.keys():
            if a[key]==True or a[key]=='True':
                r=1
            else:
                r=0
    return r

def tip_comment(item):
    business=item['business_id']
    comment=item['text']
    rating=0
    for i in comment.split(' '):
        if i.lower() in good_list:
            rating+=1
        elif i.lower() in bad_list:
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


def all_checks(d):
    return sum(d.values())

def parking(a,key):
    r=-1
    
    if a:
        if 'BusinessParking' in a.keys():
            x_dict=ast.literal_eval(a['BusinessParking'])
            if key in x_dict.keys():
                if x_dict[key]==True or x_dict[key]=='True':
                    r=1
                else:
                    r=0
    return r


def photo_count(item):
    business=item['business_id']
    id=item['photo_id']
    return(business,id)

def tip_count(item):
    business=item['business_id']
    id=item['text']
    return (business,id)

def generate_features(input_data):

    user_review_count=[]
    useful=[]
    user_avg_stars=[]
    business_review_count=[]
    business_stars=[]
    business_price=[]
    business_noise=[]
    business_kids=[]
    price_noise=[]
    price_rating=[]
    price_kid=[]
    reserve=[]
    attire=[]
    checkin=[]
    takeout=[]
    delivery=[]
    state=[]
    garage=[]
    street=[]
    validated=[]
    lot=[]
    valet=[]
    is_open=[]
    price_star=[]
    review_useful=[]
    star_useful=[]
    table_service=[]
    lat=[]
    long=[]
    photo_len=[]
    tip_len=[]
    #business_rating_photo=[]
    

    for user in input_data['user_id']:
        if user in user_features.keys():
            user_review_count.append(float(user_features[user][0]))
            useful.append(float(user_features[user][1]))
            user_avg_stars.append(float(user_features[user][2]))
            review_useful.append(float(user_features[user][1])/float(user_features[user][0]))
            star_useful.append(float(user_features[user][1])/(float(user_features[user][2])*float(user_features[user][0])))
        else:
            user_review_count.append(float(null_user_review_count))
            useful.append(float(null_user_useful))
            user_avg_stars.append(float(null_user_avg_stars))
            review_useful.append(float(null_user_useful)/float(null_user_review_count))
    for business in input_data['business_id']:
        if business in business_features.keys():
            business_review_count.append(float(business_features[business][0]))
            business_stars.append(float(business_features[business][1]))
            business_price.append(float(business_features[business][2]))
            business_noise.append(float(business_features[business][3]))
            business_kids.append(float(business_features[business][4]))
            reserve.append(float(business_features[business][5]))
            attire.append(float(business_features[business][6]))
            #takeout.append(float(business_features[business][7]))
            #delivery.append(float(business_features[business][8]))
            state.append(states.get(business[9],0))
            garage.append(float(business_features[business][10]))
            street.append(float(business_features[business][11]))
            validated.append(float(business_features[business][12]))
            lot.append(float(business_features[business][13]))
            valet.append(float(business_features[business][14]))
            is_open.append(float(business_features[business][15]))
            price_noise.append(float(business_features[business][2])*float(business_features[business][3]))
            price_kid.append(float(business_features[business][2])*float(business_features[business][4]))
            price_rating.append(float(business_features[business][2])*business_rating_dict.get(business,0))
            price_star.append(float(business_features[business][2])/float(business_features[business][1]))
            table_service.append(float(business_features[business][16]))
            lat.append(float(business_features[business][17]))
            long.append(float(business_features[business][18]))
            photo_len.append(photo_count_dict.get(business,0))
            tip_len.append(tip_count_dict.get(business,0))
            #star_review_count.append(float(business_features[business][0])/float(business_features[business][1]))
            #star_attire.append(float(business_features[business][6])*float(business_features[business][1]))
        else:
            business_review_count.append(float(null_business_review_count))
            business_stars.append(float(null_business_stars))
            #star_attire.append(float(null_business_stars)*float(business_features[business][6]))
            
            
        #business_rating_tip.append(business_rating_dict.get(business,0))  
        checkin.append(total_checkin.get(business,0))
        #business_rating_photo.append(photo_rating_dict.get(business,0))  
    df_train=pd.DataFrame({'user_review_count':user_review_count,'useful':useful,'user_avg_stars':user_avg_stars,'business_review_count':business_review_count,'business_stars':business_stars,'business_price':business_price,'business_kids':business_kids,
    'price_rating':price_rating,'reserve':reserve,'attire':attire,'checkin':checkin,'state':state,'price_noise':price_noise,'price_kid':price_kid,'garage':garage,'street':street,'validated':validated,'lot':lot,'valet':valet,'is_open':is_open,'price_star':price_star,
    'review_useful':review_useful,'star_useful':star_useful,'lat':lat,'photo_len':photo_len,'long':long})
    #'table_service':table_service

    return df_train


start=time.time()
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
sc=SparkContext('local[*]','task1')
sc.setLogLevel('ERROR')

# item based
rawdata=sc.textFile('yelp_train.csv')
rawRDD=rawdata.map(lambda line:line.split(',')).map(lambda line:(line[0],line[1],line[2]))
header=rawRDD.collect()[0]
cleanRDD=rawRDD.filter(lambda line:line!=header)
#print(cleanRDD.collect()[:10])
user_business=cleanRDD.map(lambda user:[user[0],(user[1],user[2])]).groupByKey().map(lambda item:(item[0],sorted(list(item[1])))).collectAsMap()
business_user=cleanRDD.map(lambda user:[user[1],(user[0],user[2])]).groupByKey().map(lambda item:(item[0],sorted(list(item[1])))).collectAsMap()
tip_RDD=sc.textFile('tip.json').map(json.loads)
photo_RDD=sc.textFile('photo.json').map(json.loads)

#print(user_business['vxR_YV0atFxIxfOnF9uHjQ'])
rawtest=sc.textFile('yelp_val.csv')
rawtestRDD=rawtest.map(lambda line:line.split(',')).map(lambda line:(line[0],line[1]))
header_test=header[0:2]
cleantestRDD=rawtestRDD.filter(lambda line:line!=header_test)
#print(cleantestRDD.collect()[:10])

business_average=cleanRDD.map(lambda user:(user[1],float(user[2]))).groupByKey().map(lambda item:(item[0],sum(list(item[1]))/len(list(item[1])))).collectAsMap()
user_average=cleanRDD.map(lambda user:(user[0],float(user[2]))).groupByKey().map(lambda item:(item[0],sum(list(item[1]))/len(list(item[1])))).collectAsMap()

#predicted_scores=cleanRDD.map(item_prediction)
predicted_scores_test_len=cleantestRDD.map(item_prediction)
#print(dict(itertools.islice(item_user_pred.items(), 10)))
predicted_scores_test=predicted_scores_test_len.map(lambda x:x[0])
#predicted_socres_test_list=list(predicted_scores_test.collect())
len_of_candidates=list(predicted_scores_test_len.map(lambda x:x[1]).collect())
#print(len_of_candidates[:50])

#model based
good_list=['good','great','excellent','friendly','best','professional','professionally','nice','polite','well','fantastic'
,'love','accessible','adoring','adoringly','advanced','yummy','fancy','amazing','comfort','comfortable','delicious','favorite',
'strong','worth','fresh','crisp','juicy','satisfying']
bad_list=['bad','slow','slowest','messy','expensive','fail','delay','delayed','poor','poorly','sad','saddest','lack',
'bland','burnt']
states={'AL':1,'AK':2,'AR':3,'AS':4,'CA':5,'CO':6,'CT':7,'DE':8,'DC':9,'FL':10,'GA':11,'GU':12,'HI':13,'ID':14,'IL':15,'IN':16,'IA':17,
'KS':18,'KY':19,'LA':20,'ME':21,'MD':22,'MA':23,'MI':24,'MN':25,'MS':26,'MO':27,'MT':28,'NE':29,'NV':30,'NH':31,'NJ':32,'NM':33,
'NY':34,'NC':35,'ND':36,'MP':37,'OH':38,'OK':39,'OR':40,'PA':41,'PR':42,'RI':43,'SC':44,'DS':45,'TN':46,'TX':47,'TT':48,'UT':49,
'VT':50,'VA':51,'VI':52,'WA':53,'WV':54,'WI':55,'WY':56}
business_RDD=sc.textFile('business.json').map(json.loads)
user_RDD=sc.textFile('user.json').map(json.loads)
#print(business_RDD.collect()[0])
business_rating_dict=tip_RDD.map(tip_comment).groupByKey().map(lambda business:(business[0],sum(business[1]))).collectAsMap()
photo_rating_dict=photo_RDD.map(photo_comment).groupByKey().map(lambda business:(business[0],sum(business[1]))).collectAsMap()
photo_count_dict=photo_RDD.map(photo_count).groupByKey().map(lambda business:(business[0],len(list(business[1])))).collectAsMap()
tip_count_dict=tip_RDD.map(tip_count).groupByKey().map(lambda business:(business[0],len(list(business[1])))).collectAsMap()
business_features=business_RDD.map(lambda business:[(business['business_id']),(business['review_count'],business['stars'],price_range(business['attributes'],'RestaurantsPriceRange2'),
noise(business['attributes'],'NoiseLevel'),good_for_kids(business['attributes'],'GoodForKids'),reserve(business['attributes'],'RestaurantsReservations'),attire(business['attributes'],'RestaurantsAttire'),
reserve(business['attributes'],'RestaurantsTakeOut'),reserve(business['attributes'],'RestaurantsDelivery'),business['state'],parking(business['attributes'],'garage'),parking(business['attributes'],'street'),
parking(business['attributes'],'validated'),parking(business['attributes'],'lot'),parking(business['attributes'],'valet'),business.get('is_open',-1),reserve(business['attributes'],'RestaurantsTableService'),business['latitude'],business['longitude'])]).collectAsMap()

user_features=user_RDD.map(lambda user:[(user['user_id']),(user['review_count'],user['useful'],user['average_stars'])]).collectAsMap()
checkin_RDD=sc.textFile('checkin.json').map(json.loads)
total_checkin=checkin_RDD.map(lambda business:(business['business_id'],all_checks(business['time']))).collectAsMap()

null_user_review_count=user_RDD.map(lambda user:user['review_count']).mean()
null_user_useful=0
null_user_avg_stars=user_RDD.map(lambda user:user['average_stars']).mean()
null_business_review_count=business_RDD.map(lambda business:business['review_count']).mean()
null_business_stars=business_RDD.map(lambda business:business['stars']).mean()
#predicted_scores_out=list(predicted_scores.collect())
predicted_scores_test_out=list(predicted_scores_test.collect())
#print(predicted_scores_test_out)

raw_train=pd.read_csv('yelp_train.csv')
#raw_train['predicted_stars']=list(predicted_scores.collect())
#print(raw_train.head())
train_features=generate_features(raw_train)
print(train_features.head())
train_stars=raw_train['stars']
raw_val=pd.read_csv('yelp_val.csv')
#raw_val['predicted_stars']=list(predicted_scores_test.collect())
val_features=generate_features(raw_val)
val_stars=raw_val['stars']

scaler=StandardScaler()
train_features=scaler.fit_transform(train_features)
val_features=scaler.transform(val_features)


model=XGBRegressor(learning_rate=0.18,max_depth=6,gamma=0.4)
model.fit(train_features,train_stars)
stars_pred=model.predict(val_features)

stars=[]
for j in range(len(predicted_scores_test_out)):
    stars.append(float(predicted_scores_test_out[j])*0.04+stars_pred[j]*0.96)
mse=0
for k in range(len(stars_pred)):
    mse+=((stars[k]-float(val_stars[k]))**2)
mse=mse/len(stars_pred)
rmse=math.sqrt(mse)
print(rmse)


error={'>=0 and <1':0,'>=1 and <2':0,'>=2 and <3':0,'>=3 and <4':0,'>=4':0}
for j in range(len(stars)):
    if stars[j]-float(val_stars[j])>=0 and stars[j]-float(val_stars[j])<1:
        error['>=0 and <1']+=1
    elif stars[j]-float(val_stars[j])>=1 and stars[j]-float(val_stars[j])<2:
        error['>=1 and <2']+=1
    elif stars[j]-float(val_stars[j])>=2 and stars[j]-float(val_stars[j])<3:
        error['>=2 and <3']+=1
    elif stars[j]-float(val_stars[j])>=3 and stars[j]-float(val_stars[j])<4:
        error['>=3 and <4']+=1
    elif stars[j]-float(val_stars[j])>=4:
        error['>=4']+=1
print(error)
#LR=LinearRegression()
#print(len(len_of_candidates))
#print(len(stars_pred))





# hybrid based on number of reviews
'''
stars=[]
for j in range(len(predicted_scores_test_out)):
    if len_of_candidates[j]>=50:
        stars.append(float(predicted_scores_test_out[j]))
    else:
        stars.append(float(predicted_scores_test_out[j])*0.06+stars_pred[j]*0.94)
    #stars.append(float(predicted_scores_test_out[j])*0.16+stars_pred[j]*0.84)
mse=0
for k in range(len(stars_pred)):
    mse+=((stars[k]-float(val_stars[k]))**2)
mse=mse/len(stars_pred)
rmse=math.sqrt(mse)
print(f'RMSE 50: {rmse}')
'''
'''
s_pred=[]
s_result=[]
for j in range(len(predicted_scores_test_out)):
    if len_of_candidates[j]<20:
        s_pred.append(float(stars_pred[j]))
        s_result.append(val_stars[j])

mse=0
for i in range(len(s_pred)):
    mse+=((s_pred[i]-float(s_result[i]))**2)
mse=mse/len(s_pred)
rmse=math.sqrt(mse)
print(f'RMSE for 20 on mL:{rmse}')


s_pred=[]
s_result=[]
for j in range(len(predicted_scores_test_out)):
    if len_of_candidates[j]<20:
        s_pred.append(float(predicted_scores_test_out[j]))
        s_result.append(val_stars[j])

mse=0
for i in range(len(s_pred)):
    mse+=((s_pred[i]-float(s_result[i]))**2)
mse=mse/len(s_pred)
rmse=math.sqrt(mse)
print(f'RMSE for 20 :{rmse}')
'''
'''
# test on only freq items
s_pred=[]
s_result=[]
for j in range(len(predicted_scores_test_out)):
    if len_of_candidates[j]>=50:
        s_pred.append(float(predicted_scores_test_out[j]))
        s_result.append(val_stars[j])

mse=0
for i in range(len(s_pred)):
    mse+=((s_pred[i]-float(s_result[i]))**2)
mse=mse/len(s_pred)
rmse=math.sqrt(mse)
print(f'RMSE for 50:{rmse}')

s_pred=[]
s_result=[]
for j in range(len(predicted_scores_test_out)):
    if len_of_candidates[j]>=40:
        s_pred.append(float(predicted_scores_test_out[j]))
        s_result.append(val_stars[j])

mse=0
for i in range(len(s_pred)):
    mse+=((s_pred[i]-float(s_result[i]))**2)
mse=mse/len(s_pred)
rmse=math.sqrt(mse)
print(f'RMSE for 40:{rmse}')

s_pred=[]
s_result=[]
for j in range(len(predicted_scores_test_out)):
    if len_of_candidates[j]>=30:
        s_pred.append(float(predicted_scores_test_out[j]))
        s_result.append(val_stars[j])

mse=0
for i in range(len(s_pred)):
    mse+=((s_pred[i]-float(s_result[i]))**2)
mse=mse/len(s_pred)
rmse=math.sqrt(mse)
print(f'RMSE for 30:{rmse}')
'''

'''
stars=[]
for j in range(len(predicted_scores_test_out)):
        stars.append(float(predicted_scores_test_out[j])*0.02+stars_pred[j]*(0.98))
mse=0

with open('task2_3output.csv','w',encoding='UTF-8') as f:
    f.write('user_id,business_id_2,stars\n')
    for i in range(len(raw_val)):
        f.write(str(raw_val['user_id'][i])+","+str(raw_val['business_id'][i])+","+str(stars[i])+'\n')

for k in range(len(stars_pred)):
    mse+=((stars[k]-float(val_stars[k]))**2)
mse=mse/len(stars_pred)
rmse=math.sqrt(mse)
print(rmse)
print(f'Duration: {end-start}')
'''
scores=[]
minx=10
for i in np.linspace(0,1,50):
    stars=[]
    for j in range(len(predicted_scores_test_out)):
        
        stars.append(float(predicted_scores_test_out[j])*i+stars_pred[j]*(1-i))
    mse=0
    for k in range(len(stars_pred)):
        mse+=((stars[k]-float(val_stars[k]))**2)
    mse=mse/len(stars_pred)
    rmse=math.sqrt(mse)
    scores.append([i,rmse])
print(scores)


'''
mse=0
for i in range(len(stars_pred)):
    mse+=((stars[i]-float(val_stars[i]))**2)
mse=mse/len(stars)
rmse=math.sqrt(mse)
print(rmse)

'''
'''
df_stars_pred=pd.DataFrame(stars_pred,columns=['stars_pred_by_model'])
df_stars_pred['item_based_prediction']=predicted_scores_test_out
df_stars_pred['length_of_candidates']=len_of_candidates
df_stars_pred['actual']=val_stars
X=df_stars_pred.drop(['actual'],axis=1)
y=df_stars_pred['actual']
print(df_stars_pred.head())
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=69)
LR_model=LinearRegression()
LR_model.fit(X_train,y_train)
y_pred_LR=LR_model.predict(X_test)
mse=mean_squared_error(y_pred_LR,y_test)
print(math.sqrt(mse))
'''
'''
mse=0
for k in range(len(y_pred_LR)):
    mse+=(y_pred_LR[k]-y_test[k])**2
mse=mse/len(y_pred_LR)
rmse=math.sqrt(mse)
print(rmse)
'''