import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

#%% IMport data to pandas data frame 
df = pd.read_csv("Bengaluru_House_Data.csv")
df.head
df.shape

#%% check data stats

df.groupby('area_type')['area_type'].agg('count')

#%% drop unwanted columns that has no impact in price prediction
df1 = df.drop(['area_type', 'society', 'balcony', 'availability'], axis = 'columns')
df1.head()

#%%find null values 
df1.isnull().sum()

# as null values are very less in number we drop null values
df2 = df1.dropna()
# check there are no null remaining 
df2.isnull().sum()

df2.shape

#%% I see there are different string values in size column

# find the unique values in size column
df2['size'].unique()

#having different non unifrom style of values of size column(bedrooms)
#removing text value from the column and keeping only numbers using lambda function
#adding the numerical values to new column called bhk

df2['bhk'] = df2['size'].apply(lambda x: int(x.split(' ')[0]))
df2.head
df2['bhk'].unique()

#%% chekc for unusal number of bedrooms houses

df2[df2.bhk>20]

# 2 records has more than 20 bedrooms. 
# the one recrd that has 43 bedrroms has only 2400 sq.ft area which is not possible

#%% now check the total_sqft column values

df2['total_sqft'].unique()

#%%
#we have values like "1133-1384" which is range instead of int/float value

#check for float value in the total_sqft clomun

#define function to calculate the range for values
def is_float(x):
    try:
        float(x)
    except:
        return False
    return True
#%%
#apply float fundtion to sqft column

df2[~df2['total_sqft'].apply(is_float)].head(15)

#%%

'''define function to convert range to its avg(float) and normal values 
to float and other unit vlaues to none'''

def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None


#%% Now apply the function to actual data frame.

df3 = df2.copy()
df3['total_sqft'] = df3['total_sqft'].apply(convert_sqft_to_num)
df3.head(15)

#manually validate weather the values are correct to confirm

#%% now lets do some feature engineering
# price per square foot is very inportant in realestate 
# which helps in finding outlier
# creating new column called price_per_sqft

df4 = df3.copy()
df4['price_per_sqft'] = df4['price']*100000/df4['total_sqft'] #in the given data set price in lakshs(*100000)
df4.head()

#%% now lets work on dimensionality reduction
# lets check for unique locations
len(df4['location'].unique())

# there are 1304 different location which refers to dimensionality curse(high dimension disadvantage)
# first remove extra spaces from location text column
df4.location = df4['location'].apply(lambda x: x.strip())

#now check the number of rows(properties) for each location
location_stats = df4.groupby('location')['location'].agg('count').sort_values(ascending=False)

print(location_stats)

#we found most of the location having less than 10 data points
#lets consider location having less than 10 data points as others

len(location_stats[location_stats<=10])

# now assign this to variable
location_stats_less_than_10 = location_stats[location_stats<=10]
print(location_stats_less_than_10) # unique locations <10 datapoints =1052 

#check total number of unique location
len(df4['location'].unique()) # unique location = 1293

#%% now change the location column values to others that has <10 values

df4.location = df4.location.apply(lambda x: 'others' if x in location_stats_less_than_10 else x)
#now check number of unique location values 
len(df4['location'].unique()) #new unique location value is 242

#%% 
'''Now lets check for otliers'''
#in general for 1000 sqft house ther will be 2 bedrooms
#that generally implies each bedroom genrally occupy 300 sqft
#now check the data that which is not satisfying the above requirement

df4[df4.total_sqft/df4.bhk<300].head()
len(df4[df4.total_sqft/df4.bhk<300]) #744 values
#hear we are checking the rows that are not having minimum 300sqft fro each bed room

#%%Now remove outliers from the data frame
#use nigate(~) function filter all the unwanted rows
df4.shape #(13246, 7)
df5 = df4[~(df4.total_sqft/df4.bhk<300)]
df5.shape #(12502, 7)

#%% check for sqft outliers
df5.price_per_sqft.describe()
#min value is 267.829813
#max value is 176470.588235
''' here min value is very lowthat is techincally not possible
where as max values is very high wich might possible to have such
a great price in prime locations.
let assume data is normally distributes where(68% data lies in with in mean and 1 std deviation)
now take data points that are with in 1std deviation to remove outliers
'''

#lets define a function to remove preice per sqft outliers

def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        std = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-std)) & (subdf.price_per_sqft<=(m+std))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out

#cerating new df by filtering all the price per sqft outlier

df6 = remove_pps_outliers(df5)
df6.shape #(10241, 7) removed aroun 2000 outliers 

#%%
'''I noticed in some cases property prices of 2bhk is more than 3bhk with same sqft and location
it might be because of it has extra eminities or may be from premium builder
let me visualize these points to understand this more
'''

#lets define a function to handel this ploting
def plot_scatter_chart(df,location):
    bhk2 = df[(df.location==location) & (df.bhk==2)]
    bhk3 = df[(df.location==location) & (df.bhk==3)]
    matplotlib.rcParams['figure.figsize'] =(15,10)
    plt.scatter(bhk2.total_sqft, bhk2.price, color = 'blue', label = '2Bhk', s=50)
    plt.scatter(bhk3.total_sqft, bhk3.price, marker ='+', color = 'green', label = '3bhk', s=50)
    plt.xlabel('Total square feet area')
    plt.ylabel('Price')
    plt.title('Location')
    plt.legend()
    
#now plot for our data frame
plot_scatter_chart(df6, 'Rajaji Nagar')
plot_scatter_chart(df6, 'Hebbal')

#%%
#we have few outliers, now lets write a function to remove those bhk outliers
def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
                bhk_stats[bhk] ={
                    'mean' : np.mean(bhk_df.price_per_sqft),
                    'std' : np.std(bhk_df.price_per_sqft),
                    'count' : bhk_df.shape[0]
                }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices, axis = 'index')

#apply the function to our data frame
df7 = remove_bhk_outliers(df6)
df7.shape #After reoving BHK outliers we have (7329, 7)
#check by visualizing the new data
plot_scatter_chart(df7,'Hebbal')

#%% Now lets visualise and see no of propertie per sqft area
#plot a histogram for better undersatanding

matplotlib.rcParams['figure.figsize'] = (20,10)
plt.hist(df7.price_per_sqft,rwidth =0.8)
plt.xlabel('Price per SQ feet')
plt.ylabel('Count')
#here we have most of our data lies in the range of 0-10000 price 
#looks like a normal distribution(bell shaped curve)

#%% Now lets see for bathroom outliers
print(df7.bath.unique())

df7[df7.bath>10] #there are few vaues that are having more than 10 bath rooms
# lets see using histogram
plt.hist(df7.bath,rwidth=0.8)
plt.xlabel("no of bathrooms")
plt.ylabel('Count')

#most of the count lies between 2-4 very few lies greater than 10
''' now lets consider properties having +2 bathrooms than no of bedrooms
consider them as outliers(imagine it is instructed by real estate manager)'''
df7[df7.bath>df7.bhk+2]
len(df7[df7.bath>df7.bhk+2]) #we have 4 properties with bathroom outliers

#lets remove bathroom outliers
df8 = df7[df7.bath<df7.bhk+2]
df8.shape  #we have (7251, 7) 

#%%
''' now our data set looks clean, lets remove unnecessar features
we can drop price_per_sqft as this column is created for identifying outliers
we can also drop size column as well because we have bhk clumns which is same
'''
df9 = df8.drop(['size','price_per_sqft'], axis='columns')
df9.head()

#%% 
'''before preparing a machine learning model we have a text column 
which we need to convert to numbers. we use one hot encoding for that'''

dummies = pd.get_dummies(df9.location)

#Now concatinate dummies and actual data frame
#in dummies dataframe we can drop last column which is other
df10 = pd.concat([df9,dummies.drop('others',axis='columns')],axis='columns')
df10.head()

#now drop text column(location)
df11 = df10.drop('location', axis = 'columns')
df11.head(3)
#%% Now lets split data 

df11.shape #(7251, 245)

# define independent variables
X = df11.drop('price', axis = 'columns')
X.head()

# define dependent variable
y = df11.price
y.head()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=10)


#%% Now perfrom linear regression

from sklearn.linear_model import LinearRegression

lr_clf = LinearRegression()
lr_clf.fit(X_train, y_train)
lr_clf.score(X_test, y_test) #accuracy = 84.5%

#lets perfrom shuffle split to chek the model perfromance

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits = 5, test_size = 0.2, random_state = 0)
print(cross_val_score(LinearRegression(),X,y, cv=cv))
# Shuffle Split accuracys [0.82430186 0.77166234 0.85089567 0.80837764 0.83653286]

#%% lets try other alogorithms for perfromance comparison
#lets use GridSearchCV api from sklearn
# lets use Lasso and decision regressions in grid search cv

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

#Now lets define a function for Grid search cv for lasso and decision tree models
#grid search cv not only perform multiplemodels, it also tunes the parameters calles
#hyper parameter tuning
# cv = cross validation

def find_best_model_using_gridsearchcv(X,y):
    algos = {
        'linear_regression':{
            'model':LinearRegression(),
            'params': {
                'fit_intercept': [True, False]
            }
        },
        'lasso' :{
            'model': Lasso(),
            'params':{
                'alpha': [1,2],
                'selection': ['random','cyclic']
            }
        },
        'decession_tree': {
            'model': DecisionTreeRegressor(),
            'params':{
                'criterion': ['mse','friedman_mse'],
                'splitter':['best','random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5,test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs = GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })
    return pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])

print(find_best_model_using_gridsearchcv(X, y))

'''
               model  best_score  bets_params
0  linear_regression    0.819001  {'fit_intercept': False}
1              lasso    0.687433  {'alpha': 1, 'selection': 'random'}
2     decession_tree    0.722947  {'criterion': 'friedman_mse', 'splitter': 'best'}
among 3 we have better perfromance in Linear regression
'''

#%%
#Lets build a Linear regression model with required independent features
X.columns

# Well we have converted the location column to encoded values 
# lets define function to use location and the map to encoded values for regression

def Predict_Price(location, sqft, bath, bhk,):
    loc_index = np.where(X.columns==location)[0][0]
    
    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >=0:
        x[loc_index] = 1
    
    return lr_clf.predict([x])[0]

#%% Predict price using our model
Predict_Price('Indira Nagar', 1500, 2, 3)

# for 1st Phase JP Nagar', 1000, 2, 2 = Predicted price 83.499 Lac
# for 1st Phase JP Nagar', 1500, 3, 3 = Predicted price 126.5 Lac
# for Indira Nagar', 1500, 3, 3       = Predicted price 224.29 Lac (Indira nagar is costly area)

#%% Our model building is completed 
'''
let save this model as pickle file to our local directory to use the model in 
python flask server while deploying
'''
import pickle
with open('House_Price_Prediction.Pickle', 'wb') as f:
    pickle.dump(lr_clf,f)
    
#%%
''' we also need Columns data as we have made many changes to them lets save 
them to Json file'''

import json
columns = {
    'data_columns' : [col.lower() for col in X.columns]
}
with open("columns.json","w") as f:
    f.write(json.dumps(columns))









