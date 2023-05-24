import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

train = pd.read_csv('MobileTrain.csv')
train.head()
train.columns
round(train.describe(),1).T

def understand_data(train) :
    
    return(pd.DataFrame({"Datatype":train.dtypes,
                         "No of null values":train.isna().sum(),
                         "No of unique values":train.nunique(axis=0,dropna=True),
                         "Unique values": train.apply(lambda x: str(x.unique()),axis=0)}))
understand_data(train)

train.duplicated().sum()

#boxplot of each column
train.boxplot(figsize=(20,22))
plt.show()

num_col= ['battery_power','clock_speed','fc','int_memory', 'm_dep', 'mobile_wt','pc', 'px_height','px_width', 'ram', 'sc_h', 'sc_w', 'talk_time']
plt.figure(figsize=(15, 7))
for i in range(0, len(num_col)):
    plt.subplot(5, 3, i+1)
    sns.boxplot(x=train[num_col[i]],orient='v')
    plt.tight_layout()

# Min-max scaling
df = train.drop(['battery_power','clock_speed','fc','int_memory', 'm_dep', 'mobile_wt','n_cores','pc', 'px_height','px_width', 'ram', 'sc_h', 'sc_w', 'talk_time'],axis=1)
X = train.drop(['blue','dual_sim','four_g','three_g','touch_screen','wifi','price_range'], axis=1)
X

from sklearn.preprocessing import MinMaxScaler  #importing the required library for MinMax scaling
minmax = MinMaxScaler(feature_range=(0,1))  #creating instance
X= minmax.fit_transform(X)  #Performing MinMax scaling
X

X = pd.DataFrame(X)         
X = pd.DataFrame(X)         
X.columns = ['battery_power','clock_speed','fc','int_memory', 'm_dep', 'mobile_wt','n_cores','pc', 'px_height','px_width', 'ram', 'sc_h', 'sc_w', 'talk_time'] # Giving the columns their respective names
X

df = pd.concat([df,X],axis = 1)
df.head()

x = df.drop('price_range', axis=1)
y = df['price_range']

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest =train_test_split(x,y, test_size=0.25,random_state = 42)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
model_lr = lr.fit(xtrain,ytrain)
ypred_lr = model_lr.predict(xtest)
# checking the validation of the model
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score,accuracy_score,classification_report
print(classification_report(ytest,ypred_lr))
print('Accuracy score is:',accuracy_score(ytest,ypred_lr))
print('f1 score is:', f1_score(ytest, ypred_lr,average='weighted'))
al = accuracy_score(ytest,ypred_lr)
print(al)
pickle.dump(model_lr,open('model.pkl','wb'))