#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import datasets
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score

import matplotlib.pyplot as plt


# In[3]:


from sklearn.datasets import load_wine
wine=load_wine()


# In[4]:


#Conver to pandas dataframe
data=pd.DataFrame(data=np.c_[wine['data'],wine['target']],columns=wine['feature_names']+['target'])


# In[5]:


#Check data with info function
data.info()


# In[6]:


data.head()


# In[7]:


data.tail()


# In[13]:


data.describe()


# In[9]:


data.shape


# In[10]:


# Search for missing, NA and null values)


(data.isnull() | data.empty | data.isna()).sum()


# ## Data analysis

# In[11]:


#Let's see the frequency of the variable target.
#Convert variable to categorical.
data.target=data.target.astype('int64').astype('category')

#Frequency.
freq=data['target'].value_counts()

freq


# In[12]:


#Let's check graphically.
freq.plot(kind='bar')


# In[14]:


#Let's show the histograms of the variables alcohol, magnesium y color_intensity.
#Histogramas
data[['alcohol','magnesium','color_intensity']].hist()


# ## EDA

# In[15]:


feats_to_explore = ['alcohol', 'magnesium', 'color_intensity']


# In[17]:


# Alcohol variable histograms.
x1 = data.loc[data.target==0, 'alcohol']
x2 = data.loc[data.target==1, 'alcohol']
x3 = data.loc[data.target==2, 'alcohol']

kwargs = dict(alpha=0.3,bins=25)

plt.hist(x1, **kwargs, color='g', label='Type 0')
plt.hist(x2, **kwargs, color='b', label='Type 1')
plt.hist(x3, **kwargs, color='r', label='Type 2')
plt.gca().set(title='Alcohol Frequency by type of wine ', ylabel='Frequency')

plt.legend();


# In[18]:


#Color_intensity histograms

x1 = data.loc[data.target==0, 'color_intensity']
x2 = data.loc[data.target==1, 'color_intensity']
x3 = data.loc[data.target==2, 'color_intensity']

kwargs = dict(alpha=0.3,bins=25)

plt.hist(x1, **kwargs, color='g', label='Type 0')
plt.hist(x2, **kwargs, color='b', label='Type 1')
plt.hist(x3, **kwargs, color='r', label='Type 2')
plt.gca().set(title='Frequency of color intensity by type of wine', ylabel='Frequency')

plt.legend();


# In[19]:


#Magnesium histograms

x1 = data.loc[data.target==0, 'magnesium']
x2 = data.loc[data.target==1, 'magnesium']
x3 = data.loc[data.target==2, 'magnesium']

kwargs = dict(alpha=0.3,bins=25)

plt.hist(x1, **kwargs, color='g', label='Type 0')
plt.hist(x2, **kwargs, color='b', label='Type 1')
plt.hist(x3, **kwargs, color='r', label='Type 2')
plt.gca().set(title='Magnesium frequency by type of wine', ylabel='Frequency')

plt.legend();


# In[21]:


#Alcohol histograms with the mean and the standard deviation.

x1 = data.loc[data.target==0, 'alcohol']
x2 = data.loc[data.target==1, 'alcohol']
x3 = data.loc[data.target==2, 'alcohol']

kwargs = dict(alpha=0.3,bins=25)

plt.hist(x1, **kwargs, color='g', label='Type 0'+  str("{:6.2f}".format(x1.std())))
plt.hist(x2, **kwargs, color='b', label='Type 1'+  str("{:6.2f}".format(x2.std())))
plt.hist(x3, **kwargs, color='r', label='Type 2'+  str("{:6.2f}".format(x3.std())))
plt.gca().set(title='Frequency of alcohol by type of wine', ylabel='Frequency')
plt.axvline(x1.mean(), color='g', linestyle='dashed', linewidth=1)
plt.axvline(x2.mean(), color='b', linestyle='dashed', linewidth=1)
plt.axvline(x3.mean(), color='r', linestyle='dashed', linewidth=1)
plt.legend();


# In[22]:


#color_intensity histograms with the mean and the standard deviation..


x1 = data.loc[data.target==0, 'color_intensity']
x2 = data.loc[data.target==1, 'color_intensity']
x3 = data.loc[data.target==2, 'color_intensity']

kwargs = dict(alpha=0.3,bins=25)

plt.hist(x1, **kwargs, color='g', label='Type 0'+  str("{:6.2f}".format(x1.std())))
plt.hist(x2, **kwargs, color='b', label='Type 1'+  str("{:6.2f}".format(x2.std())))
plt.hist(x3, **kwargs, color='r', label='Type 2'+  str("{:6.2f}".format(x3.std())))
plt.gca().set(title='Color intensity frequency by type of wine', ylabel='Frequency')
plt.axvline(x1.mean(), color='g', linestyle='dashed', linewidth=1)
plt.axvline(x2.mean(), color='b', linestyle='dashed', linewidth=1)
plt.axvline(x3.mean(), color='r', linestyle='dashed', linewidth=1)
plt.legend();


# In[23]:


#magnesium histograms with the mean and the standard deviation..


x1 = data.loc[data.target==0, 'magnesium']
x2 = data.loc[data.target==1, 'magnesium']
x3 = data.loc[data.target==2, 'magnesium']

kwargs = dict(alpha=0.3,bins=25)

plt.hist(x1, **kwargs, color='g', label='Type 0'+  str("{:6.2f}".format(x1.std())))
plt.hist(x2, **kwargs, color='b', label='Type 1'+  str("{:6.2f}".format(x2.std())))
plt.hist(x3, **kwargs, color='r', label='Type 2'+  str("{:6.2f}".format(x3.std())))
plt.gca().set(title='Magnesium frequency by type of wine', ylabel='Frequency')
plt.axvline(x1.mean(), color='g', linestyle='dashed', linewidth=1)
plt.axvline(x2.mean(), color='b', linestyle='dashed', linewidth=1)
plt.axvline(x3.mean(), color='r', linestyle='dashed', linewidth=1)
plt.legend();


# In[24]:


#Correlation table
df=data[['alcohol','magnesium','color_intensity']]
df.corr()


# In[25]:


#scatter plots
df=data[['alcohol','magnesium','color_intensity','target']]
sns.pairplot(df,hue='target')


# ## Dimensionality reduction

# In[29]:


#Import standardscaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#Remove target columns.
x = data.loc[:,data.columns != 'target'].values
y = data.loc[:,['target']].values


# In[30]:


#Scale the data
x= pd.DataFrame(StandardScaler().fit_transform(x))
y=pd.DataFrame(y)
# Create PCA object.
pca = PCA(n_components=2)


# In[31]:


#Run PCA.
pComp=pca.fit_transform(x)

principalDf = pd.DataFrame(data = pComp
             , columns = ['PC 1', 'PC 2'])

principalDf.head()


# In[32]:


# Join again the target variable

finalDf = pd.concat([principalDf, data[['target']]], axis = 1)
finalDf.head()


# In[33]:


# Show the graphics.
fig = plt.figure(figsize = (10,10))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('PCA', fontsize = 20)
targets = [0.0, 1.0, 2.0]
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'PC 1']
               , finalDf.loc[indicesToKeep, 'PC 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()


# In[35]:


#Use same variables as in the previous point, they are already standarized
# Create TSNE object.
from sklearn.manifold import TSNE
X_embedded = TSNE(n_components=2,perplexity=15,random_state=42).fit_transform(x)

tsneDf = pd.DataFrame(data = X_embedded
             , columns = ['PC 1', 'PC 2'])

tsneDf.head()


# In[36]:


# Join the target variable

ftnseDf = pd.concat([tsneDf, data[['target']]], axis = 1)
ftnseDf.head()


# In[37]:


# Show the graphic.
fig = plt.figure(figsize = (10,10))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('TSNE', fontsize = 25)
targets = [0.0, 1.0, 2.0]
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = ftnseDf['target'] == target
    ax.scatter(ftnseDf.loc[indicesToKeep, 'PC 1']
               , ftnseDf.loc[indicesToKeep, 'PC 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()


# In[38]:


# Split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33,random_state=42)

X_train.shape


# In[39]:


X_test.shape


# In[41]:


#Create the classifier.
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=10, random_state=42)

clf.fit(X_train,y_train.values.ravel())


# In[43]:


#Apply cross validation to evaluate the results.
from sklearn.model_selection import cross_val_score
scores=cross_val_score(clf,X_train,y_train.values.ravel(),cv=5)
scores


# In[44]:


#Calculate the mean and the standard deviation of the validation
print("Mean: %0.2f ; Standard Dev.: %0.2f)" % (scores.mean(), scores.std()))


# In[45]:


#Apply PCA.

# Create PCA object.
pca = PCA(n_components=2)

#Apply PCA on training data
pComp=pca.fit_transform(X_train)

#Run PCA


principalDf = pd.DataFrame(data = pComp
             , columns = ['PC 1', 'PC 2'])

principalDf.head()


# In[46]:


#Create the classifier

pcaclf=RandomForestClassifier(n_estimators=10,random_state=42)

pcaclf.fit(principalDf,y_train.values.ravel())


# In[47]:


#Apply cross validation
scores=cross_val_score(pcaclf,principalDf,y_train.values.ravel(),cv=5)
scores


# In[48]:


#Mean and standard deviation of the validation.
print("Mean: %0.2f ; Standard dev.: %0.2f)" % (scores.mean(), scores.std()))


# In[49]:


#Run TSNE.


X_embedded = TSNE(n_components=2,perplexity=15).fit_transform(X_train)


tsneDf = pd.DataFrame(data = X_embedded
             , columns = ['PC 1', 'PC 2'])
tsneDf.head()


# In[50]:


#Create the classifier

tclf=RandomForestClassifier(n_estimators=10, random_state=42 )

tclf.fit(tsneDf,y_train.values.ravel())


# In[51]:


#Apply cross validation
scores=cross_val_score(tclf,tsneDf,y_train.values.ravel(),cv=5)
scores


# In[52]:


#Calculate mean and standard deviation of the validation
print("Mean: %0.2f ; Standard dev.: %0.2f)" % (scores.mean(), scores.std()))


# In[53]:


#Let's transform test data

PCA_test=pca.transform(X_test)

pcaTestDf = pd.DataFrame(data = PCA_test
             , columns = ['PC 1', 'PC 2'])

pcaTestDf.shape


# In[54]:


prediction=pcaclf.predict(pcaTestDf)
prediction


# In[56]:


#Cross validation and metrics.
from sklearn.metrics import accuracy_score
acc_score=accuracy_score(y_test,prediction)
acc_score


# In[58]:


#We get a 98% accuracy, let's see confussion matrix.
from sklearn.metrics import confusion_matrix
conf_matrix=confusion_matrix(y_test,prediction)
conf_matrix


# In[59]:


# Using n_estimators
from matplotlib.legend_handler import HandlerLine2D

n_estimators = [1, 2, 4, 8, 16, 32, 64, 100, 200]
train_results = []
test_results = []
#Save precision data in arrays in order to show the graphic.

for estimator in n_estimators:
   clf = RandomForestClassifier(n_estimators=estimator,random_state=42)
   clf.fit(X_train,y_train.values.ravel())
   pred_train = clf.predict(X_train)
   acc_score_train = accuracy_score(y_train,pred_train)
   train_results.append(acc_score_train)
   pred_test = clf.predict(X_test)
   acc_score_test=accuracy_score(y_test,pred_test)
   test_results.append(acc_score_test)

line1, = plt.plot(n_estimators, train_results, 'b', label='Train accuracy')
line2, = plt.plot(n_estimators, test_results, 'r', label='Test accuracy')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('Accuracy')
plt.xlabel('n_estimators')
plt.show()


# In[60]:


#Continue with max_depth

max_depths = np.linspace(1, 32, 32, endpoint=True)
train_results = []
test_results = []

#Save precision data in arrays in order to show the graphic
for max_depth in max_depths:
   clf = RandomForestClassifier(n_estimators=4,max_depth=max_depth,random_state=42)
   clf.fit(X_train,y_train.values.ravel())
   pred_train = clf.predict(X_train)
   acc_score_train = accuracy_score(y_train,pred_train)
   train_results.append(acc_score_train) 
   pred_test = clf.predict(X_test)
   acc_score_test=accuracy_score(y_test,pred_test)
   test_results.append(acc_score_test)


line1, = plt.plot(max_depths, train_results, 'b', label='Train accuracy')
line2, = plt.plot(max_depths, test_results, 'r', label='Test accuracy')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('Accuracy')
plt.xlabel('max_depths')
plt.show()


# In[61]:


#Finally, min_samples_split

min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)
test_results = []
train_results = []
#Save precision data in arrays in order to show the graphic

for min_samples_split in min_samples_splits:
   clf = RandomForestClassifier(n_estimators=4,max_depth=2,min_samples_split=min_samples_split,random_state=42)
   clf.fit(X_train,y_train.values.ravel())
   pred_train = clf.predict(X_train)
   acc_score_train = accuracy_score(y_train,pred_train)
   train_results.append(acc_score_train) 
   pred_test = clf.predict(X_test)
   acc_score_test=accuracy_score(y_test,pred_test)
   test_results.append(acc_score_test)

line1, = plt.plot(min_samples_splits, train_results, 'b', label='Train accuracy')
line2, = plt.plot(min_samples_splits, test_results, 'r', label='Test accuracy')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('Accuracy')
plt.xlabel('min_samples_splits')
plt.show()


# In[ ]:




