#!/usr/bin/env python
# coding: utf-8

# In[24]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
raw_mail_data = pd.read_csv('spam.csv')
print (raw_mail_data)


# In[25]:


mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)),'')


# In[26]:


mail_data.head()


# In[27]:


mail_data.shape


# In[28]:


mail_data.loc[mail_data['Category'] == 'spam', 'Category',] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category',] = 1


# In[29]:


X = mail_data['Message']
Y = mail_data['Category']


# In[30]:


print (X)


# In[31]:


print (Y)


# In[32]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)


# In[33]:


print (X.shape)
print (X_train.shape)
print (X_test.shape)


# In[34]:


feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english') 
feature_extraction = TfidfVectorizer(lowercase = True)

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')


# In[35]:


print(X_train)

print(X_train_features)


# In[36]:


model = LogisticRegression()


# In[37]:


model.fit(X_train_features, Y_train)


# In[38]:


prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)


# In[39]:


print('Accuracy on training data : ', accuracy_on_training_data)


# In[40]:


prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)


# In[41]:


print('Accuracy on test data : ', accuracy_on_test_data)


# In[42]:


input_mail = ["I've been searching for the right words to thank you for this breather. I promise i wont take your help for granted and will fulfil my promise. You have been wonderful and a blessing at all times"]

# convert text to feature vectors
input_data_features = feature_extraction.transform(input_mail)

# making prediction

prediction = model.predict(input_data_features)
print(prediction)


if (prediction[0]==1):
  print('Ham mail')

else:
  print('Spam mail')


# In[43]:


input_mail = ["Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"]

# convert text to feature vectors
input_data_features = feature_extraction.transform(input_mail)

# making prediction

prediction = model.predict(input_data_features)
print(prediction)


if (prediction[0]==1):
  print('Ham mail')

else:
  print('Spam mail')


# In[ ]:




