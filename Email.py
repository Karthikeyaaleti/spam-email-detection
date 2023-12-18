#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[4]:


import os
for dirname, _, filenames in os.walk('input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


data=pd.read_csv('spam.csv')
data


# In[ ]:


data.columns


# In[ ]:


data.info()


# In[ ]:


data.isna().sum()


# In[ ]:


data['Spam']=data['Category'].apply(lambda x:1 if x=='spam' else 0)
data.head(5)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(data.Message,data.Spam,test_size=0.25)


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer


# In[ ]:


from sklearn.naive_bayes import MultinomialNB


# In[ ]:


from sklearn.pipeline import Pipeline
clf=Pipeline([
    ('vectorizer',CountVectorizer()),
    ('nb',MultinomialNB())
])


# In[ ]:


clf.fit(X_train,y_train)


# In[ ]:


emails=[
    'Sounds great! Are you home now?',
    'Will u meet ur dream partner soon? Is ur career off 2 a flyng start? 2 find out free, txt HORO followed by ur star sign, e. g. HORO ARIES'
]


# In[ ]:


clf.predict(emails)


# In[ ]:


clf.score(X_test,y_test)

