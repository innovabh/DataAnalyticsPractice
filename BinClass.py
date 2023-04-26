#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
attrition_past=pd.read_csv('https://bradfordtuckfield.com/
attrition_past.csv')


# In[2]:


import pandas as pd
attrition_past=pd.read_csv('https://bradfordtuckfield.com/attrition_past.csv')


# In[3]:


print(attrition_past.head())


# In[4]:


pd.set_option('display.max_columns', 6)


# In[5]:


print(attrition_past['exited'].mean())


# In[6]:


from matplotlib import pyplot as plt
plt.scatter(attrition_past['lastmonth_activity'],attrition_past['exited'])
plt.title('Historical Attrition')
plt.xlabel('Last Month\'s Activity')
plt.ylabel('Attrition')
plt.show()


# In[8]:


x = attrition_past['lastmonth_activity'].values.reshape(-1,1)
y = attrition_past['exited'].values.reshape(-1,1)
from sklearn.linear_model import LinearRegression 
regressor = LinearRegression() 
regressor.fit(x, y)


# In[9]:


from matplotlib import pyplot as plt
plt.scatter(attrition_past['lastmonth_activity'],attrition_past['exited'])
prediction = [regressor.coef_[0]*x+regressor.intercept_[0]
for x in \
list(attrition_past['lastmonth_activity'])]
plt.plot(attrition_past['lastmonth_activity'], prediction,
color='red')
plt.title('Historical Attrition')
plt.xlabel('Last Month\'s Activity')
plt.ylabel('Attrition')
plt.show()


# In[10]:


attrition_past['predicted']=regressor.predict(x)


# In[11]:


print(attrition_past.head())


# In[12]:


attrition_future=pd.read_csv('http://bradfordtuckfield.com/attrition2.csv')


# In[14]:


x = attrition_future['lastmonth_activity'].values.reshape(-1,1)
attrition_future['predicted']=regressor.predict(x)


# In[15]:


print(attrition_future.head())


# In[16]:


print(attrition_future.nlargest(5,'predicted'))


# In[17]:


themedian=attrition_past['predicted'].median()
prediction=list(1*(attrition_past['predicted']>themedian))
actual=list(attrition_past['exited'])


# In[18]:


from sklearn.metrics import confusion_matrix
print(confusion_matrix(prediction,actual))


# In[19]:


conf_mat = confusion_matrix(prediction,actual)
precision = conf_mat[0][0]/(conf_mat[0][0]+conf_mat[0][1])
recall = conf_mat[0][0]/(conf_mat[0][0]+conf_mat[1][0])


# In[20]:


x3 = attrition_past.loc[:,['lastmonth_activity',
'lastyear_activity',\
'number_of_employees']].values.reshape(-1,3)
y = attrition_past['exited'].values.reshape(-1,1)
regressor_multi = LinearRegression()
regressor_multi.fit(x3, y)


# In[21]:


attrition_future['predicted_multi']=regressor_multi.predict(x3)


# In[22]:


print(attrition_future.nlargest(5,'predicted_multi'))


# In[23]:


print(list(attrition_future.sort_values(by='predicted_multi',
\
ascending=False).loc[:,'corporation']))


# In[24]:


attrition_future['activity_per_employee']=attrition_future.loc[:,
\
'lastmonth_activity']/
attrition_future.loc[:,'number_of_employees']


# In[25]:


attrition_future['activity_per_employee']=attrition_future.loc[:,\ 'lastmonth_activity']/
attrition_future.loc[:,'number_of_employees']


# In[26]:


attrition_future['activity_per_employee']=attrition_future.loc[:,
\
'lastmonth_activity']/attrition_future.loc[:,'number_of_employees']


# In[27]:


attrition_past['activity_per_employee']=attrition_past.loc[:,
\
'lastmonth_activity']/
attrition_past.loc[:,'number_of_employees']
x = attrition_past.loc[:,
['activity_per_employee','lastmonth_activity',\
'lastyear_activity',
'number_of_employees']].values.reshape(-1,4)
y = attrition_past['exited'].values.reshape(-1,1)
regressor_derived= LinearRegression()
regressor_derived.fit(x, y)
attrition_past['predicted3']=regressor_derived.predict(x)
x = attrition_future.loc[:,
['activity_per_employee','lastmonth_activity',\
'lastyear_activity',
'number_of_employees']].values.reshape(-1,4)
attrition_future['predicted3']=regressor_derived.predict(x)


# In[28]:


attrition_past['activity_per_employee']=attrition_past.loc[:,
\
'lastmonth_activity']/attrition_past.loc[:,'number_of_employees']
x = attrition_past.loc[:,
['activity_per_employee','lastmonth_activity',\
'lastyear_activity',
'number_of_employees']].values.reshape(-1,4)
y = attrition_past['exited'].values.reshape(-1,1)
regressor_derived= LinearRegression()
regressor_derived.fit(x, y)
attrition_past['predicted3']=regressor_derived.predict(x)
x = attrition_future.loc[:,
['activity_per_employee','lastmonth_activity',\
'lastyear_activity',
'number_of_employees']].values.reshape(-1,4)
attrition_future['predicted3']=regressor_derived.predict(x)


# In[29]:


print(list(attrition_future.sort_values(by='predicted3',ascending


# In[30]:


themedian=attrition_past['predicted3'].median()
prediction=list(1*(attrition_past['predicted3']>themedian))
actual=list(attrition_past['exited'])


# In[31]:


print(confusion_matrix(prediction,actual))


# In[32]:


from matplotlib import pyplot as plt
import numpy as np
import math
x = np.arange(-5, 5, 0.05)
y = (1/(1+np.exp(-1-2*x)))
plt.plot(x,y)
plt.xlabel("X")
plt.ylabel("Value of Logistic Function")
plt.title('A Logistic Curve')
plt.show()


# In[33]:


from matplotlib import pyplot as plt
import numpy as np
import math
x = np.arange(-5, 5, 0.05)
y = (1/(1+np.exp(1+2*x)))
plt.plot(x,y)
plt.xlabel("X")
plt.ylabel("Value of Logistic Function")
plt.title('A Logistic Curve')
plt.show()


# In[34]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='liblinear',
random_state=0)
x =
attrition_past['lastmonth_activity'].values.reshape(-1,1)
y = attrition_past['exited']
model.fit(x, y)


# In[35]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='liblinear',
random_state=0)
x = attrition_past['lastmonth_activity'].values.reshape(-1,1)
y = attrition_past['exited']
model.fit(x, y)


# In[36]:


attrition_past['logisticprediction']=model.predict_proba(x)
[:,1]


# In[37]:


attrition_past['logisticprediction']=model.predict_proba(x) [:,1]


# In[ ]:




