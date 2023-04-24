#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
mlb=pd.read_csv('mlb.csv')


# In[2]:


print(mlb.head())
print(mlb.shape)


# In[3]:


print(mlb.describe())


# In[4]:


import matplotlib.pyplot as plt
fig1, ax1 = plt.subplots()
ax1.boxplot([mlb['height']])
ax1.set_ylabel('Height (Inches)')
plt.title('MLB Player Heights')
plt.xticks([1], ['Full Population'])
plt.show()


# In[5]:


sample1=mlb.sample(n=30,random_state=8675309)
sample2=mlb.sample(n=30,random_state=1729)


# In[6]:


sample3=[71, 72, 73, 74, 74, 76, 75, 75, 75, 76, 75, 77,
76, 75, 77, 76, 75,\
76, 76, 75, 75, 81,77, 75, 77, 75, 77, 77, 75, 75]


# In[7]:


import numpy as np
fig1, ax1 = plt.subplots()
ax1.boxplot([mlb['height'],sample1['height'],sample2['height'],np.array(sample3)])
ax1.set_ylabel('Height (Inches)')
plt.title('MLB Player Heights')
plt.xticks([1,2,3,4], ['Full Population','Sample 1','Sample
2','Sample 3'])
plt.show()


# In[8]:


import numpy as np
fig1, ax1 = plt.subplots()
ax1.boxplot([mlb['height'],sample1['height'],sample2['height'],np.array(sample3)])
ax1.set_ylabel('Height (Inches)')
plt.title('MLB Player Heights')
plt.xticks([1,2,3,4], ['Full Population','Sample 1','Sample
2','Sample3'])
plt.show()


# In[9]:


import numpy as np
fig1, ax1 = plt.subplots()
ax1.boxplot([mlb['height'],sample1['height'],sample2['height'],np.array(sample3)])
ax1.set_ylabel('Height (Inches)')
plt.title('MLB Player Heights')
plt.xticks([1,2,3,4], ['Full Population','Sample1','Sample2','Sample3'])
plt.show()


# In[10]:


print(np.mean(sample1['height']))
print(np.mean(sample2['height']))
print(np.mean(sample3))


# In[11]:


alldifferences=[]
for i in range(1000):
newsample1=mlb.sample(n=30,random_state=i*2)
newsample2=mlb.sample(n=30,random_state=i*2+1)
alldifferences.append(newsample1['height'].mean()-
newsample2['height'].mean())
print(alldifferences[0:10])


# In[12]:


alldifferences=[]
for i in range(1000):
    newsample1=mlb.sample(n=30,random_state=i*2)
newsample2=mlb.sample(n=30,random_state=i*2+1)
alldifferences.append(newsample1['height'].mean()-
newsample2['height'].mean())
print(alldifferences[0:10])


# In[13]:


alldifferences=[]
for i in range(1000):
    newsample1=mlb.sample(n=30,random_state=i*2)
    newsample2=mlb.sample(n=30,random_state=i*2+1)
    alldifferences.append(newsample1['height'].mean()-
newsample2['height'].mean())
    
    print(alldifferences[0:10])


# In[14]:


import seaborn as sns
sns.set()
ax=sns.distplot(alldifferences).set_title("Differences
Between Sample Means")
plt.xlabel('Difference Between Means (Inches)')
plt.ylabel('Relative Frequency')
plt.show()


# In[15]:


import seaborn as sns
sns.set()
ax=sns.distplot(alldifferences).set_title("Differences Between Sample Means")
plt.xlabel('Difference Between Means (Inches)')
plt.ylabel('Relative Frequency')
plt.show()


# In[17]:


import seaborn as sns
sns.set()
ax=sns.displot(alldifferences).set_title("Differences Between Sample Means")
plt.xlabel('Difference Between Means (Inches)')
plt.ylabel('Relative Frequency')
plt.show()


# In[18]:


largedifferences=[diff for diff in alldifferences if
abs(diff)>=1.6]
print(len(largedifferences))


# In[19]:


smalldifferences=[diff for diff in alldifferences if
abs(diff)>=0.6]
print(len(smalldifferences))


# In[20]:


import scipy.stats
scipy.stats.ttest_ind(sample1['height'],sample2['height'])


# In[21]:


scipy.stats.ttest_ind(sample1['height'],sample3)


# In[22]:


scipy.stats.mannwhitneyu(sample1['height'],sample2['height'])


# In[23]:


desktop=pd.read_csv('desktop.csv')
laptop=pd.read_csv('laptop.csv')


# In[24]:


print(desktop.head())
print(laptop.head())


# In[25]:


import matplotlib.pyplot as plt
sns.reset_orig()
fig1, ax1 = plt.subplots()
ax1.set_title('Spending by Desktop and Laptop Subscribers')
ax1.boxplot([desktop['spending'].values,laptop['spending'].values])
ax1.set_ylabel('Spending ($)')
plt.xticks([1,2], ['Desktop Subscribers','Laptop
Subscribers'])
plt.show()


# In[26]:


import matplotlib.pyplot as plt
sns.reset_orig()
fig1, ax1 = plt.subplots()
ax1.set_title('Spending by Desktop and Laptop Subscribers')
ax1.boxplot([desktop['spending'].values,laptop['spending'].values])
ax1.set_ylabel('Spending ($)')
plt.xticks([1,2], ['Desktop Subscribers','Laptop Subscribers'])
plt.show()


# In[27]:


print(np.mean(desktop['age']))
print(np.mean(laptop['age']))
print(np.median(desktop['age']))
print(np.median(laptop['age']))
print(np.quantile(laptop['spending'],.25))
print(np.quantile(desktop['spending'],.75))
print(np.std(desktop['age']))


# In[28]:


scipy.stats.ttest_ind(desktop['spending'],laptop['spending'])


# In[29]:


import numpy as np
medianage=np.median(desktop['age'])
groupa=desktop.loc[desktop['age']<=medianage,:]
groupb=desktop.loc[desktop['age']>medianage,:]


# In[30]:


emailresults1=pd.read_csv('emailresults1.csv')


# In[31]:


print(emailresults1.head())


# In[32]:


groupa_withrevenue=groupa.merge(emailresults1,on='userid')
groupb_withrevenue=groupb.merge(emailresults1,on='userid')


# In[ ]:




