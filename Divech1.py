#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[3]:


hour=pd.read_csv ('hour.csv')


# In[4]:


print (hour.head())


# In[5]:


print (hour['count'].mean())


# In[6]:


print (hour['count'].median())
print (hour['count'].std())
print (hour['registered'].min())
print (hour['registered'].max())


# In[7]:


print (hour.describe())


# In[8]:


print (hour.loc[3,'count'])


# In[9]:


print (hour.loc[2:4,'registered'])


# In[10]:


print (hour.loc[hour['hr']<5,'registered'].mean())


# In[11]:


print (hour.loc[(hour['hr']<5) &
(hour['temp']<.50),'count'].mean())
print (hour.loc[(hour['hr']<5) &
(hour['temp']>.50),'count'].mean())


# In[12]:


print (hour.loc[(hour['temp']>0.5) |
(hour['hum']>0.5),'count'].mean())


# In[13]:


print (hour.groupby(['season'])['count'].mean())


# In[14]:


print (hour.groupby(['season','holiday'])['count'].mean())


# In[15]:


import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(x = hour['instant'], y = hour['count'])
plt.show()


# In[16]:


fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(x = hour['instant'], y = hour['count'])
plt.xlabel("Hour")
plt.ylabel("Count")
plt.title("Ridership Count by Hour")
plt.show()


# In[17]:


hour_first48=hour.loc[0:48,:]
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(x = hour_first48['instant'], y =
hour_first48['count'])
plt.xlabel("Hour")
plt.ylabel("Count")
plt.title("Count by Hour - First Two Days")
plt.show()


# In[18]:


fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(x = hour_first48['instant'], y =
hour_first48['count'],c='red',marker='+')
plt.xlabel("Hour")
plt.ylabel("Count")
plt.title("Count by Hour - First Two Days")
plt.show()


# In[19]:


fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(hour_first48['instant'],
hour_first48['casual'],c='red',label='casual',linestyle='-')
ax.plot(hour_first48['instant'],\
hour_first48['registered'],c='blue',label='registered',linestyle='--')
ax.legend()
plt.show()


# In[20]:


import seaborn as sns
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(x='hr', y='registered', data=hour)
plt.xlabel("Hour")
plt.ylabel("Count")
plt.title("Counts by Hour")
plt.show()


# In[21]:


fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(hour['count'],bins=80)
plt.xlabel("Ridership")
plt.ylabel("Frequency")
plt.title("Ridership Histogram")
plt.show()


# In[22]:


thevariables=['hr','temp','windspeed']
hour_first100=hour.loc[0:100,thevariables]
sns.pairplot(hour_first100, corner=True)
plt.show()


# In[23]:


print (hour['casual'].corr(hour['registered']))
print (hour['temp'].corr(hour['hum']))


# In[24]:


thenames=['hr','temp','windspeed']
cor_matrix = hour[thenames].corr()
print(cor_matrix)


# In[25]:


plt.figure(figsize=(14,10))
corr = hour[thenames].corr()
sns.heatmap(corr, annot=True,cmap='binary',
fmt=".3f",
xticklabels=thenames,
yticklabels=thenames)
plt.show()


# In[26]:


# Create a pivot table
df_hm =hour.pivot_table(index = 'hr',columns
='weekday',values ='count')
# Draw a heatmap
plt.figure(figsize = (20,10)) # To resize the plot
sns.heatmap(df_hm, fmt="d", cmap='binary',linewidths=.5,
vmin = 0)
plt.show()


# In[ ]:




