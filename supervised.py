#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
news=pd.read_csv('OnlineNewsPopularity.csv')


# In[2]:


from matplotlib import pyplot as plt
plt.scatter(news[' global_sentiment_polarity'],news['
shares'])
plt.title('Popularity by Sentiment')
plt.xlabel('Sentiment Polarity')
plt.ylabel('Shares')
plt.show()


# In[3]:


from matplotlib import pyplot as plt
plt.scatter(news[' global_sentiment_polarity'],news['shares'])
plt.title('Popularity by Sentiment')
plt.xlabel('Sentiment Polarity')
plt.ylabel('Shares')
plt.show()


# In[4]:


from matplotlib import pyplot as plt
plt.scatter(news[' global_sentiment_polarity'],news[' shares'])
plt.title(' Popularity by Sentiment')
plt.xlabel(' Sentiment Polarity')
plt.ylabel(' Shares')
plt.show()


# In[5]:


from sklearn.linear_model import LinearRegression
x = news[' global_sentiment_polarity'].values.reshape(-1,1)
y = news[' shares'].values.reshape(-1,1)
regressor = LinearRegression()
regressor.fit(x, y)
print(regressor.coef_)
print(regressor.intercept_


# In[6]:


from sklearn.linear_model import LinearRegression
x = news[' global_sentiment_polarity'].values.reshape(-1,1)
y = news[' shares'].values.reshape(-1,1)
regressor = LinearRegression()
regressor.fit(x, y)
print(regressor.coef_)
print(regressor.intercept_)


# In[7]:


regline=regressor.predict(x)
plt.scatter(news[' global_sentiment_polarity'],news['
shares'],color='blue')
plt.plot(sorted(news['
global_sentiment_polarity'].tolist()),regline,'r')
plt.title('Shares by Sentiment')
plt.xlabel('Sentiment')
plt.ylabel('Shares')
plt.show()


# In[8]:


regline=regressor.predict(x)
plt.scatter(news[' global_sentiment_polarity'],news[' shares'],color='blue')
plt.plot(sorted(news['
global_sentiment_polarity'].tolist()),regline,'r')
plt.title(' Shares by Sentiment')
plt.xlabel(' Sentiment')
plt.ylabel(' Shares')
plt.show()


# In[9]:


regline=regressor.predict(x)
plt.scatter(news[' global_sentiment_polarity'],news[' shares'],color='blue')
plt.plot(sorted(news[' global_sentiment_polarity'].tolist()),regline,'r')
plt.title(' Shares by Sentiment')
plt.xlabel(' Sentiment')
plt.ylabel(' Shares')
plt.show()


# In[ ]:




