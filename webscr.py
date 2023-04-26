#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
urltoget = 'https://bradfordtuckfield.com/
indexarchive20210903.html'
pagecode = requests.get(urltoget)
print(pagecode.text[0:600])


# In[2]:


import requests urltoget = 'https://bradfordtuckfield.com/indexarchive20210903.html'
pagecode = requests.get(urltoget)
print(pagecode.text[0:600])


# In[3]:


import requests
urltoget = 'https://bradfordtuckfield.com/indexarchive20210903.html'
pagecode = requests.get(urltoget)
print(pagecode.text[0:600])


# In[4]:


print(pagecode.text[0:600]


# In[5]:


urltoget = 'https://bradfordtuckfield.com/
contactscrape.html'
pagecode = requests.get(urltoget)
mail_beginning=pagecode.text.find('Email:')
print(mail_beginning)


# In[6]:


radfordtuckfield.com/contactscrape.html'
pagecode = requests.get(urltoget)
mail_beginning=pagecode.text.find('Email:')
print(mail_beginning)


# In[7]:


urltoget = 'https://bradfordtuckfield.com/contactscrape.html'
pagecode = requests.get(urltoget)
mail_beginning=pagecode.text.find('Email:')
print(mail_beginning)


# In[8]:


print(pagecode.text[(mail_beginning):(mail_beginning+80)])


# In[9]:


print(pagecode.text[(mail_beginning+38):(mail_beginning+64)])


# In[10]:


urltoget = 'https://bradfordtuckfield.com/contactscrape.html'
pagecode = requests.get(urltoget)
at_beginning=pagecode.text.find('@')
print(at_beginning)


# In[11]:


print(pagecode.text[(at_beginning-4):(at_beginning+22)])


# In[12]:


import re
print(re.search(r'recommend','irrelevant text I recommend
irrelevant text').span())


# In[13]:


import re
print(re.search(r'recommend','irrelevant text I recommend irrelevant text').span())


# In[15]:


import re
print(re.search('rec+om+end', 'irrelevant text I recommend irrelevant text').span())


# In[16]:


import re
print(re.search('rec+om+end','irrelevant text I recomend irrelevant text').span())
print(re.search('rec+om+end','irrelevant text I reccommend irrelevant text').span())
print(re.search('rec+om+end','irrelevant text I reommend irrelevant text').span())
print(re.search('rec+om+end','irrelevant text I recomment irrelevant text').span())


# In[17]:


re.search('10*','My bank balance is 100').span()


# In[18]:


import re
print(re.search('10*','My bank balance is 1').span())
print(re.search('10*','My bank balance is 1000').span())
print(re.search('10*','My bank balance is 9000').span())
print(re.search('10*','My bank balance is 1000000').span())


# In[19]:


print(re.search('Clarke?','Please refer questions to Mr.Clark').span())


# In[20]:


re.search('[a-zA-Z]+@[a-zA-Z]+\.[a-zA-Z]+',\ 'My Twitter is @fake; my email is abc@def.com').span()


# In[21]:


re.search('[a-zA-Z]+@[a-zA-Z]+\.[a-zA-Z]+',\
'My Twitter is @fake; my email is abc@def.com').span()


# In[22]:


import requests
from bs4 import BeautifulSoup
URL = 'https://bradfordtuckfield.com/
indexarchive20210903.html'
response = requests.get(URL)
soup = BeautifulSoup(response.text, 'lxml')
all_urls = soup.find_all('a')
for each in all_urls:
print(each['href'])


# In[23]:


import requests
from bs4 import BeautifulSoup
URL = 'https://bradfordtuckfield.com/indexarchive20210903.html'
response = requests.get(URL)
soup = BeautifulSoup(response.text, 'lxml')
all_urls = soup.find_all('a')
for each in all_urls:
print(each['href'])


# In[24]:


import requests
from bs4 import BeautifulSoup
URL = 'https://bradfordtuckfield.com/
indexarchive20210903.html'
response = requests.get(URL)
soup = BeautifulSoup(response.text, 'lxml')
all_urls = soup.find_all('a')
    for each in all_urls:
    print(each['href'])


# In[25]:


import requests
from bs4 import BeautifulSoup
URL = 'https://bradfordtuckfield.com/indexarchive20210903.html'
response = requests.get(URL)
soup = BeautifulSoup(response.text, 'lxml')
all_urls = soup.find_all('a')
for each in all_urls:
    print(each['href'])


# In[26]:


import requests
from bs4 import BeautifulSoup
URL = 'https://bradfordtuckfield.com/contactscrape.html'
response = requests.get(URL)
soup = BeautifulSoup(response.text, 'lxml')
email = soup.find('label',{'class':'email'}).text
mobile = soup.find('label',{'class':'mobile'}).text
website = soup.find('a',{'class':'website'}).text
print("Email : {}".format(email))
print("Mobile : {}".format(mobile))
print("Website : {}".format(website))


# In[27]:


import requests
from bs4 import BeautifulSoup
URL = 'https://bradfordtuckfield.com/
user_detailsscrape.html'
response = requests.get(URL)
soup = BeautifulSoup(response.text, 'lxml')
all_user_entries = soup.find_all('tr',{'class':'user-
details'})
for each_user in all_user_entries:
user = each_user.find_all("td")
print("User Firstname : {}, Lastname : {}, Age: {}"\
.format(user[0].text, user[1].text, user[2].text))


# In[28]:


import requests
from bs4 import BeautifulSoup
URL = 'https://bradfordtuckfield.com/user_detailsscrape.html'
response = requests.get(URL)
soup = BeautifulSoup(response.text, 'lxml')
all_user_entries = soup.find_all('tr',{'class':'user-details'})
for each_user in all_user_entries:
user = each_user.find_all("td")
print("User Firstname : {}, Lastname : {}, Age: {}"\
.format(user[0].text, user[1].text, user[2].text))


# In[31]:


import requests
from bs4 import BeautifulSoup
URL = 'https://bradfordtuckfield.com/user_detailsscrape.html'
response = requests.get(URL)
soup = BeautifulSoup(response.text, 'lxml')
all_user_entries = soup.find_all('tr',{'class':'user-details'})
for each_user in all_user_entries:
    user = each_user.find_all("td")
    print("User Firstname : {}, Lastname : {}, Age: {}"\
    .format(user[0].text, user[1].text, user[2].text))


# In[ ]:




