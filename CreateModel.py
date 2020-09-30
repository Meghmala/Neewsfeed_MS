
# coding: utf-8
import requests
import pandas as pd
import json
pd.set_option('display.max_colwidth', 200)

# Generates the code, obtain the consumer key using the link https://getpocket.com/developer/apps/new
auth_params = {'consumer_key': '69647-66146e0e8109ae1b769dfc2f', 'redirect_uri':'https://www.github.com/krishnachaurasia/'}
tkn = requests.post('https://getpocket.com/v3/oauth/request',data=auth_params)
tkn.content

# Visit this URL and authorize : https://getpocket.com/auth/authorize?request_token=7fc580a7-f6db-dcba-7b19-fae9c8&redirect_uri=https://github.com/KrishnaChaurasia/
# Make the appropriate changes below and generate the access token
usr_params = {'consumer_key':'69647-66146e0e8109ae1b769dfc2f', 'code':'7fc580a7-f6db-dcba-7b19-fae9c8'}
usr = requests.post('https://getpocket.com/v3/oauth/authorize',data=usr_params)
usr.content

# Make the appropriate changes and retrieve the 'n' tagged articles in the json format
no_params = {'consumer_key':'69647-66146e0e8109ae1b769dfc2f', 'access_token':'92c4768f-677b-7114-cdb6-1ba48c','tag': 'n'}
no_result = requests.post('https://getpocket.com/v3/get',data=no_params)

# Retrive the links of 'n' tagged articles' urls
no_jf = json.loads(no_result.text)
no_jd = no_jf['list']
no_urls = []
for i in no_jd.values():
    no_urls.append(i.get('resolved_url'))

# Store them as the dataframe
no_uf = pd.DataFrame(no_urls, columns=['urls'])
no_uf = no_uf.assign(wanted = lambda x: 'n')

# Repeat the same steps for the 'y' tagged articles
yes_params = {'consumer_key':'69647-66146e0e8109ae1b769dfc2f', 'access_token':'92c4768f-677b-7114-cdb6-1ba48c','tag': 'y'}
yes_result = requests.post('https://getpocket.com/v3/get',data=yes_params)

# Retrive the links of 'n' tagged articles' urls
yes_jf = json.loads(yes_result.text)
yes_jd = yes_jf['list']
yes_urls = []
for i in yes_jd.values():
    yes_urls.append(i.get('resolved_url'))

# Store them as the dataframe
yes_uf = pd.DataFrame(yes_urls, columns=['urls'])
yes_uf = yes_uf.assign(wanted = lambda x: 'y')

# Join the two dataframes using concat
df = pd.concat([yes_uf, no_uf])
df.dropna(inplace = True)

# Using the embed.ly API to download story bodies
import urllib
def get_html(x):
    qurl = urllib.parse.quote(x)
    rhtml = requests.get('https://api.embedly.com/1/extract?url=' + qurl + '&key=aab6514b92e84ae0a406db478a3d426f')
    ctnt = json.loads(rhtml.text).get('content')
    return ctnt
df.loc[:,'html'] = df['urls'].map(get_html)
df.dropna(inplace = 1)
df.to_excel(pd.ExcelWriter('NewsStories.xlsx', engine='xlsxwriter'), index=False)

# Parser to extract text from HTML
from bs4 import BeautifulSoup
def get_text(x):
    soup = BeautifulSoup(x, 'lxml')
    text = soup.get_text()
    return text
df.loc[:,'text'] = df['html'].map(get_text)

# Generate the tf-idf matrix for our text
from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer(ngram_range = (1,3), stop_words = 'english', min_df = 3)
tv = vect.fit_transform(df['text'])

# Feed the tf-idf matrix to generate the Linear SVM model
from sklearn.svm import LinearSVC
clf = LinearSVC()
model = clf.fit(tv, df['wanted'])

# Authorize Gspread
import json
import gspread
from oauth2client.service_account import ServiceAccountCredentials
scope = ['https://spreadsheets.google.com/feeds']
credentials = ServiceAccountCredentials.from_json_keyfile_name(r'C:\Users\Dell\Desktop\ML Projects\Custom Newsfeed App\NewsFeed-b6c37e687f23.json', scope)
gc = gspread.authorize(credentials)

# Load the spreadsheet and format it as dataframe
ws = gc.open("NewsStories.xlsx")
sh = ws.sheet1
zd = list(zip(sh.col_values(2), sh.col_values(3), sh.col_values(4)))
zf = pd.DataFrame(zd, columns=['title','urls','html'])
zf.replace('', pd.np.nan, inplace=True)
zf.dropna(inplace=True)
zf.drop(zf.index[[0]], inplace=True)

zf.loc[:,'text'] = zf['html'].map(get_text) 
zf.reset_index(drop=True, inplace=True)
test_matrix = vect.transform(zf['text'])

# Test the model
results = pd.DataFrame(model.predict(test_matrix), columns = ['wanted'])

rez = pd.merge(results, zf, left_index=True, right_index=True)

change_to_no = [130, 145, 148, 163, 178, 199, 219, 222, 223, 226, 235, 279, 348, 357, 427, 440, 542, 544, 546]
change_to_yes = [0, 9, 29, 35, 42, 71, 110, 190, 319, 335, 344, 371, 385, 399, 408, 409, 422, 472, 520, 534]

# Modify the incorrectly marked labels
for i in rez.iloc[change_to_yes].index:
    rez.iloc[i]['wanted'] = 'y'
for i in rez.iloc[change_to_no].index:
    rez.iloc[i]['wanted'] = 'n'

# Merge the labels with the df
combined = pd.concat([df[['wanted', 'text']], rez[['wanted','text']]])

tvcomb = vect.fit_transform(combined['text'], combined['wanted'])
model = clf.fit(tvcomb, combined['wanted'])

# Dump the model using pickle
import pickle
pickle.dump(model, open(r'C:\Users\Dell\Desktop\ML Projects\Custom Newsfeed App\news_model_pickle.p', 'wb'))
pickle.dump(vect, open(r'C:\Users\Dell\Desktop\ML Projects\Custom Newsfeed App\news_vect_pickle.p', 'wb'))
