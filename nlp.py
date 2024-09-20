#Text PreProcessing

# !pip install nltk
# !pip install textblob
# !pip install wordcloud
from warnings import filterwarnings
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from PIL import Image
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate
from sklearn.preprocessing import LabelEncoder
from textblob import Word, TextBlob
from wordcloud import WordCloud

filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
df = pd.read_csv("datasets/amazon_reviews.csv", sep=",")
#print(df.head())

# Normalizing Case Folding
#print(df['reviewText'])
df['reviewText'] = df['reviewText'].str.lower()

df['reviewText'] = df['reviewText'].str.replace(r"[^\w\s]", "", regex=True)
df['reviewText'] = df['reviewText'].str.replace(r'\d', '', regex=True)

#STOP WORDS  (LIKE OF THIS THE..... ETC)
#there is a list which include stop words in NLTK library
#nltk.download('stopwords')
sw=stopwords.words('english')
print(sw)
df['reviewText']=df['reviewText'].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))


# RARE WORDS
# If we want to delete rare words. We can choose and delete it.

tempDf=pd.Series(''.join(df['reviewText']).split()).value_counts()
drops=tempDf[tempDf<=1]
print(drops)

df['reviewText']=df['reviewText'].apply(lambda x: " ".join(x for x in str(x).split() if x not in drops))
print(df['reviewText'])