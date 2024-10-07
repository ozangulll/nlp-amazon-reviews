#Text PreProcessing

# !pip install nltk
# !pip install textblob
# !pip install wordcloud
from warnings import filterwarnings
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
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

#print(df.head())

##Text Preprocessing
##Text Visualization
##Sentiment Analysis
##Feature Engineering
##Sentiment Modeling

#->Text Preprocessing
df = pd.read_csv("datasets/amazon_reviews.csv", sep=",")
df.head()

##Normalizing Case Folding
df["reviewText"].head()
df["reviewText"] = df["reviewText"].str.lower()
##Punctuations
df["reviewText"].head()
df["reviewText"] = df["reviewText"].str.replace(r'[^\w\s]', '', regex=True)
df["reviewText"].head()
