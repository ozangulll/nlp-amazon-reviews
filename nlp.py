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
##Numbers
df["reviewText"] = df["reviewText"].str.replace(r'\d', '', regex=True)
df["reviewText"].tail()

##StopWords  (NLTK Library include stop words)
#nltk.download('stopwords')  // Package stopwords is already up-to-date!

sw = stopwords.words('english')
print(sw)
print(type(sw))
#we should split in terms of blank and compare with stopwords
df["reviewText"] = df["reviewText"].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))
df["reviewText"].head()
##RareWords
temp_df = pd.Series(' '.join(df['reviewText']).split()).value_counts()
#print(temp_df)
drops = temp_df[temp_df <= 1]
print(drops)
df["reviewText"] = df["reviewText"].apply(lambda x: " ".join(x for x in str(x).split() if x not in drops))
df["reviewText"]

##Tokenization

#nltk.download("punkt") Package punkt is already up-to-date!

df["reviewText"].apply(lambda x: TextBlob(x).words).head()

#Lemmataization
#nltk.download('wordnet')  Package wordnet is already up-to-date!
df['reviewText'] = df['reviewText'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
df["reviewText"].head()

####################################
#Text Visualization
####################################

#To visualize the text, calculate the frequency of them (words)

tf = df["reviewText"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()

tf.columns = ["words", "tf"]

tf.sort_values("tf", ascending=False)
##Bar Plot
#greater than 500
tf[tf["tf"] > 500].plot.bar(x="words", y="tf")
#plt.show()

##Word Cloud
text = " ".join(i for i in df.reviewText)

wordcloud2 = WordCloud().generate(text)
plt.imshow(wordcloud2, interpolation="bilinear")
plt.axis("off")
# plt.show()

tr_mask = np.array(Image.open("template/tr.png"))

wc = WordCloud(background_color="white",
               max_words=1000,
               mask=tr_mask,
               contour_width=3,
               contour_color="firebrick")

wc.generate(text)
plt.figure(figsize=[10, 10])
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
#plt.show()
df["reviewText"].head()


#SENTIMENT ANALYSIS
#nltk.download('vader_lexicon')
sia=SentimentIntensityAnalyzer()
# sia.polarity_scores("The film was awesome")  #TRY FOR SIA

df["reviewText"][0:10].apply(lambda x: sia.polarity_scores(x))
df["reviewText"][0:10].apply(lambda x: sia.polarity_scores(x)["compound"])
df["polarity_score"]=df["reviewText"].apply(lambda x: sia.polarity_scores(x)["compound"])
print(df["polarity_score"].head())


