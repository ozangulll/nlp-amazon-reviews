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
#print(sw)
df['reviewText']=df['reviewText'].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))
# RARE WORDS
# If we want to delete rare words. We can choose and delete it.
tempDf=pd.Series(' '.join(df['reviewText']).split()).value_counts()
drops=tempDf[tempDf<=1]
#print(drops)

df['reviewText']=df['reviewText'].apply(lambda x: " ".join(x for x in x.split() if x not in drops))
#print(df['reviewText'])

#nltk.download('punkt')

df["reviewText"].apply(lambda x: TextBlob(x).words).head()

#nltk.download('wordnet')
df['reviewText'] = df['reviewText'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
#print(df['reviewText'])

#FREQUENCY CALCULATION
tf=df["reviewText"].apply(lambda x:pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
tf.columns=["words","tf"]
tf.sort_values(by=['tf'],ascending=False)
print(tf)
#BAR PLOT
tf[tf["tf"]>500].plot.bar(x="words",y="tf")
#plt.show()

#WORD CLOUD

text=" ".join(i for i in df['reviewText'])
wordCloud=WordCloud().generate(text)
plt.imshow(wordCloud,interpolation='bilinear')
plt.axis('off')
#plt.show()

wordCloud=WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
plt.figure()
plt.imshow(wordCloud,interpolation='bilinear')
plt.axis('off')
#plt.show()
#ŞABLONLARA GÖRE WORDCLOUD OLUŞTURMA
mask_path = 'template/tr.png'  # kendi dosya yolunuza göre ayarlayın
img = Image.open(mask_path)
enhancer = ImageEnhance.Contrast(img)
enhanced_img = enhancer.enhance(2)
enhanced_mask = np.array(enhanced_img)
wc=WordCloud(background_color="white",max_words=1000,mask=enhanced_mask,contour_width=3,contour_color="firebrick")
wc.generate(text)
plt.figure(figsize = [10,10])
plt.imshow(wc,interpolation='bilinear')
plt.axis('off')
#plt.show()

#SENTIMENT ANALYSIS
print(df.head())
#nltk.download('vader_lexicon')
sia=SentimentIntensityAnalyzer()
sia.polarity_scores("The film was awesome")
print(type(sia.polarity_scores("The film was awesome")))
df['polarity_scores'] = df['reviewText'].apply(lambda x: sia.polarity_scores(x))
df['compound'] = df['polarity_scores'].apply(lambda x: x['compound'])
#overall score u 3 den küçük olup polarity score u 0 dan büyük olanları araştır mesela bununla ilgili bir çalışma yapılabilir
#çünkü burada aslında bir çatışma var.
# Şartlara uygun incelemeleri filtrele
filtered_reviews = df[(df['overall'] < 3) & (df['compound'] > 0)]

# Sonuçları yazdır
print(filtered_reviews[['reviewText', 'overall', 'compound']])


