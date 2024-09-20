
#Text PreProcessing

# !pip install nltk
# !pip install textblob
# !pip install wordcloud
import numpy as np
import pandas as pd

df = pd.read_csv("datasets/amazon_reviews.csv", sep=",")
print(df.head())

#Normalizing Case Folding
print(df['reviewText'])
df['reviewText'] = df['reviewText'].str.lower()
print(df['reviewText'])



