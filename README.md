# nlp-amazon-reviews
The project covers multiple aspects of the NLP pipeline, including text preprocessing, visualization, sentiment analysis, feature engineering, and sentiment modeling for a amazon product comment reviews.


# Text PreProcessing and Sentiment Analysis

## Table of Contents
1. [Installation](#installation)
2. [Project Overview](#project-overview)
3. [Steps](#steps)
   - [Text Preprocessing](#text-preprocessing)
   - [Text Visualization](#text-visualization)
   - [Sentiment Analysis](#sentiment-analysis)
   - [Sentiment Modeling](#sentiment-modeling)
4. [Usage](#usage)

## Installation

To install the necessary libraries, use the following commands:

```bash
pip install nltk
pip install textblob
pip install wordcloud
pip install matplotlib
pip install pandas
pip install scikit-learn
```
# Project Overview
This project processes Amazon reviews for sentiment analysis and modeling. It covers data preprocessing, including case normalization, removing punctuations, handling stop words, and tokenizing and lemmatizing words. Next, it visualizes the cleaned text using word clouds and bar plots. Finally, the project performs sentiment analysis using SentimentIntensityAnalyzer from NLTK.

## Steps

### 1. Text Preprocessing

#### a. Load Data
- Load the dataset using `pandas` and view the structure with `df.head()`.

#### b. Normalize and Clean Text
- **Lowercase Conversion**: Converts all text to lowercase to standardize case.
- **Punctuation Removal**: Removes punctuation for cleaner data.
- **Number Removal**: Eliminates numbers that may not add value to text analysis.

#### c. Stop Words Removal
- Uses stop words from the NLTK library to remove common but less meaningful words (e.g., "the", "and").

#### d. Rare Word Removal
- Removes words that appear infrequently (frequency of 1) to reduce noise.

#### e. Tokenization and Lemmatization
- **Tokenization**: Splits text into individual words.
- **Lemmatization**: Converts words to their base or root form using TextBlob.

### 2. Text Visualization

#### a. Word Frequency
- Calculates word frequency to understand commonly used terms in the text.

#### b. Bar Plot
- Creates a bar plot to visualize word frequencies for words with a frequency count greater than 500.

#### c. Word Cloud
- Generates a word cloud to represent the most frequent terms, creating a more visual understanding of word distribution.

### 3. Sentiment Analysis

- Uses `SentimentIntensityAnalyzer` from the NLTK library to compute sentiment scores for each review.
- Analyzes the compound score to determine overall sentiment (positive, neutral, or negative) for each entry.

### 4. Sentiment Modeling

- Applies basic sentiment models using:
  - **RandomForestClassifier**
  - **LogisticRegression**
- Utilizes cross-validation to assess model performance and improve accuracy.

## Usage

1. Ensure all dependencies are installed as outlined in the **Installation** section.
2. Load the dataset into a `DataFrame` with `pd.read_csv()` and assign it to `df`.
3. Run each step in the **Text Preprocessing** section sequentially to clean and transform the data.
4. Use the **Text Visualization** steps to generate visual representations of word frequency.
5. Perform **Sentiment Analysis** to get sentiment scores for each review.
6. Optionally, proceed to **Sentiment Modeling** to build predictive models for sentiment classification.

```python
# Sample Usage Code
df = pd.read_csv("datasets/amazon_reviews.csv", sep=",")
# Run preprocessing, visualization, sentiment analysis, and modeling steps as needed.
```
# Plots from the code
![plot1bar](https://github.com/user-attachments/assets/f2874b07-f6f4-4960-84c5-b01dbc26c03e)


![plot2wordcloud](https://github.com/user-attachments/assets/76c26d74-d5db-4360-b21c-b0a965979054)


![plot3wordcloud](https://github.com/user-attachments/assets/f5559641-eab8-43e8-8a09-8cc190854a19)
