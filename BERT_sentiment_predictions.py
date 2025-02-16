# -*- coding: utf-8 -*-
"""Untitled17.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1fUHS0KdyotJPesGbeA9XYQ9GZN4W0fxv
"""

#download 100 movie reviews from and IMDB dataset of 50k Movie reviews
!wget https://samsclass.info/ML/proj/IMDB100a.csv

import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification


pd.set_option('display.max_rows', 4)

df_orig = pd.read_csv('IMDB100a.csv')
print("Original Data:")
print(df_orig)
df = df_orig.drop('sentiment', axis=1)

print()
print("Data Without Sentiment")
print(df)

#run a pretrained BERT sentiment analysis model on the data:
tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
def sentiment_movie_score(movie_review):
	token = tokenizer.encode(movie_review, return_tensors = 'pt')
	result = model(token)
	return int(torch.argmax(result.logits))+1
df['sentiment'] = df['review'].apply(lambda x: sentiment_movie_score(x[:512]))

print(df)

#count the number of reviews that are erroneously classified, in two categories:
#Actually positive but sentiment is 1 or 2
#Actually negative but sentiment is 4 or 5

num_wrong = 0
sentiment_sum = 0

for (sentiment_predicted, sentiment_correct) in zip(df['sentiment'], df_orig['sentiment']):
  if (sentiment_predicted < 3) and (sentiment_correct == "positive"):
    num_wrong += 1
    print(sentiment_predicted, sentiment_correct)
  elif (sentiment_predicted > 3) and (sentiment_correct == "negative"):
    num_wrong += 1
    print(sentiment_predicted, sentiment_correct)
  sentiment_sum += sentiment_predicted

print()
print("BERT got ", num_wrong, "wrong out of 100 movies.")
print()
print("Total sentiment:", sentiment_sum)