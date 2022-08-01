import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import plotly.express as px
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.model_selection import train_test_split

#0. Small EDA - completed
data=pd.read_csv('Instagram.csv')
print(data.head())

null_cells=data.isnull().sum()
print(null_cells)

data=data.dropna()
data.info()

#1. Impressions Distribution

plt.figure(figsize=(10,8))
plt.style.use('fivethirtyeight')
plt.title("From home impressions distributions")
#sns.displot(data['From Home'])
plt.hist(data['From Home'])
plt.show()

#2. Wordcloud + mask - completed

mask=np.array(Image.open('plane.jpg'))

text = " ".join(i for i in data.Caption)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, mask=mask, background_color="white").generate(text)
plt.style.use('classic')
plt.figure(figsize=(12,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

#3. Relationships (scatter plt)
## TBD