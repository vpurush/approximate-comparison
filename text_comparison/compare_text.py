from sklearn.feature_extraction.text import CountVectorizer
from lsh.lsh import LSH
import numpy as np

texts = [
    'Jack went to the market to buy some fruits',
    'Jane went to the market to buy some fruits today',
    'Robert and his team played hockey today'
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts).toarray().reshape(len(texts), 1, -1)

lshModel = LSH(noOfHashers=25, noOfHash=3, dimension=X.shape[2])

for i in range(0, X.shape[0]):
    lshModel.train(X[i], { "name": texts[i] })

print(lshModel.isSimilar(X[0], X[1]))
print(lshModel.isSimilar(X[0], X[2]))
print(lshModel.isSimilar(X[1], X[2]))