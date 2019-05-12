from gensim.models import word2vec
import sys

args = sys.argv

model = word2vec.Word2Vec.load("./model/app.model")
results = model.wv.most_similar(positive=[args[1]])

for result in results:
    print(result)
