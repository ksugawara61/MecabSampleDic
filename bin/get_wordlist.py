from gensim.models import word2vec

model = word2vec.Word2Vec.load("./model/app.model")
results = model.wv.vocab

for result in results:
    print(result)
