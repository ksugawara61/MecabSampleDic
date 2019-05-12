from gensim.models import word2vec
import sys

args = sys.argv
word = args[1]

model = word2vec.Word2Vec.load("./model/app.model")
print(word)
print(model.wv.get_vector(word))
