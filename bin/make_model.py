from gensim.models import word2vec
import logging
import sys

args = sys.argv

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences = word2vec.Text8Corpus(args[1])

model = word2vec.Word2Vec(sentences, size=200, min_count=15, window=15)
model.save("./model/app.model")
