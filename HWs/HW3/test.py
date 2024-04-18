import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec


# Read the .csv file
df = pd.read_csv('WikipediaEvents.csv')

# Tokenize the text data
df['tokens'] = df['text'].apply(word_tokenize)

# Remove stop words
stop_words = set(stopwords.words('english'))
df['tokens'] = df['tokens'].apply(lambda tokens: [token for token in tokens if token not in stop_words])

# Load the GloVe model

glove_input_file = '../../../Dataset/glove.6B.100d.txt'
word2vec_output_file = '../../../Dataset/glove.6B.100d.word2vec.txt'
glove2word2vec(glove_input_file, word2vec_output_file)

# Now load the model
glove_model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
# Convert words to vectors
df['vectors'] = df['tokens'].apply(lambda tokens: [glove_model[token] for token in tokens if token in glove_model])