import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
from tqdm import tqdm

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Read the dataset
df = pd.read_csv('WikipediaEvents.csv')

# Select the column containing events
events_column = 'text'  # Replace 'Event_Column_Name' with the actual column name
events = df[events_column]

# Define stop words
stop_words = set(stopwords.words('english'))

# Preprocessing function
def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stop words and punctuation
    tokens = [word.lower() for word in tokens if word.isalnum() and word.lower() not in stop_words]
    
    # Lemmatize using spaCy
    tokens = [token.lemma_ for token in nlp(" ".join(tokens))]
    
    return tokens

# Apply preprocessing to all events
preprocessed_events = []
for event in tqdm(events):
    preprocessed_event = preprocess_text(event)
    preprocessed_events.append(preprocessed_event)

# Load GloVe embeddings
embeddings_index = {}
with open('glove.6B.100d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

# Convert words to vectors using GloVe embeddings
def words_to_vectors(words, embeddings_index, embedding_dim):
    vectors = []
    for word in words:
        if word in embeddings_index:
            vectors.append(embeddings_index[word])
        else:
            vectors.append(np.zeros(embedding_dim))  # Use zeros for out-of-vocabulary words
    return vectors

embedding_dim = 100  # Adjust according to the dimensions of your GloVe embeddings
event_vectors = [words_to_vectors(event, embeddings_index, embedding_dim) for event in tqdm(preprocessed_events)]
