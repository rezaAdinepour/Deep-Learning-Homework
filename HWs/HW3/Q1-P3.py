from warnings import filterwarnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nltk
from PIL import Image
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate, train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from textblob import Word, TextBlob
from wordcloud import WordCloud
from utils import*
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from textblob import Word
from wordcloud import WordCloud
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from textblob import Word
from wordcloud import WordCloud
from nltk.corpus import stopwords
from minisom import MiniSom
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors


from sklearn.manifold import TSNE

glove_input_file = "../../../Dataset/glove.6B.100d.txt"
word2vec_output_file = "../../../Dataset/glove.word2vec.txt"
glove2word2vec(glove_input_file, word2vec_output_file)

glove_model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)


def convert_to_vector(text):
    """
    Convert words to vectors using a pre-trained GloVe model.

    This function converts each word in the text to its corresponding vector representation
    using a pre-trained GloVe model. If a word is not in the model's vocabulary, it is ignored.

    Parameters:
    text (str): The text to convert.

    Returns:
    list: A list of vectors representing the words in the text.
    """
    return [glove_model[word] for word in text.split() if word in glove_model]


def remove_stopwords(text):
    """
    Remove stopwords from text data.

    This function filters out common stopwords from the text data. 
    Stopwords are removed based on the NLTK's English stopwords list.

    Parameters:
    text (pandas.Series): A pandas Series containing text data.

    Returns:
    pandas.Series: A pandas Series with stopwords removed from the text.
    """
    # Removing stopwords
    text = text.apply(lambda x: " ".join(word for word in str(x).split() if word not in stop_words))
    return text







def clean_text(text):
    """
    Clean and preprocess text data.

    This function performs several cleaning operations on text data:
    - Lowercases the text (Case Folding)
    - Removes punctuation
    - Removes numbers
    - Removes newline characters

    Parameters:
    text (pandas.Series): A pandas Series containing text data.

    Returns:
    pandas.Series: A pandas Series with cleaned text.
    """
    # Lowercasing (Case Folding)
    text = text.str.lower()
    # Removing punctuations, numbers, and newline characters
    text = text.str.replace(r'[^\w\s]', '', regex=True)
    text = text.str.replace("\n", '', regex=True)
    text = text.str.replace('\d', '', regex=True)
    return text





def remove_rare_words(df, column_name, n_rare_words=1000):
    """
    Remove rare words from a specified column in a pandas DataFrame.

    This function identifies and removes the least frequently occurring words
    in the text data. It is useful for removing rare words that might not contribute
    significantly to the analysis or modeling.

    Parameters:
    df (pandas.DataFrame): A pandas DataFrame containing the text data.
    column_name (str): The name of the column in the DataFrame to clean.
    n_rare_words (int): The number of least frequent words to remove.

    Returns:
    pandas.DataFrame: A DataFrame with rare words removed from the specified column.
    """
    # Identifying the rare words
    freq = pd.Series(' '.join(df[column_name]).split()).value_counts()
    rare_words = freq[-n_rare_words:]

    # Removing the rare words
    df[column_name] = df[column_name].apply(lambda x: " ".join(word for word in x.split() if word not in rare_words))
    return df


def apply_lemmatization(df, column_name):
    """
    Apply lemmatization to a specified column in a pandas DataFrame.

    This function performs lemmatization on the text data in the specified column.
    Lemmatization involves reducing each word to its base or root form.

    Parameters:
    df (pandas.DataFrame): A pandas DataFrame containing the text data.
    column_name (str): The name of the column in the DataFrame to process.

    Returns:
    pandas.DataFrame: A DataFrame with lemmatized text in the specified column.
    """
    # Applying lemmatization
    df[column_name] = df[column_name].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

    return df

def plot_tf_and_wordcloud(df, column_name, tf_threshold=2000, max_font_size=50, max_words=100, background_color="black"):
    """
    Calculate term frequency (TF) and generate a word cloud for a specified column in a pandas DataFrame.

    This function performs two main tasks:
    1. Term Frequency Calculation and Bar Chart: Calculates the frequency of each word in the specified column and plots a bar chart for words with a frequency above a certain threshold.
    2. Word Cloud Generation: Generates and displays a word cloud based on the text in the specified column.

    Parameters:
    df (pandas.DataFrame): A pandas DataFrame containing the text data.
    column_name (str): The name of the column to analyze.
    tf_threshold (int): The threshold for term frequency to be included in the bar chart.
    max_font_size (int): Maximum font size for the word cloud.
    max_words (int): The maximum number of words for the word cloud.
    background_color (str): Background color for the word cloud.

    Returns:
    None: This function only plots the results and does not return any value.
    """
    # 1. Term Frequency Calculation and Bar Chart
    tf = df[column_name].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
    tf.columns = ["words", "tf"]
    high_tf = tf[tf["tf"] > tf_threshold]
    high_tf.plot.bar(x="words", y="tf", title="Term Frequency Bar Chart")
    plt.show()

    # 2. Word Cloud Generation
    text = " ".join(i for i in df[column_name])
    wordcloud = WordCloud(max_font_size=max_font_size, max_words=max_words, background_color=background_color).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.title("Word Cloud")
    plt.axis("off")
    plt.show()






filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.2f' % x)



df = pd.read_csv("WikipediaEvents.csv", index_col=0)
print("shape of dataset: {}" .format(df.shape))
df.head()


df["text"]


df["text"] = clean_text(df["text"])
df["text"]

nltk.download("stopwords")
stop_words = stopwords.words("english")

df["text"] = remove_stopwords(df["text"])
df["text"]

df = remove_rare_words(df, 'text', 1000)
df["text"]

nltk.download('punkt')
df["text"].apply(lambda x: TextBlob(x).words)

nltk.download('wordnet')
nltk.download('omw-1.4')

df = apply_lemmatization(df, 'text')
df['text']

plot_tf_and_wordcloud(df, "text", tf_threshold=30)

df['text']




df['vectors'] = df['text'].apply(convert_to_vector)
df.to_csv('preprocessed_data.csv', index=False)




# Extract the word vectors and their corresponding words
words = []
vectors = []
for _, row in df.iterrows():
    for word, vector in zip(row['text'].split(), row['vectors']):
        words.append(word)
        vectors.append(vector)

# Use t-SNE to reduce the dimension of the vectors to 2D
# Convert list of lists to numpy array
vectors = np.array(vectors)

tsne = TSNE(n_components=2, random_state=0)
vectors_2d = tsne.fit_transform(vectors)

# Plot the 2D vectors
# plt.figure(figsize=(10, 10))
# plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], edgecolors='k', c='r')
# for word, (x, y) in zip(words, vectors_2d):
#     plt.text(x, y, word)
# plt.show()


# Importing necessary libraries
# Define the dimensions of the SOM grid
grid_rows = 10
grid_cols = 10

# Define the input dimension (same as the dimension of the word vectors)
input_dim = vectors.shape[1]

# Initialize the SOM network
som = MiniSom(grid_rows, grid_cols, input_dim, sigma=0.5, learning_rate=0.5)

# Initialize the weights of the SOM network with random values
som.random_weights_init(vectors)

# Train the SOM network
som.train_random(vectors, 100)  # You can adjust the number of iterations as needed

# Get the coordinates of the BMUs (Best Matching Units) for each input vector
bmu_coords = np.array([som.winner(x) for x in vectors])

# Plot the 2D vectors with SOM cluster assignments
plt.figure(figsize=(10, 10))
plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], c=bmu_coords.flatten(), edgecolors='k')
for word, (x, y) in zip(words, vectors_2d):
    plt.text(x, y, word, alpha=0.5)
plt.title('SOM Clustering')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.colorbar(label='SOM Cluster')
plt.grid()
plt.show()

