from warnings import filterwarnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from sklearn.manifold import TSNE
from minisom import MiniSom


filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

df = pd.read_csv("WikipediaEvents.csv", index_col=0)
print("shape of dataset: {}" .format(df.shape))
df.head()

df["text"]

def clean_text(text):
    # lowercasing (Case Folding)
    text = text.str.lower()
    # removing punctuations, numbers, and newline characters
    text = text.str.replace(r'[^\w\s]', '', regex=True)
    text = text.str.replace("\n", '', regex=True)
    text = text.str.replace('\d', '', regex=True)
    return text


df["text"] = clean_text(df["text"])
df["text"]

nltk.download("stopwords")
stop_words = stopwords.words("english")

print(stop_words)

def remove_stopwords(text):
    text = text.apply(lambda x: " ".join(word for word in str(x).split() if word not in stop_words))
    return text


df["text"] = remove_stopwords(df["text"])
df["text"]

nltk.download('punkt')
df["text"].apply(lambda x: TextBlob(x).words)


# load glove pretrained model
PATH_in = "/mnt/9636D17436D15639/University/CE/Deep Learning/Dr Safabakhsh/Spring 2024/Dataset/glove.6B.100d.txt"
PATH_out = "/mnt/9636D17436D15639/University/CE/Deep Learning/Dr Safabakhsh/Spring 2024/Dataset/glove.word2vec.txt"

glove2word2vec(PATH_in, PATH_out)

glove_model = KeyedVectors.load_word2vec_format(PATH_out, binary=False)

print(glove_model)

def convert_to_vector(text):
    return [glove_model[word] for word in text.split() if word in glove_model]

df["vectors"] = df['text'].apply(convert_to_vector)
df.to_csv("word2vec_out.csv", index=False)
df["vectors"]

data = df.values
print(data.shape)
# data

# extract the word vectors and their corresponding words
words = []
vectors = []
for _, row in df.iterrows():
    for word, vector in zip(row["text"].split(), row["vectors"]):
        words.append(word)
        vectors.append(vector)

# use t-SNE to reduce the dimension of the vectors to 2D
vectors = np.array(vectors)
tsne = TSNE(n_components=2, random_state=0)
vectors_2d = tsne.fit_transform(vectors)

# vrctors_2d = vectors.reshape(-1, 2)



from nltk.tokenize import word_tokenize
sample_questions = [
                        "Who won the 2022 soccer world cup?",
                        "When did Sweden join NATO?",
                        "Who joined NATO in 2023?",
                        "Who joined NATO in 2024?",
                        "Which is the 31st member of NATO?",
                        "Which is the 32nd member of NATO?",
                        "Who won the Cricket World Cup in 2023?",
                        "Who defeated India in Cricket World Cup final in 2023?",
                        "Name the former prime minister of Japan that was assassinated in 2022?",
                        "When did Chandrayaan-3 land near the south pole of the Moon?",
                        "Where did Chandrayaan-3 land on the Moon?",
                        "Who acquired Twitter in 2022?",
                        "Who owns Twitter?",
                        "Who acquired Activision Blizzard in 2023?"
                   ]

#print(len(sample_questions))

# tokenized questions
tokenized_questions = [word_tokenize(i) for i in sample_questions]
# print(len(tokenized_questions))

for i in tokenized_questions:
    print(i)

sample_questions_vector = np.array([])

for i in sample_questions:
    sample_questions_vector = np.append(sample_questions_vector, convert_to_vector(i))


print(sample_questions_vector.shape)
print(sample_questions_vector)


word2vec_output_file = "/mnt/9636D17436D15639/University/CE/Deep Learning/Dr Safabakhsh/Spring 2024/Dataset/glove.word2vec.txt"

glove_model_question = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

# Create a DataFrame with sample questions
df_question = pd.DataFrame({"text": sample_questions})

# Apply the convert_to_vector function to each question
df_question["vectors"] = df_question["text"].apply(convert_to_vector)

# Save the DataFrame to a CSV file
df_question.to_csv("sample_questions_vector.csv", index=False)

# Print the vectors for each question
print(df_question["vectors"])


# extract the word vectors and their corresponding words
words_question = []
vectors_question = []
for _, row in df_question.iterrows():
    for word, vector in zip(row["text"].split(), row["vectors"]):
        words_question.append(word)
        vectors_question.append(vector)

# use t-SNE to reduce the dimension of the vectors to 2D
vectors_question = np.array(vectors_question)
tsne_question = TSNE(n_components=2, random_state=0)
vectors_2d_question = tsne.fit_transform(vectors_question)

# vrctors_2d = vectors.reshape(-1, 2)

plt.figure(figsize=(10, 10))
cmap = plt.get_cmap("Spectral")
colors = cmap(np.linspace(0, 1, len(vectors_2d_question)))
plt.scatter(vectors_2d_question[:, 0], vectors_2d_question[:, 1], edgecolors='k', c=colors, alpha=1, marker='.', s=100)
plt.grid(True, color='gray', linestyle='--', linewidth=0.5)

for word, (x, y) in zip(words_question, vectors_2d_question):
    plt.text(x, y, word, fontsize=10)

plt.title("t-SNE visualization of question word vectors", fontsize=20)
plt.xlabel("X1", fontsize=15)
plt.ylabel("X2", fontsize=15)

plt.show()


som_shape = (2, 3)
som = MiniSom(som_shape[0], som_shape[1], vectors_2d_question.shape[1], sigma=0.5, learning_rate=0.5,
            neighborhood_function='gaussian', random_seed=10)

som.train_batch(vectors_2d_question, 50000, verbose=True)

winner_coordinates = np.array([som.winner(x) for x in vectors_2d_question]).T
cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)





plt.figure(figsize=(8, 6))
for c in np.unique(cluster_index):
    plt.scatter(vectors_2d_question[cluster_index == c, 0],
                vectors_2d_question[cluster_index == c, 1], label='cluster='+str(c), alpha=0.7)
    plt.title("Question clusters", fontsize=20)
    plt.xlabel("X1", fontsize=15)
    plt.ylabel("X2", fontsize=15)

for centroid in som.get_weights():
    plt.scatter(centroid[:, 0], centroid[:, 1], marker='x',
                s=100, linewidths=2, color='k', label='centroid')
    



def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)

    # Get Euclidean/L2 norm
    norm_vector1 = np.sqrt(np.sum(vector1**2))
    norm_vector2 = np.sqrt(np.sum(vector2**2))

    return dot_product / (norm_vector1 * norm_vector2)