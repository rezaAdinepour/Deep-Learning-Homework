import requests
import os

# Define the URL to download GloVe embeddings
glove_url = 'http://nlp.stanford.edu/data/glove.6B.zip'

# Define the directory where you want to save the downloaded file
download_dir = 'glove/'

# Create the directory if it doesn't exist
os.makedirs(download_dir, exist_ok=True)

# Define the filename for the downloaded file
filename = os.path.join(download_dir, 'glove.6B.zip')

# Download the file
print("Downloading GloVe embeddings...")
response = requests.get(glove_url)
with open(filename, 'wb') as f:
    f.write(response.content)

print("GloVe embeddings downloaded successfully!")
