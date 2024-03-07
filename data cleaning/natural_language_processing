import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from gensim.models import Word2Vec

# 1. Load the dataset
file_path = 'US_youtube_trending_data.csv'  # Use the uploaded file path
data = pd.read_csv(file_path)

# Set a seed and randomly select 5000 rows
np.random.seed(42)
random_indices = np.random.choice(data.index, size=5000, replace=False)
new_data = data.loc[random_indices]
tags_data = new_data['tags']

# Helper function to tokenize tags
def tokenize_tags(tags_str):
    return tags_str.split('|')

# 2. Vectorization methods
# Count Vectorization
count_vec = CountVectorizer(tokenizer=tokenize_tags)
X_count1 = count_vec.fit_transform(tags_data)
X_count = X_count1.toarray()

# TF-IDF Vectorization
tfidf_vec = TfidfVectorizer(tokenizer=tokenize_tags)
X_tfidf1 = tfidf_vec.fit_transform(tags_data)
X_tfidf = X_tfidf1.toarray()

# Word Embeddings
# Assuming the dataset is large and training might be computationally expensive,
# we will use a simpler model like Word2Vec on a subset of data for demonstration.
sentences = tags_data.apply(tokenize_tags)
word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
def document_vector(word_list):
    # Remove out-of-vocabulary words and sum the remaining vectors
    return np.sum([word2vec_model.wv[word] for word in word_list if word in word2vec_model.wv], axis=0)

X_word2vec = np.array(list(map(document_vector, sentences)))

# 3. Visualizing the vectorizations
def plot_embeddings(X, title):
    pca = PCA(n_components=50)
    X_pca = pca.fit_transform(X.todense() if hasattr(X, 'todense') else X)
    X_tsne = TSNE(n_components=2, perplexity=50).fit_transform(X_pca)
    
    plt.figure(figsize=(10, 5))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=1)
    plt.title(title)
    plt.show()

# Run the visualization for each vectorization method
plot_embeddings(X_count, 'Count Vectorization')
plot_embeddings(X_tfidf, 'TF-IDF Vectorization')
plot_embeddings(X_word2vec, 'Word Embeddings')

# 4. Conclude the best method
# The conclusion will be based on the visual clusters formed in the t-SNE plots and potentially other metrics if computed.
