import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt

# Load the data from the CSV file
file_path = 'US_youtube_trending_data.csv'
data = pd.read_csv(file_path)

# Set a seed and randomly select 5000 rows
np.random.seed(42)
random_indices = np.random.choice(data.index, size=5000, replace=False)
new_data = data.loc[random_indices]

# Extract the 'tags' column and clean/vectorize it
tags_data = new_data['tags']
def clean_and_vectorize_tags(tags):
    unique_tags = set(tags.split('|'))
    sorted_tags = sorted(list(unique_tags))
    return sorted_tags
new_data['vectorized_tags'] = tags_data.apply(clean_and_vectorize_tags)
new_data['joined_tags'] = new_data['vectorized_tags'].apply(lambda x: ' '.join(x))

# Using TfidfVectorizer
tfidf = TfidfVectorizer(tokenizer=lambda txt: txt.split())
tags_tfidf = tfidf.fit_transform(new_data['joined_tags'])

# Dimensionality Reduction
svd = TruncatedSVD(n_components=100)
tags_reduced = svd.fit_transform(tags_tfidf)

# Initialize lists to store metrics
silhouette_scores = []
db_indices = []
calinski_harabasz_scores = []
ks = range(20, 40) #modify (2,100)

for k in ks:
    kmeans = KMeans(n_clusters=k, random_state=42).fit(tags_reduced)
    labels = kmeans.labels_
    
    silhouette_scores.append(silhouette_score(tags_reduced, labels))
    db_indices.append(davies_bouldin_score(tags_reduced, labels))
    calinski_harabasz_scores.append(calinski_harabasz_score(tags_reduced, labels))

# Plotting the metrics
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.plot(ks, silhouette_scores, marker='o', linestyle='-', color='blue')
plt.title('Silhouette Score')
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.xticks(ks)
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(ks, db_indices, marker='o', linestyle='-', color='red')
plt.title('Davies-Bouldin Index')
plt.xlabel('Number of Clusters')
plt.ylabel('Index')
plt.xticks(ks)
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(ks, calinski_harabasz_scores, marker='o', linestyle='-', color='green')
plt.title('Calinski-Harabasz Index')
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.xticks(ks)
plt.grid(True)

plt.tight_layout()
plt.show()
