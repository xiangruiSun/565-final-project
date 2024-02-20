import pandas as pd

# Load the data from the Excel file
file_path = '/Users/sunxiangrui/Desktop/youtube trend data.csv'
data = pd.read_csv(file_path)

# Extract the 'tags' column
tags_data = data['tags']

# Function to clean and vectorize the tags from a single row
def clean_and_vectorize_tags(tags):
    # Split the tags by '|' and remove duplicates by converting to a set
    unique_tags = set(tags.split('|'))
    # Convert back to list and sort
    sorted_tags = sorted(list(unique_tags))
    return sorted_tags

# Apply the function to the 'tags' column
data['vectorized_tags'] = data['tags'].apply(clean_and_vectorize_tags)

# Now we'll implement three different ways to vectorize these tags

# Method 1: Using MultiLabelBinarizer from sklearn to create a binary array indicating the presence of a tag
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()
tags_mlb = mlb.fit_transform(data['vectorized_tags'])

# Method 2: Using CountVectorizer from sklearn to create a term frequency vector for the tags
from sklearn.feature_extraction.text import CountVectorizer

# We join the cleaned tags with space to use them with CountVectorizer
data['joined_tags'] = data['vectorized_tags'].apply(lambda x: ' '.join(x))
cv = CountVectorizer(tokenizer=lambda txt: txt.split())
tags_cv = cv.fit_transform(data['joined_tags'])

# Method 3: Using TF-IDF Vectorizer from sklearn to create a term frequency-inverse document frequency vector
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(tokenizer=lambda txt: txt.split())
tags_tfidf = tfidf.fit_transform(data['joined_tags'])

# Show the shapes of the created vectors to confirm
(tags_mlb.shape, tags_cv.shape, tags_tfidf.shape)

from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Step 1: Vectorization is already done (e.g., using TF-IDF)

# Step 2: Dimensionality Reduction
svd = TruncatedSVD(n_components=100)  # Number of components to keep
tags_reduced = svd.fit_transform(tags_tfidf)

# Step 3: Clustering
# Determine the number of clusters (using Elbow method, Silhouette score, etc.)
# As an example, let's assume we decide on 10 clusters
kmeans = KMeans(n_clusters=10, random_state=42)
clusters = kmeans.fit_predict(tags_reduced)

# Optionally, evaluate the silhouette score
score = silhouette_score(tags_reduced, clusters)
