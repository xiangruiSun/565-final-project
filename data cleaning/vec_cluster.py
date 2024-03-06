from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd

# Load the data from the Excel file
file_path = 'US_youtube_trending_data 2.csv'
data = pd.read_csv(file_path)
new_data = data.iloc[:150000]
# Extract the 'tags' column
# Assuming new_data is already loaded and contains the first 1000 rows
# Extract the 'tags' column from new_data instead of the entire dataset
tags_data = new_data['tags']

# Function to clean and vectorize the tags from a single row
def clean_and_vectorize_tags(tags):
    unique_tags = set(tags.split('|'))  # Split and remove duplicates
    sorted_tags = sorted(list(unique_tags))  # Sort
    return sorted_tags

# Apply the function to the 'tags' column of new_data
new_data['vectorized_tags'] = tags_data.apply(clean_and_vectorize_tags)

# Using MultiLabelBinarizer
mlb = MultiLabelBinarizer()
tags_mlb = mlb.fit_transform(new_data['vectorized_tags'])

# Using CountVectorizer
new_data['joined_tags'] = new_data['vectorized_tags'].apply(lambda x: ' '.join(x))
cv = CountVectorizer(tokenizer=lambda txt: txt.split())
tags_cv = cv.fit_transform(new_data['joined_tags'])

# Using TfidfVectorizer
tfidf = TfidfVectorizer(tokenizer=lambda txt: txt.split())
tags_tfidf = tfidf.fit_transform(new_data['joined_tags'])

# Dimensionality Reduction
svd = TruncatedSVD(n_components=100)  # Adjust n_components as needed
tags_reduced = svd.fit_transform(tags_tfidf)

# Clustering
kmeans = KMeans(n_clusters=10, random_state=42)  # Adjust n_clusters as needed
clusters = kmeans.fit_predict(tags_reduced)

# Silhouette Score
score = silhouette_score(tags_reduced, clusters)

(tags_mlb.shape, tags_cv.shape, tags_tfidf.shape, score)