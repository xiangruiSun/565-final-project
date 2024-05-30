import pandas as pd
import numpy as np

data = pd.read_csv('US_youtube_trending_data.csv', usecols=['categoryId','tags','view_count', 'likes', 'dislikes', 'comment_count'])
categotyid = np.array(data['categoryId'])

# Mapping of categoryId to Category
category_map = {
    1: "Film & Animation",
    2: "Autos & Vehicles",
    10: "Music",
    15: "Pets & Animals",
    17: "Sports",
    18: "Short Movies",
    19: "Travel & Events",
    20: "Gaming",
    21: "Videoblogging",
    22: "People & Blogs",
    23: "Comedy",
    24: "Entertainment",
    25: "News & Politics",
    26: "Howto & Style",
    27: "Education",
    28: "Science & Technology",
    29: "Nonprofits & Activism",
    30: "Movies",
    31: "Anime/Animation",
    32: "Action/Adventure",
    33: "Classics",
    34: "Comedy",
    35: "Documentary",
    36: "Drama",
    37: "Family",
    38: "Foreign",
    39: "Horror",
    40: "Sci-Fi/Fantasy",
    41: "Thriller",
    42: "Shorts",
    43: "Shows",
    44: "Trailers"
}

# Map categoryId to Category in the dataframe
data['Category'] = data['categoryId'].map(category_map)

gaming_data = data[data['Category'] == 'Gaming']
travel_data = data[data['Category'] == 'Travel & Events']

filtered_file_path1 = 'Travel.csv'
travel_data.to_csv(filtered_file_path1, index=False)

filtered_file_path2 = 'Gaming.csv'
gaming_data.to_csv(filtered_file_path2, index=False)
