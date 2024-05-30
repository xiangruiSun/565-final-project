import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import re

# DELTA = 1
# ALPHA = 3
# BETA = 3
# GAMMA = 0.5


# def popularity_calculation(raw_data):
    
#     tmp1 = DELTA * np.log10(raw_data[:,0] + 1)
#     tmp2 = ALPHA * raw_data[:,1] / (raw_data[:,0] + 1)
#     tmp3 = BETA * raw_data[:,2] / (raw_data[:,0] + 1)
#     tmp4 = GAMMA * raw_data[:,3] / (raw_data[:,0] + 1)
    
#     numerator = 10 * (tmp1 + tmp2 - tmp3 + tmp4)
#     denominator = DELTA + ALPHA + BETA + GAMMA
    
#     result =  numerator / denominator
    
#     return result.reshape(-1, 1)

DELTA = 1
ALPHA = 5
BETA = 2


def popularity_calculation(raw_data):
    
    tmp1 = DELTA * np.log10(raw_data[:,0] + 1)
    tmp2 = ALPHA * raw_data[:,1] / (raw_data[:,0] + 1)
    tmp3 = BETA * raw_data[:,3] / (raw_data[:,0] + 1)
    
    numerator = 10 * (tmp1 + tmp2 + tmp3)
    denominator = DELTA + ALPHA + BETA
    
    result =  numerator / denominator
    
    return result.reshape(-1, 1)

def get_popularity(dataset_path):
    
    df = pd.read_csv(dataset_path, usecols=['view_count', 'likes', 'dislikes', 'comment_count'])
    raw_data = df.values
    
    popularity_scores = popularity_calculation(raw_data)
    
    pscore_multiple = np.ones(raw_data.shape[0])
    for i in range(raw_data.shape[1]):
        if i == 2:
            continue
        pscore_multiple *= raw_data[:,i]

    non_zero_idx = np.where(pscore_multiple != 0)[0]

    # print(popularity_scores[non_zero_idx].max())

    return popularity_scores, non_zero_idx

def get_tags(dataset_path):
    
    df = pd.read_csv(dataset_path, usecols=['tags'])
    tags = np.array(df.values)
    return tags

def clean(tags, popularity_scores, number_data):
    
    print('Data Cleaning Processing...')
    valid_idx = []
    non_English = False
    is_None = False
    # print(valid_idx)
    for i in tqdm(range(tags.shape[0])):
        
        if (i == tags.shape[0] - 1) or (i == tags.shape[0] - 2):
            continue
        
        non_english_regex = re.compile(r'[^\x00-\x7F]+')
        if non_english_regex.search(tags[i][0]):
            non_English = True
                
        if tags[i][0] == '[None]':
            is_None = True
            
        if non_English or is_None:
            non_English = False
            is_None = False
            continue
        else :
            valid_idx.append(i)
            
            
    if number_data == 'Whole':
        tags_clean = tags[valid_idx]
        popularity_scores_clean = popularity_scores[valid_idx]
        
    else:
        random.shuffle(valid_idx)
        tags_clean = tags[valid_idx[:number_data]]
        popularity_scores_clean = popularity_scores[valid_idx[:number_data]]
    
    
    
    
    return tags_clean, popularity_scores_clean

def get_valid_data(dataset_path, number_data):
    
    popularity_scores_raw, non_zero_idx = get_popularity(dataset_path)
    tags_raw = get_tags(dataset_path)
    
    tags, popularity_scores = clean(tags_raw[non_zero_idx], popularity_scores_raw[non_zero_idx], number_data)
    
    # print(popularity_scores.max())
    # print(popularity_scores.min())
    # print(popularity_scores.mean())
    return tags, popularity_scores
    
    




if __name__ == '__main__':
    
    csv_file_path = 'dataset.csv'
    a = np.array([[0,97565,0,1931]])
    # score = popularity_calculation(a)
    # print(score)
    get_valid_data(csv_file_path, 'Whole')