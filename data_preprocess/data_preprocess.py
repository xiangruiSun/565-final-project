import pandas as pd
import numpy as np
from data_preprocess.data_clean import get_valid_data
# from data_clean import get_valid_data
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import random

import matplotlib.pyplot as plt

class Data_Preprocess:
    
    def __init__(self, dataset_path, vec_method = 'Transformer', number_data = 'Whole', loading = True):
        
        self.dataset_path = dataset_path
        self.vec_method = vec_method
        self.number_data = number_data
        self.loading = loading
        
        self.tags, self.scores = get_valid_data(self.dataset_path, self.number_data)
        # print(self.scores.max())

    def get_high_popularity_data(self, threshold):
        
        scaled_scores = self.pscore_standardization()
        
        idx = np.where(scaled_scores >= threshold)[0]
        
        tags_vec = self.vectorization(self.tags)
        
        tags_vec_with_high_score = tags_vec[idx]
        
        tags_raw = self.tags[idx]
        
        scores = scaled_scores[idx]
        
        return tags_vec_with_high_score, scores, tags_raw

    def get_low_popularity_data(self, number_data, threshold):
        
        scaled_scores = self.pscore_standardization()
        
        idx = np.where(scaled_scores <= threshold)[0]
        
        tags_vec = self.vectorization(self.tags)
        
        random.shuffle(idx)
        sample_idx = idx[:number_data]
        tags_vec_with_low_score = tags_vec[sample_idx]
        scores = scaled_scores[sample_idx]
        tags_raw = self.tags[sample_idx]
        
        
        return tags_vec_with_low_score, scores, tags_raw
    
    def split_config(self):
        
        tags_vec = self.vectorization(self.tags)
        x_train, x_test, y_train, y_test = train_test_split(tags_vec, self.scores, test_size=0.2, random_state=4)
        x_train_norm, x_test_norm = self.normalization(x_train, x_test)
        
        return x_train_norm, x_test_norm, np.ravel(y_train), np.ravel(y_test)
    
    def normalization(self, x_train, x_test):
        
        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(x_train)
        
        X_test_scaled = scaler.transform(x_test)
        
        return X_train_scaled, X_test_scaled
    
    def vectorization(self, tags):
        
        if self.vec_method == 'Transformer':
            print("Vectorization by Using Transformer")
            if not self.loading:
                tags_vec = self.vectorization_transformer(tags)
                np.save('tags_vec_trans.npy', tags_vec)
            else:
                tags_vec = np.load('tags_vec_trans.npy')
        
        elif self.vec_method == 'Count':
            print("Vectorization by Using CountVectorizer")
            if not self.loading:
                tags_vec = self.Countvectorizer(tags)
                np.save('tags_vec_count.npy', tags_vec)
            else:
                tags_vec = np.load('tags_vec_count.npy')
        
        elif self.vec_method == 'TFIDF':
            print("Vectorization by Using TFIDF")
            if not self.loading:
                tags_vec = self.TFIDF(tags)
                np.save('tags_vec_tfidf.npy', tags_vec)
            else:
                tags_vec = np.load('tags_vec_tfidf.npy')
            
            
        return tags_vec
         
    def vectorization_transformer(self, tags):
        
        model = SentenceTransformer('bert-base-nli-mean-tokens')

        vec_arr = np.zeros((tags.shape[0], 768))
        
        for i in tqdm(range(tags.shape[0])):
    
            sub_tags = str(tags[i]).split('|')
            
            for j in range(len(sub_tags)):
                
                vec_arr[i] += model.encode([sub_tags[j]])[0]
                
            vec_arr[i] /= len(sub_tags)
            
        
        return vec_arr   

    def Countvectorizer(self, tags):
        
        vectorizer = CountVectorizer()
        
        tags_list = []
        
        for i in tqdm(range(tags.shape[0])):
    
            sub_tags = str(tags[i]).replace("|", " ")
            tags_list.append(sub_tags)
        
        tag_vec =  vectorizer.fit_transform(tags_list) 
        
        return tag_vec.toarray()
    
    def TFIDF(self, tags):

        vectorizer = TfidfVectorizer()
        
        tags_list = []
        
        for i in tqdm(range(tags.shape[0])):
    
            sub_tags = str(tags[i]).replace("|", " ")
            tags_list.append(sub_tags)
        
        tag_vec =  vectorizer.fit_transform(tags_list) 
        
        return tag_vec.toarray()  

    def tags_distribution_visualization(self):
        
        tags_vec = self.vectorization(self.tags)
        pca = PCA(n_components=2)
        pca.fit(tags_vec)
        data_pca = pca.transform(tags_vec)
        plt.title('Tags Data Distribution by Using ' + self.vec_method + ' Encoding')
        plt.scatter(data_pca[:, 0], data_pca[:, 1])
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.show()
    
    def pscore_standardization(self):
        
        scaler = MinMaxScaler()
        scaled_scores = scaler.fit_transform(self.scores.reshape(-1, 1)).flatten() * 10
        scaled_scores = scaled_scores.astype(int)
        
        return scaled_scores
         
    def pscore_raw_distribution_visualization(self):
        
        scaled_scores = self.scores.astype(int).ravel()

        fig, ax = plt.subplots(1, 2, figsize = (11,5))
       
        ax[0].hist(scaled_scores, bins=range(max(scaled_scores)+2), align='left', rwidth=0.8)
        ax[0].set_title('Popularity Raw Score Number Distribution')
        ax[0].set_xlabel('Popularity Score')
        ax[0].set_ylabel('Numbers')
        ax[1].hist(scaled_scores, bins=range(max(scaled_scores)+2), density= True, align='left', rwidth=0.8)
        ax[1].set_title('Popularity Raw Score Density Distribution')
        ax[1].set_xlabel('Popularity Score')
        ax[1].set_ylabel('Density')

        plt.show()

    def pscore_scaled_distribution_visualization(self):

        scaled_scores = self.pscore_standardization()
        fig, ax = plt.subplots(1, 2, figsize = (11,5))
        
        ax[0].hist(scaled_scores, bins=range(max(scaled_scores)+2), align='left', rwidth=0.8)
        ax[0].set_title('Popularity Scaled Score Number Distribution')
        ax[0].set_xlabel('Popularity Score')
        ax[0].set_ylabel('Numbers')
        ax[1].hist(scaled_scores, bins=range(max(scaled_scores)+2), density= True, align='left', rwidth=0.8)
        ax[1].set_title('Popularity Scaled Score Density Distribution')
        ax[1].set_xlabel('Popularity Score')
        ax[1].set_ylabel('Density')

        plt.show()

if __name__ == '__main__':
    
    csv_file_path = 'dataset.csv'
    number_data = 1000
    vec_method = 'TFIDF'
    vec_method = 'Count'
    # vec_method = 'Transformer'
    data_preprocesser = Data_Preprocess(csv_file_path, vec_method, number_data)
    data_preprocesser.pscore_distribution_visualization()
    
    # a = data_preprocesser.get_high_popularity_data()
    # print(a.shape)
    # data_preprocesser.vectorization()
    # train_data, test_data, train_label, test_label = data_preprocesser.split_config()
    
    # print(train_data.shape)
    # print(train_label.shape)
    # print(test_data.shape)
    # print(test_label.shape)
    
    
