RATINGS_PATH = 'datasets/ml-latest-small/ratings.csv'
ITEMS_PATH = ''


import pandas as pd
from surprise import Reader, Dataset
from surprise import KNNWithMeans, SVD

import numpy as np
import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.metrics import precision_score




class GRSD():

    def __init__(self, rating_data='', data_frame='', item_data=''):
        ''' Sets ratings, sim_options, trainset.
            Cleans item_data dataframe, in this case, based on MovieLens dataset.
            Sets items.
        '''
        if rating_data:
            reader = Reader(line_format='user item rating timestamp', sep=',')
            self.ratings = Dataset.load_from_file(rating_data, reader)
            self.trainset = self.ratings.build_full_trainset()
            self.sim_options = {'name': 'cosine','user_based': False}
            self.df_ratings = pd.read_csv(rating_data, low_memory=False, names=['userId', 'movieId', 'rating','timestamp'])
        elif not data_frame.empty:
            reader = Reader(rating_scale=(0, 5))
            self.ratings = Dataset.load_from_df(data_frame[['userId', 'movieId', 'rating']], reader)
            self.trainset = self.ratings.build_full_trainset()
            self.sim_options = {'name': 'cosine','user_based': False}
        if item_data:
            self.items = pd.read_csv(item_data, low_memory=False)
            self.items['year'] = self.items['title'].apply(lambda x: x[-5:-1])
            self.items['title'] = self.items['title'].apply(lambda x: x[:-7])
            self.items['genres'] = self.items['genres'].apply(lambda x: x.replace('|',', '))


    def random_group(self, n):
        ''' Generates a random group size n.
            Sets users_list.
            Returns the group.
        '''
        self.users_list = list(self.df_ratings['userId'])
        random_group = random.sample(self.users_list,n)

        return random_group


    def set_k(self, k_value=''):
        ''' Sets the prediction algorithm used. The default is SVD.
        '''
        if k_value:
            algo = KNNWithMeans(k=k_value, sim_options=self.sim_options)
            self.algo = algo
            self.algo.fit(self.trainset)
        else:
            algo = SVD()
            self.algo = algo
            self.algo.fit(self.trainset)


    def set_testset(self, group):
        ''' Sets which items are considered as candidate items for a group, if the members are provided.
            Updates testset.
            Returns the updated testset.
        '''
        if group:
            user_ratings = self.trainset.ur
            items_ids = list(self.items['movieId'])
            global_mean=self.trainset.global_mean
            my_testset = []
            
            for user in group:
                iuid = self.trainset.to_inner_uid(str(user))
                for item in items_ids:
                    is_in = False
                    for rating in user_ratings[iuid]:
                        if int(item) == int(self.trainset.to_raw_iid(int(rating[0]))):
                            is_in = True
                            break
                    if not is_in:
                        my_tuple = (str(user),str(item),global_mean)
                        my_testset.append(my_tuple)
                        
            self.testset = my_testset
        else:
            testset = self.trainset.build_anti_testset()
            self.testset = testset

        return self.testset


    def predict_ratings(self,group=''):
        ''' Predicts ratings for all pairs (u, i) that are NOT in the training set. In other words, predicts ratings from candidate items.
            Sets predictions
        '''
        testset = self.set_testset(group)
        predictions = self.algo.test(testset)
        self.predictions = predictions


    def set_profile_items(self, group):
        ''' Items that were rated for AT LEAST ONE group member will compound the group profile.
            Sets group_sparse_mtx, profile_items
        '''
        metadata = pd.read_csv(RATINGS_PATH, low_memory=False, names=['userId', 'movieId', 'rating','timestamp'])
        metadata = metadata.drop(columns="timestamp")

        metadata_filtered = metadata[metadata.userId.isin(group)]

        self.group_sparse_mtx = pd.pivot_table(metadata_filtered, values='rating', index=['userId'], columns=['movieId'], fill_value=0)
        self.profile_items = list(self.group_sparse_mtx)


    ### You MUST call self.set_profile_items() before
    def set_candidate_items(self):
        ''' Items that were NOT rated by any group member will be candidates for recommendation.
            Sets group_sparse_mtx, profile_items
        '''
        candidate_items = []
        for item in self.items.iterrows():
        #     get the Id of each item in items dataframe
            if item[1].values[0] not in self.profile_items:
                candidate_items.append(item[1].values[0])
        self.candidate_items = candidate_items


    def calc_similarity_matrix(self):
        ''' Calculates the items similarity matrix using cosine similarity. This function was developed based on MovieLens dataset, using titles and genres.
            Sets cosine_sim_movies_title, cosine_sim_movies_genres
        '''
        #Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
        tfidf = TfidfVectorizer(stop_words='english')
        
        #Replace NaN with an empty string
        self.items['title'] = self.items['title'].fillna('')
        self.items['genres'] = self.items['genres'].fillna('')
        
        #Construct the required TF-IDF matrix by fitting and transforming the data
        tfidf_matrix_title = tfidf.fit_transform(self.items['title'])
        tfidf_matrix_genres = tfidf.fit_transform(self.items['genres'])
        
        #Compute the cosine similarity matrix
        self.cosine_sim_movies_title = cosine_similarity(tfidf_matrix_title, tfidf_matrix_title)
        self.cosine_sim_movies_genres = cosine_similarity(tfidf_matrix_genres, tfidf_matrix_genres)


    def get_similar_items(self, references, title_weight=0.8, k=10):
        ''' Searches for the top-k most similar items in candidate items to a given reference list. This function is based on MovieLens dataset.
            Returns a list of items.
        '''
        recs = []
        for item in references:
            # Get the pairwsie similarity scores of all movies with that movie
            movie_idx = int(self.items[self.items['movieId']==item['movieID']].index[0])
            sim_scores_title = list(enumerate(self.cosine_sim_movies_title[movie_idx]))
            sim_scores_genres = list(enumerate(self.cosine_sim_movies_genres[movie_idx]))
            
            # Calculate total similarity based on title and genres
            total_sim_score = []
            for i in range(len(sim_scores_title)):
                aux = (sim_scores_title[i][1]*title_weight) + (sim_scores_genres[i][1]*(1-title_weight))
                total_sim_score.append((i, aux))
                
            # Sort the movies based on the similarity scores
            total_sim_score = sorted(total_sim_score, key=lambda x: x[1], reverse=True)
            
            candidates_sim_score = []
            for sim_item in total_sim_score:
                if self.items.loc[sim_item[0]].values[0] not in self.profile_items:
                    candidates_sim_score.append(sim_item)
            
            # Get the scores of the top-k most similar movies
            k = k + 1
            candidates_sim_score = candidates_sim_score[1:k]
            recs.append(candidates_sim_score)
            
        return recs


    def get_relevance_score(self, recs, references):
        ''' Calculates the relevance of recommendations.
            Creates a dictionary for better manipulation of data, containing: 
                movie_id, movie_title, movie_genres, movie_similarity and movie_relevance. This function is based on MovieLens dataset.
            Returns a dict sorted by movie_relevance.
        '''
        count = 0
        recs_dict = []
        for reference in references:
        #     print('Referência: {}\t gêneros: {}'.format(refinedMyAlgo.movies[refinedMyAlgo.movies['movieId']==reference['movieID']].values[0][1], refinedMyAlgo.movies[refinedMyAlgo.movies['movieId']==reference['movieID']].values[0][2]))

            for movie in recs[count]:
                aux = {}

                movie_id = self.items.loc[movie[0]].values[0]
                movie_title = self.items.loc[movie[0]].values[1]
                movie_genres = self.items.loc[movie[0]].values[2]
                movie_similarity = movie[1]
                movie_relevance = round(((reference['rating']/5.0)+movie_similarity)/2, 3)

                aux['movie_id'] = movie_id
                aux['movie_title'] = movie_title
                aux['movie_genres'] = movie_genres
                aux['movie_similarity'] = movie_similarity
                aux['movie_relevance'] = movie_relevance

                recs_dict.append(aux)

        #         print('\tSim: {},\trelevance: {},\tmovieId: {},\ttitle: {}'.format(aux['movie_similarity'], aux['movie_relevance'], aux['movie_id'], aux['movie_title']))

            count=count+1

        recs_dict = sorted(recs_dict, key = lambda i: i['movie_relevance'],reverse=True)

        return recs_dict


    def calc_distance_item_in_list(self, item, this_list, title_weight=0.8):
        ''' Calculates the total distance of an item in relation to a given list.
            Returns the total distance.
        '''
        idx_i = int(self.items[self.items['movieId']==int(item['movie_id'])].index[0])

        total_dist = 0
        for movie in this_list:
            
            idx_j = int(self.items[self.items['movieId']==int(movie['movie_id'])].index[0])

            sim_i_j = (self.cosine_sim_movies_title[idx_i][idx_j]*title_weight) + (self.cosine_sim_movies_genres[idx_i][idx_j]*(1-title_weight))
            dist_i_j = 1 - sim_i_j
            total_dist = total_dist + dist_i_j

        result = total_dist/len(this_list)

        return result


    def calc_diversity_score(self, actual_list, candidates_list, alfa=0.5):
        '''
            This function implemented here was based on MARIUS KAMINSKAS and DEREK BRIDGE paper: Diversity, Serendipity, Novelty, and Coverage: A Survey and Empirical Analysis of Beyond-Accuracy Objectives in Recommender Systems
                
                func(i,R) = (relevance[i]*alfa) + (dist_i_R(i,R)*(1-alfa))

            Calculates the diversity score that an item represents to a given list.
            Returns a dict with calculated values.
        '''
        diversity_score = []
        count = 0

        for item in candidates_list:

            aux = {}
            dist_item_R = self.calc_distance_item_in_list(item=item, this_list=actual_list)
            aux['div_score'] = (item['movie_relevance']*alfa) + (dist_item_R*(1-alfa))
            aux['idx'] = count
            diversity_score.append(aux)
            count = count + 1

        return diversity_score


    def diversify_recs_list(self, recs, k=10):
        '''
            This function implemented here was based on MARIUS KAMINSKAS and DEREK BRIDGE paper: Diversity, Serendipity, Novelty, and Coverage: A Survey and Empirical Analysis of Beyond-Accuracy Objectives in Recommender Systems
        
                The Greedy Reranking Algorithm.

            Given a list, returns another list with top-k items diversified based on the Greedy algorithm.
        '''
        diversified_list = []
        
        while len(diversified_list) < k:
            if len(diversified_list) == 0:
                diversified_list.append(recs[0])
                recs.pop(0)
            else:
                diversity_score = self.calc_diversity_score(actual_list=diversified_list, candidates_list=recs)
                diversity_score = sorted(diversity_score, key = lambda i: i['div_score'],reverse=True)
                #  Add the item that maximize diversity in the list 
                item = diversity_score[0]
                diversified_list.append(recs[item['idx']])
                #  Remove this item from the candidates list
                recs.pop(item['idx'])
    
        return diversified_list


    def diversify_recs_list_bounded_random(self, recs, k=10):
        '''
            This function implemented here was based on KEITH BRADLEY and BARRY SMYTH paper: Improving Recommendation Diversity
                
                The Bounded Random Selection Algorithm.

            Returns a list with top-k items diversified based on the Bounded Random algorithm.
        '''
        diversified_list = random.sample(recs,k)

        return diversified_list


    # # # # # # # # # # # # # #
    # # >> EVALUATION MODULE
    # # # # # # # # # # # # # # # # # # # # # #
    # # # >> Intra List Diversity (ILD) module
    # # # # # # # # # # # # # # # # # # # # # #
    def calc_distance_i_j(self, idx_i, idx_j, title_weight=0.8):
        ''' Calculates the distace between item i and item j.
            Returns the distance.
        '''
        sim_genre = self.cosine_sim_movies_genres[idx_i][idx_j]
        sim_title = self.cosine_sim_movies_title[idx_i][idx_j]
        total_sim = (sim_title*title_weight) + (sim_genre*(1-title_weight))
        distance_score = 1 - total_sim

        return distance_score


    def get_distance_matrix(self, final_recs, title_weight=0.8):
        ''' Creates a distace matrix from item in a given list.
            Returns the distance matrix.
        '''
        distance_matrix = []
        for i in final_recs:
            aux = []
            movie_idx_i = int(self.items[self.items['movieId']==i['movie_id']].index[0])
            for j in final_recs:
                movie_idx_j = int(self.items[self.items['movieId']==j['movie_id']].index[0])
                distance_i_j = self.calc_distance_i_j(movie_idx_i, movie_idx_j, title_weight=0.8)
                aux.append(distance_i_j)
            distance_matrix.append(aux)
            
        return distance_matrix


    def get_ILD_score(self, final_recs, title_weight=0.8):
        ''' Returns the ILD score of a given list.
        '''
        distance_matrix = self.get_distance_matrix(final_recs, title_weight=0.8)
        np_dist_mtx = np.array(distance_matrix)
        upper_right = np.triu_indices(np_dist_mtx.shape[0], k=1)

        ild_score = np.mean(np_dist_mtx[upper_right])
        
        return ild_score


    # # # # # # # # # # # # # # # # # # # # # #
    # # # >> Precision module
    # # # # # # # # # # # # # # # # # # # # # #
    def get_mean(self, item):
        ''' Returns the mean of ratings of an item.
        '''
        converted_values = []
        for ratings in item['ratings']:
            for rating in ratings:
                aux = float(rating)
                converted_values.append(aux)

        my_mean = sum(converted_values) / len(converted_values)
        my_mean = round(my_mean, 3)

        return my_mean


    def get_items_means(self, items_list, at):
        ''' Returns the mean of ratings of each item in a list AT certain top.
        '''
        my_copy = self.df_ratings.copy()

        df_items_ratings = my_copy.groupby('movieId')['rating'].apply(list).reset_index(name='ratings')

        items_means = []

        for i in items_list[:at]:
            item = df_items_ratings[df_items_ratings['movieId']==i['movie_id']]
            items_means.append(self.get_mean(item))

        return items_means


    def binary_mean(self, items_mean, cutoff):
        ''' Returns the precision score using binary mean.
        '''
        binary_mean = []
        returned_items = []
        for item in items_mean:
            if item >= cutoff:
                binary_mean.append(1)
            else:
                binary_mean.append(0)

            returned_items.append(1)

        return precision_score(binary_mean, returned_items)


    def precision_at(self, items_list, at):
        ''' Returns the precision score AT certain point of a the list.
        '''
        global_mean = self.trainset.global_mean
        items_list_mean = self.get_items_means(items_list, at)

        print("Global mean: {}, items_list_mean: {}".format(global_mean, items_list_mean))

        precision = self.binary_mean(items_list_mean, global_mean)

        return precision