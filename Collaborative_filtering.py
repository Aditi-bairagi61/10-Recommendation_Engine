# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 08:23:01 2024

@author: Lenovo
"""

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

#Load Dataset
file_path = "game.csv"
data = pd.read_csv(file_path)

#Step 1 : Create a user-item matrix (rows:users, columns: games,values:rating)
user_item_matrix = data.pivot_table(index='userId',columns='game',values='rating')
'''
pivot_table : This function reshapes the dataframe into a matrix where:
    Each row represents a user(identified by userId).
    Each column represents a game (identified by game).
    The values in the matrix represents the rating that
    user gave to the games.
'''
#Step 2: Fill NaN values with 0 (assuming no rating means the game has not)
user_item_matrix_filled = user_item_matrix.fillna(0)
'''
This line replace any missing values (NaNs)
in the user-item matrix with 0,
indicating that the user did not rate that particular game
'''

#Step 3: Compute the cosine similarity between user based on raw ratings
user_similarity = cosine_similarity(user_item_matrix_filled)
#Convert similarity matrix to a dataframe for easy reference
user_similarity_df = pd.DataFrame(user_similarity,index=user_item_matrix.index,columns=user_item_matrix.index) 

#Step 4: Function to get game recommendation for a specific user based on
def get_collaborative_recomm_for_user(user_id,num_recommendations=5):
    #Get the similarity scores for the input user with all other users
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)
    
    #Get the most similar users (excluding the user themselves)
    similar_users = similar_users.drop(user_id)
    
    #Select the top N similar users to limit noise (e.g. top 50 users)
    top_similar_users = similar_users.head(50)
    
    #This selections the top 50 most similar users to limit the noise in the recommendations
    #get ratings of these similar users, weighted by their similarity score
    weighted_rating=np.dot(top_similar_users.values,user_item_matrix_filled.loc[top_similar_users.index])
    #np.dot : This computes the dot product between the similarity scores of the top 50 similar users and
    #their corrosponding ratings in the users-item matrix.
    sum_of_similarities = top_similar_users.sum()
    if sum_of_similarities >0:
        weighted_rating /= sum_of_similarities
    #The weighted rating are normalized by dividing by the sum of similarities to avoid biasing towards users with higher ratings.
    
    #Recommend games that the user hasn't rated yet
    user_ratings = user_item_matrix_filled.loc[user_id]
    unrated_games = user_ratings[user_ratings == 0]

    #Get the weighted scores for unrated games
    game_recommendations = pd.Series(weighted_rating,index=user_item_matrix_filled.columns).loc[unrated_games.index]
    return game_recommendations.sort_values(ascending=False).head(num_recommendations)
recommended_games = get_collaborative_recomm_for_user(user_id=3)
print("Recommended System : for user =3 ",recommended_games)
