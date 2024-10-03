# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 09:07:25 2024

@author: HP
"""

#Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#load the csv file
file_path='D:/DS/10-Recommendation_Engine/Entertainment.csv'
data=pd.read_csv(file_path)

#step1: preprocess the 'category' column using TF-IDF
tfidf=TfidfVectorizer(stop_words='english')
tfidf_matrix=tfidf.fit_transform(data['Category'])

#step2:compute the cosine similarity between titles
cosine_sim=cosine_similarity(tfidf_matrix,tfidf_matrix)

#step3: create  a function to recommend titles basedon similarity
def get_recommendations(title,cosine_sim=cosine_sim):
    #get the index of the title that matches the input title
    idx=data[data['Titles']==title].index[0]
    '''
    data['Titles']==title
    this creates a boolean mask(a series of true and false values)
    indicating which rows in the titles column
    match the input  title.
    for example, if the title is "Toy stroy (1995)",
    this comparison results in somethig like:
        0 true
        1 false
        2 false
        
        name:titles,dtype:boot
        why[0]is needed:
            even though the title  should be unique,
   '''
#get the pairwise similarity scores of all titles with that titles
    sim_scores=list(enumerate(cosine_sim[idx]))
#sort the titles based on the similarity scores in descending order
    sim_scores=sorted(sim_scores,key=lambda x:x[1],reverse=True)

     #get the indicates of the most similar titles
    sim_indices=[i[0]for i in sim_scores[1:6]] 
      
     #return the top 5 most similar titles
    return data['Titles'].iloc[sim_indices]   
#test the recommendation system with an example title
example_title="Toy Story (1995)"
recommended_titles=get_recommendations(example_title)

#print the recommendation
print(f"recommendation for '{example_title}':")
for title in recommended_titles:
    
    
      
            