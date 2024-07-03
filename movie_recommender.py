import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import streamlit as st


# Loading the similarity scores from the pickle file
with open('similarity_scores.pkl', 'rb') as file:
    similarity = pickle.load(file)



# Loading the movies_data dataframe from the pickle file
with open('movies_data.pkl', 'rb') as file:
    movies_data = pickle.load(file)


# Title
st.title('Movie Recommendation System')

# User input
movie_name = st.text_input('Enter your favorite movie name:')

if st.button('Recommend'):
    list_of_all_titles = movies_data['title'].tolist()

    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
    if len(find_close_match) == 0:
        st.write('Movie not found in the list. Please try with another movie name.')
    else:
        close_match = find_close_match[0]

        index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]

        similarity_score = list(enumerate(similarity[index_of_the_movie]))

        sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

        st.write('Movies suggested for you:\n')

        i = 1

        for movie in sorted_similar_movies:
            index = movie[0]
            title_from_index = movies_data[movies_data.index == index]['title'].values[0]
            if i < 11:
                st.write(f'{i}. {title_from_index}')
                i += 1