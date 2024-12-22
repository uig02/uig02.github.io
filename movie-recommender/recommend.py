import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import numpy as np

# 加载电影和评分数据
movies = pd.read_csv('/Users/liuzizhuang/Desktop/movie-recommender/ml-latest-small/movies.csv')
ratings = pd.read_csv('/Users/liuzizhuang/Desktop/movie-recommender/ml-latest-small/ratings.csv')


def get_movies_by_genre_sorted_by_rating(genre, num_movies):
    # 筛选包含该类别的电影
    genre_movies = movies[movies['genres'].str.contains(genre, case=False, na=False)]
    # 根据评分对电影进行排序
    movie_ratings = ratings[ratings['movieId'].isin(genre_movies['movieId'])]
    avg_ratings = movie_ratings.groupby('movieId')['rating'].mean()
    genre_movies['rating'] = genre_movies['movieId'].map(avg_ratings)
    genre_movies = genre_movies.dropna(subset=['rating'])  # 移除没有评分的电影
    genre_movies_sorted = genre_movies.sort_values(by='rating', ascending=False)
    return genre_movies_sorted[['title', 'genres', 'rating']].head(num_movies).to_dict('records')

# 2. 根据用户偏好进行推荐
def genre_based_recommendations(user_id, num_recommendations):
    # 获取用户的评分数据
    user_ratings = ratings[ratings['userId'] == user_id]
    # 获取用户评分过的电影ID
    rated_movie_ids = user_ratings['movieId'].unique()
    # 排除用户已经评分的电影
    unrated_movies = movies[~movies['movieId'].isin(rated_movie_ids)]

    # 按电影类别推荐用户可能喜欢的电影
    genre_counts = user_ratings.groupby('movieId')['rating'].mean()
    top_genres = unrated_movies['genres'].str.split('|', expand=True).stack().value_counts().index[:5]

    # 推荐与用户评分过的电影类型相似的电影
    recommendations = []
    for genre in top_genres:
        genre_movies = get_movies_by_genre_sorted_by_rating(genre)
        recommendations.extend(genre_movies)

    return recommendations[:num_recommendations]

# 3. 根据用户观看历史进行推荐
def collaborative_recommendations(user_id, num_recommendations):
    # 计算与该用户评分相似的其他用户
    user_ratings = ratings[ratings['userId'] == user_id]
    user_movie_ids = user_ratings['movieId'].unique()

    # 获取所有用户的评分数据
    all_ratings = ratings.pivot_table(index='userId', columns='movieId', values='rating')

    # 计算用户之间的相似度
    from sklearn.metrics.pairwise import cosine_similarity
    similarity_matrix = cosine_similarity(all_ratings.fillna(0))

    # 获取与当前用户最相似的用户
    user_index = all_ratings.index.get_loc(user_id)
    similar_users = similarity_matrix[user_index]
    similar_users_indices = similar_users.argsort()[::-1][1:]

    # 从最相似的用户中找到他们评分过的未看过的电影
    recommendations = []
    for idx in similar_users_indices:
        similar_user_id = all_ratings.index[idx]
        similar_user_ratings = ratings[ratings['userId'] == similar_user_id]
        recommended_movies = similar_user_ratings[~similar_user_ratings['movieId'].isin(user_movie_ids)]
        recommended_movies = recommended_movies.sort_values(by='rating', ascending=False)
        recommendations.extend(recommended_movies[['movieId', 'rating']].head(3).to_dict('records'))

    recommended_movie_ids = [r['movieId'] for r in recommendations]
    return movies[movies['movieId'].isin(recommended_movie_ids)].to_dict('records')[:num_recommendations]

# 4. 综合推荐：结合用户偏好和观看历史进行推荐
def hybrid_recommendations(user_id, num_recommendations):
    genre_recs = genre_based_recommendations(user_id, num_recommendations)
    collaborative_recs = collaborative_recommendations(user_id, num_recommendations)

    # 结合两种推荐结果，去重后返回
    hybrid_recs = {rec['title']: rec for rec in genre_recs + collaborative_recs}
    return list(hybrid_recs.values())[:num_recommendations]


def recommend_movies_based_on_mood(mood, num_movies):
    mood_genres = {
        'happy': 'Comedy|Romance',
        'sad': 'Drama|Romance',
        'tense': 'Thriller|Action',
        'relaxed': 'Documentary|Comedy'
    }

    genre = mood_genres.get(mood, 'Comedy')
    return get_movies_by_genre_sorted_by_rating(genre, num_movies)