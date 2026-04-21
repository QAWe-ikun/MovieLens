"""
基于内容的推荐模块
利用电影标签(genres)构建TF-IDF特征，计算相似度并生成推荐
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple


class ContentBasedRecommender:
    """基于内容的推荐器"""
    
    def __init__(self, tfidf_max_features: int = 100):
        """
        初始化基于内容的推荐器
        
        Args:
            tfidf_max_features: TF-IDF 特征最大数量
        """
        self.tfidf_max_features = tfidf_max_features
        self.tfidf_matrix = None
        self.movies_df = None
        self.tfidf_vectorizer = None
        self.movie_id_to_idx = None
        self.idx_to_movie_id = None
        
    def fit(self, movies_df: pd.DataFrame) -> None:
        """
        训练基于内容的推荐模型
        
        Args:
            movies_df: 电影数据 (包含 movieId, title, genres)
        """
        self.movies_df = movies_df.copy()
        
        # 创建 movieId 到索引的映射
        self.movie_id_to_idx = {mid: idx for idx, mid in enumerate(movies_df['movieId'].values)}
        self.idx_to_movie_id = {idx: mid for mid, idx in self.movie_id_to_idx.items()}
        
        # 预处理 genres 列
        # 将 genres 从 "Action|Comedy|Drama" 转换为 "Action Comedy Drama"
        genres_processed = movies_df['genres'].apply(
            lambda x: x.replace('|', ' ') if isinstance(x, str) else ''
        )
        
        # 构建 TF-IDF 特征矩阵
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.tfidf_max_features,
            token_pattern=r'[A-Za-z]+'
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(genres_processed)
        
        print(f"TF-IDF 矩阵形状: {self.tfidf_matrix.shape}")
        print(f"特征数量: {self.tfidf_matrix.shape[1]}")
    
    def get_similar_movies(self, movie_id: int, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        获取与指定电影相似的电影
        
        Args:
            movie_id: 目标电影ID
            top_k: 相似电影数量
            
        Returns:
            similar_movies: [(movie_id, similarity_score), ...]
        """
        if movie_id not in self.movie_id_to_idx: # type: ignore
            return []
        
        movie_idx = self.movie_id_to_idx[movie_id] # type: ignore
        
        # 计算该电影与所有电影的相似度
        similarities = cosine_similarity(
            self.tfidf_matrix[movie_idx:movie_idx+1], # type: ignore
            self.tfidf_matrix
        ).flatten()
        
        # 获取top-k相似电影（排除自身）
        similar_indices = np.argsort(similarities)[::-1]
        similar_movies = []
        
        for idx in similar_indices:
            if idx != movie_idx and len(similar_movies) < top_k:
                similar_movie_id = self.idx_to_movie_id[idx] # type: ignore
                similar_movies.append((similar_movie_id, similarities[idx]))
        
        return similar_movies
    
    def build_user_profile(self, user_ratings: pd.DataFrame) -> np.ndarray:
        """
        构建用户兴趣画像（基于评分加权的TF-IDF向量）
        
        Args:
            user_ratings: 用户评分数据 (包含 movieId, rating)
            
        Returns:
            user_profile: 用户兴趣向量
        """
        user_profile = np.zeros(self.tfidf_matrix.shape[1]) # type: ignore
        total_weight = 0
        
        for _, row in user_ratings.iterrows():
            movie_id = row['movieId']
            rating = row['rating']
            
            if movie_id in self.movie_id_to_idx:
                movie_idx = self.movie_id_to_idx[movie_id] # type: ignore
                movie_vector = self.tfidf_matrix[movie_idx].toarray().flatten() # type: ignore
                
                # 使用评分作为权重
                user_profile += rating * movie_vector
                total_weight += rating
        
        # 归一化
        if total_weight > 0:
            user_profile /= total_weight
        
        return user_profile
    
    def predict(self, user_id: int, item_id: int, 
                user_item_matrix: pd.DataFrame) -> float:
        """
        预测用户对物品的评分
        
        Args:
            user_id: 用户ID
            item_id: 物品ID
            user_item_matrix: 用户-物品评分矩阵
            
        Returns:
            predicted_rating: 预测评分
        """
        if item_id not in self.movie_id_to_idx: # type: ignore
            return user_item_matrix.values.mean()
        
        # 获取用户历史评分
        if user_id in user_item_matrix.index:
            user_ratings = user_item_matrix.loc[user_id].dropna()
            
            if len(user_ratings) == 0:
                return user_item_matrix.values.mean()
            
            # 构建用户画像
            user_ratings_df = pd.DataFrame({
                'movieId': user_ratings.index,
                'rating': user_ratings.values
            })
            
            user_profile = self.build_user_profile(user_ratings_df)
            
            # 计算目标电影与用户画像的相似度
            item_idx = self.movie_id_to_idx[item_id] # type: ignore
            item_vector = self.tfidf_matrix[item_idx] # type: ignore
            
            similarity = cosine_similarity(
                item_vector.reshape(1, -1),
                user_profile.reshape(1, -1)
            )[0][0]
            
            # 将相似度映射到评分范围 (1-5)
            predicted_rating = 1 + 4 * similarity
            
            return predicted_rating
        
        return user_item_matrix.values.mean()
    
    def recommend(self, user_id: int, user_item_matrix: pd.DataFrame,
                  top_k: int = 10, exclude_rated: bool = True) -> List[Tuple[int, float]]:
        """
        为用户生成基于内容的推荐（使用矩阵化运算加速）

        Args:
            user_id: 用户ID
            user_item_matrix: 用户-物品评分矩阵
            top_k: 推荐数量
            exclude_rated: 是否排除已评分物品

        Returns:
            recommendations: [(item_id, predicted_score), ...]
        """
        if user_id not in user_item_matrix.index:
            return []

        # 获取用户历史评分
        user_ratings = user_item_matrix.loc[user_id].dropna()
        if len(user_ratings) == 0:
            return []

        # 构建用户画像（只构建一次）
        user_ratings_df = pd.DataFrame({
            'movieId': user_ratings.index,
            'rating': user_ratings.values
        })
        user_profile = self.build_user_profile(user_ratings_df)

        # 使用矩阵运算一次性计算所有物品与用户画像的相似度
        similarities = cosine_similarity(
            self.tfidf_matrix, # type: ignore
            user_profile.reshape(1, -1)
        ).flatten()

        # 将相似度映射到评分范围 (1-5)
        predictions = 1 + 4 * similarities

        # 获取用户已评分物品
        rated_items = set()
        if exclude_rated:
            rated_items = set(user_item_matrix.loc[user_id].dropna().index)

        # 构建 (item_id, score) 列表，排除已评分物品
        all_items = user_item_matrix.columns
        predictions_list = []

        assert self.movie_id_to_idx, "TF-IDF矩阵和用户画像维度不匹配"
        for idx, item_id in enumerate(all_items):
            if item_id not in rated_items and item_id in self.movie_id_to_idx:
                predictions_list.append((item_id, predictions[idx]))

        # 按预测分数排序
        predictions_list.sort(key=lambda x: x[1], reverse=True)

        return predictions_list[:top_k]
    
    def recommend_all_users(self, user_item_matrix: pd.DataFrame,
                           top_k: int = 10) -> Dict[int, List[Tuple[int, float]]]:
        """
        为所有用户生成推荐

        Args:
            user_item_matrix: 用户-物品评分矩阵
            top_k: 推荐数量
            
        Returns:
            all_recommendations: {user_id: [(item_id, score), ...]}
        """
        from tqdm import tqdm
        
        all_recommendations = {}
        
        for user_id in tqdm(user_item_matrix.index, desc="  推荐进度 (Content-Based)", total=len(user_item_matrix)):
            recs = self.recommend(user_id, user_item_matrix, top_k)
            if recs:
                all_recommendations[user_id] = recs
        
        return all_recommendations
    
    def get_movie_features(self, movie_id: int) -> np.ndarray:
        """
        获取指定电影的TF-IDF特征向量
        
        Args:
            movie_id: 电影ID
            
        Returns:
            feature_vector: TF-IDF特征向量
        """
        if movie_id not in self.movie_id_to_idx: # type: ignore
            return np.zeros(self.tfidf_matrix.shape[1]) # type: ignore
        
        movie_idx = self.movie_id_to_idx[movie_id] # type: ignore
        return self.tfidf_matrix[movie_idx].toarray().flatten() # type: ignore


if __name__ == "__main__":
    # 测试基于内容推荐
    from data_loader import MovieLensDataLoader
    
    loader = MovieLensDataLoader()
    ratings, movies, tags = loader.load_data()
    
    # 训练模型
    cb = ContentBasedRecommender()
    cb.fit(movies)
    
    # 获取相似电影
    movie_id = 1
    print(f"\n与电影 {movie_id} 相似的电影:")
    similar = cb.get_similar_movies(movie_id, top_k=5)
    for sim_movie_id, score in similar:
        title = movies[movies['movieId'] == sim_movie_id]['title'].values
        if len(title) > 0:
            print(f"  - {title[0]} (相似度: {score:.3f})")
    
    # 构建用户-物品矩阵
    user_item_matrix = loader.build_user_item_matrix(ratings)
    
    # 为用户生成推荐
    user_id = 1
    print(f"\n为用户 {user_id} 推荐 (Content-Based):")
    recs = cb.recommend(user_id, top_k=10, user_item_matrix=user_item_matrix)
    for item_id, score in recs[:5]:
        title = movies[movies['movieId'] == item_id]['title'].values
        if len(title) > 0:
            print(f"  - {title[0]} (预测评分: {score:.2f})")
