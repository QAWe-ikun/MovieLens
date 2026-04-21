"""
协同过滤推荐模块
实现用户-用户协同过滤和物品-物品协同过滤
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from typing import Dict, List, Optional, Tuple


class CollaborativeFilteringRecommender:
    """协同过滤推荐器"""
    
    def __init__(self, n_components: int = 50, similarity_metric: str = 'cosine'):
        """
        初始化协同过滤推荐器
        
        Args:
            n_components: SVD 降维维度
            similarity_metric: 相似度计算方法 ('cosine', 'pearson')
        """
        self.n_components = n_components
        self.similarity_metric = similarity_metric
        self.user_similarity = None
        self.item_similarity = None
        self.user_item_matrix: Optional[pd.DataFrame] = None
        self.svd = None
        self.user_means = None
        
    def fit(self, user_item_matrix: pd.DataFrame) -> None:
        """
        训练协同过滤模型
        
        Args:
            user_item_matrix: 用户-物品评分矩阵 (用户为行，物品为列)
        """
        self.user_item_matrix = user_item_matrix.copy()
        
        # 计算用户平均评分
        self.user_means = user_item_matrix.mean(axis=1)
        
        # 填充缺失值为用户平均评分
        filled_matrix = user_item_matrix.sub(self.user_means, axis=0).fillna(0)
        
        # SVD 矩阵分解降维
        self.svd = TruncatedSVD(n_components=self.n_components, random_state=42)
        user_factors = self.svd.fit_transform(filled_matrix.values)
        item_factors = self.svd.components_.T
        
        # 计算用户相似度
        self.user_similarity = cosine_similarity(user_factors)
        
        # 计算物品相似度
        self.item_similarity = cosine_similarity(item_factors)
        
        print(f"SVD 解释方差比: {self.svd.explained_variance_ratio_.sum():.4f}")
        print(f"用户相似度矩阵形状: {self.user_similarity.shape}")
        print(f"物品相似度矩阵形状: {self.item_similarity.shape}")
    
    def predict_user_cf(self, user_id: int, item_id: int, k: int = 30) -> float:
        """
        使用用户-用户协同过滤预测评分
        
        Args:
            user_id: 用户ID
            item_id: 物品ID
            k: 最近邻数量
            
        Returns:
            预测评分
        """
        assert self.user_item_matrix is not None, "模型未训练"
        if user_id not in self.user_item_matrix.index:
            return self.user_item_matrix.values.mean()

        assert self.user_means is not None, "模型未训练"
        if item_id not in self.user_item_matrix.columns:
            return self.user_means.get(user_id, self.user_item_matrix.values.mean())

        assert self.user_similarity is not None, "模型未训练"
        # 获取用户相似度
        user_idx = list(self.user_item_matrix.index).index(user_id)
        similarities = self.user_similarity[user_idx]
        
        # 找到对该物品评分过的用户
        item_ratings = self.user_item_matrix[item_id].dropna()
        rated_users = item_ratings.index
        
        # 获取这些用户的相似度
        neighbor_sims = []
        neighbor_ratings = []
        neighbor_user_ids = []
        
        for rated_user in rated_users:
            if rated_user != user_id:
                rated_user_idx = list(self.user_item_matrix.index).index(rated_user)
                sim = similarities[rated_user_idx]
                neighbor_sims.append(sim)
                neighbor_ratings.append(item_ratings[rated_user])
                neighbor_user_ids.append(rated_user)
        
        if len(neighbor_sims) == 0:
            return self.user_means.get(user_id, self.user_item_matrix.values.mean())
        
        # 加权平均预测
        neighbor_sims = np.array(neighbor_sims)
        neighbor_ratings = np.array(neighbor_ratings)
        neighbor_user_ids = np.array(neighbor_user_ids)
        
        # 只取相似度为正的用户
        positive_mask = neighbor_sims > 0
        if positive_mask.sum() > 0:
            neighbor_sims = neighbor_sims[positive_mask]
            neighbor_ratings = neighbor_ratings[positive_mask]
            neighbor_user_ids = neighbor_user_ids[positive_mask]
        
        # 取top-k个最近邻
        if len(neighbor_sims) > k:
            top_k_idx = np.argsort(neighbor_sims)[-k:]
            neighbor_sims = neighbor_sims[top_k_idx]
            neighbor_ratings = neighbor_ratings[top_k_idx]
            neighbor_user_ids = neighbor_user_ids[top_k_idx]
        
        if neighbor_sims.sum() == 0:
            return self.user_means.get(user_id, self.user_item_matrix.values.mean())
        
        # 预测评分 = 用户平均 + 加权偏差
        user_mean = self.user_means[user_id]
        weighted_sum = np.sum(neighbor_sims * (neighbor_ratings - self.user_means[neighbor_user_ids]))
        sim_sum = np.sum(np.abs(neighbor_sims))
        
        predicted_rating = user_mean + weighted_sum / sim_sum if sim_sum > 0 else user_mean
        
        return predicted_rating
    
    def predict_item_cf(self, user_id: int, item_id: int, k: int = 30) -> float:
        """
        使用物品-物品协同过滤预测评分
        
        Args:
            user_id: 用户ID
            item_id: 物品ID
            k: 最近邻数量
            
        Returns:
            预测评分
        """
        assert self.user_item_matrix is not None, "模型未训练"
        if user_id not in self.user_item_matrix.index:
            return self.user_item_matrix.values.mean()

        assert self.user_means is not None, "模型未训练"
        if item_id not in self.user_item_matrix.columns:
            return self.user_means.get(user_id, self.user_item_matrix.values.mean())
        
        # 获取物品相似度
        assert self.item_similarity is not None, "模型未训练"
        item_idx = list(self.user_item_matrix.columns).index(str(item_id))
        similarities = self.item_similarity[item_idx]
        
        # 获取用户评分过的物品
        user_ratings = self.user_item_matrix.loc[user_id].dropna()
        rated_items = user_ratings.index
        
        # 获取这些物品与目标物品的相似度
        neighbor_sims = []
        neighbor_ratings = []
        
        for rated_item in rated_items:
            if rated_item != item_id:
                rated_item_idx = list(self.user_item_matrix.columns).index(rated_item)
                sim = similarities[rated_item_idx]
                neighbor_sims.append(sim)
                neighbor_ratings.append(user_ratings[rated_item])
        
        if len(neighbor_sims) == 0:
            return self.user_means.get(user_id, self.user_item_matrix.values.mean())
        
        neighbor_sims = np.array(neighbor_sims)
        neighbor_ratings = np.array(neighbor_ratings)
        
        # 只取相似度为正的物品
        positive_mask = neighbor_sims > 0
        if positive_mask.sum() > 0:
            neighbor_sims = neighbor_sims[positive_mask]
            neighbor_ratings = neighbor_ratings[positive_mask]
        
        # 取top-k个相似物品
        if len(neighbor_sims) > k:
            top_k_idx = np.argsort(neighbor_sims)[-k:]
            neighbor_sims = neighbor_sims[top_k_idx]
            neighbor_ratings = neighbor_ratings[top_k_idx]
        
        if neighbor_sims.sum() == 0:
            return self.user_means.get(user_id, self.user_item_matrix.values.mean())
        
        # 加权平均预测
        predicted_rating = np.sum(neighbor_sims * neighbor_ratings) / np.sum(neighbor_sims)
        
        return predicted_rating
    
    def recommend(self, user_id: int, top_k: int = 10, 
                  method: str = 'user_cf', 
                  exclude_rated: bool = True) -> List[Tuple[int, float]]:
        """
        为用户生成推荐（使用矩阵化运算加速）
        
        Args:
            user_id: 用户ID
            top_k: 推荐数量
            method: 推荐方法 ('user_cf', 'item_cf')
            exclude_rated: 是否排除已评分物品
            
        Returns:
            recommendations: [(item_id, predicted_score), ...] 按分数降序排列
        """
        assert self.user_item_matrix is not None, "模型未训练"
        if user_id not in self.user_item_matrix.index:
            return []
        
        # 获取用户已评分物品
        rated_items = set()
        if exclude_rated:
            rated_items = set(self.user_item_matrix.loc[user_id].dropna().index)
        
        # 使用矩阵化运算快速预测
        user_idx = list(self.user_item_matrix.index).index(user_id)
        
        if method == 'user_cf':
            # User-CF: 使用用户相似度矩阵
            # 找到所有对该物品评分过的用户的加权和
            assert self.user_similarity is not None, "模型未训练"
            user_similarities = self.user_similarity[user_idx]
            
            # 获取用户-物品矩阵（填充0）
            filled_matrix = self.user_item_matrix.fillna(0).values
            
            # 计算预测: 对于每个物品，使用相似用户的评分加权
            # pred = sum(sim * rating) / sum(|sim|)
            numerator = user_similarities @ filled_matrix  # (n_items,)
            denominator = np.abs(user_similarities).sum()
            
            if denominator > 0:
                predictions = numerator / denominator
            else:
                assert self.user_means is not None, "模型未训练"
                predictions = np.full(filled_matrix.shape[1], self.user_means[user_id])
            
        elif method == 'item_cf':
            # Item-CF: 使用物品相似度矩阵
            user_ratings = self.user_item_matrix.loc[user_id].fillna(0).values

            # 计算预测: 对于每个物品，使用相似物品的评分加权
            item_similarities = self.item_similarity

            assert item_similarities is not None, "模型未训练"
            numerator = item_similarities @ user_ratings  # (n_items,)
            denominator = item_similarities.sum(axis=1)

            # 避免除以0或负数
            assert denominator is not None, "模型未训练"
            safe_denominator = np.where(denominator > 0, denominator, 1.0)
            predictions = numerator / safe_denominator
            assert self.user_means is not None, "模型未训练"
            predictions = np.where(denominator > 0, predictions, self.user_means[user_id])
        else:
            raise ValueError(f"未知方法: {method}")
        
        # 排除已评分物品
        all_items = self.user_item_matrix.columns
        predictions_list = []
        
        for idx, item_id in enumerate(all_items):
            if item_id not in rated_items:
                predictions_list.append((item_id, predictions[idx]))
        
        # 按预测分数排序
        predictions_list.sort(key=lambda x: x[1], reverse=True)
        
        return predictions_list[:top_k]
    
    def recommend_all_users(self, top_k: int = 10, 
                           method: str = 'user_cf',
                           max_users: Optional[int] = None) -> Dict[int, List[Tuple[int, float]]]:
        """
        为所有用户生成推荐
        
        Args:
            top_k: 推荐数量
            method: 推荐方法
            max_users: 最大用户数（用于加速测试）
            
        Returns:
            all_recommendations: {user_id: [(item_id, score), ...]}
        """
        from tqdm import tqdm

        assert self.user_item_matrix is not None, "模型未训练"
        all_recommendations = {}
        users = self.user_item_matrix.index
        
        if max_users is not None:
            users = users[:max_users]
        
        total_users = len(users)
        for user_id in tqdm(users, desc=f"  推荐进度 ({method})", total=total_users):
            recs = self.recommend(user_id, top_k, method)
            if recs:
                all_recommendations[user_id] = recs
        
        return all_recommendations


if __name__ == "__main__":
    # 测试协同过滤
    from data_loader import MovieLensDataLoader
    
    loader = MovieLensDataLoader()
    ratings, movies, tags = loader.load_data()
    
    # 构建用户-物品矩阵
    user_item_matrix = loader.build_user_item_matrix(ratings)
    
    # 训练模型
    cf = CollaborativeFilteringRecommender(n_components=50)
    cf.fit(user_item_matrix)
    
    # 为用户生成推荐
    user_id = 1
    print(f"\n为用户 {user_id} 推荐 (User-CF):")
    recs = cf.recommend(user_id, top_k=10, method='user_cf')
    for item_id, score in recs[:5]:
        movie_title = movies[movies['movieId'] == item_id]['title'].values
        if len(movie_title) > 0:
            print(f"  - {movie_title[0]} (预测评分: {score:.2f})")
