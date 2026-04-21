"""
基于内容推荐模块测试
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

# 添加 src 目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from content_based import ContentBasedRecommender


@pytest.fixture
def sample_movies():
    """创建样例电影数据"""
    return pd.DataFrame({
        'movieId': [101, 102, 103, 104, 105],
        'title': ['Movie A', 'Movie B', 'Movie C', 'Movie D', 'Movie E'],
        'genres': [
            'Action|Adventure',
            'Comedy|Romance',
            'Action|Sci-Fi',
            'Drama|Romance',
            'Comedy|Adventure'
        ]
    })


@pytest.fixture
def cb_recommender(sample_movies):
    """创建基于内容推荐器实例"""
    cb = ContentBasedRecommender()
    cb.fit(sample_movies)
    return cb


@pytest.fixture
def sample_user_ratings():
    """创建样例用户评分"""
    return pd.DataFrame({
        'movieId': [101, 102, 103],
        'rating': [5.0, 4.0, 3.0]
    })


class TestContentBased:
    """基于内容推荐测试类"""
    
    def test_fit(self, sample_movies):
        """测试模型训练"""
        cb = ContentBasedRecommender()
        cb.fit(sample_movies)
        
        assert cb.tfidf_matrix is not None
        assert cb.tfidf_matrix.shape[0] == 5  # 5部电影
        assert len(cb.movie_id_to_idx) == 5
    
    def test_get_similar_movies(self, cb_recommender):
        """测试获取相似电影"""
        similar = cb_recommender.get_similar_movies(101, top_k=3)
        assert isinstance(similar, list)
        assert len(similar) <= 3
        assert all(isinstance(item, tuple) and len(item) == 2 for item in similar)
        
        # 相似度分数应该在 0-1 之间
        for movie_id, score in similar:
            assert 0.0 <= score <= 1.0
    
    def test_build_user_profile(self, cb_recommender, sample_user_ratings):
        """测试构建用户画像"""
        user_profile = cb_recommender.build_user_profile(sample_user_ratings)
        assert isinstance(user_profile, np.ndarray)
        assert len(user_profile) > 0
    
    def test_predict(self, cb_recommender):
        """测试预测评分"""
        user_item_matrix = pd.DataFrame({
            101: [5.0],
            102: [4.0],
            103: [np.nan],
            104: [np.nan],
            105: [np.nan]
        }, index=[1])
        
        prediction = cb_recommender.predict(1, 103, user_item_matrix)
        assert isinstance(prediction, float)
        assert 1.0 <= prediction <= 5.0
    
    def test_recommend(self, cb_recommender):
        """测试推荐功能"""
        user_item_matrix = pd.DataFrame({
            101: [5.0],
            102: [4.0],
            103: [3.0],
            104: [np.nan],
            105: [np.nan]
        }, index=[1])
        
        recs = cb_recommender.recommend(1, top_k=3, user_item_matrix=user_item_matrix)
        assert isinstance(recs, list)
        assert len(recs) <= 3
    
    def test_get_movie_features(self, cb_recommender):
        """测试获取电影特征"""
        features = cb_recommender.get_movie_features(101)
        assert isinstance(features, np.ndarray)
        assert len(features) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
