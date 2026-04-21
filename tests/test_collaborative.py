"""
协同过滤推荐模块测试
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

# 添加 src 目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from collaborative_filtering import CollaborativeFilteringRecommender


@pytest.fixture
def sample_user_item_matrix():
    """创建样例用户-物品矩阵"""
    data = {
        101: [5.0, 3.0, np.nan, np.nan],
        102: [4.0, np.nan, 2.0, np.nan],
        103: [np.nan, 4.0, 5.0, 3.0],
        104: [2.0, np.nan, 4.0, 5.0],
        105: [np.nan, 5.0, np.nan, 4.0]
    }
    return pd.DataFrame(data, index=[1, 2, 3, 4])


@pytest.fixture
def cf_recommender(sample_user_item_matrix):
    """创建协同过滤推荐器实例"""
    cf = CollaborativeFilteringRecommender(n_components=2)
    cf.fit(sample_user_item_matrix)
    return cf


class TestCollaborativeFiltering:
    """协同过滤测试类"""
    
    def test_fit(self, sample_user_item_matrix):
        """测试模型训练"""
        cf = CollaborativeFilteringRecommender(n_components=2)
        cf.fit(sample_user_item_matrix)
        
        assert cf.user_similarity is not None
        assert cf.item_similarity is not None
        assert cf.user_similarity.shape[0] == 4  # 4个用户
        assert cf.item_similarity.shape[0] == 5  # 5个物品
    
    def test_predict_user_cf(self, cf_recommender):
        """测试用户CF预测"""
        prediction = cf_recommender.predict_user_cf(1, 103)
        assert isinstance(prediction, float)
        assert 1.0 <= prediction <= 5.0
    
    def test_predict_item_cf(self, cf_recommender):
        """测试物品CF预测"""
        prediction = cf_recommender.predict_item_cf(1, 103)
        assert isinstance(prediction, float)
        assert 1.0 <= prediction <= 5.0
    
    def test_recommend(self, cf_recommender):
        """测试推荐功能"""
        recs = cf_recommender.recommend(1, top_k=5, method='user_cf')
        assert isinstance(recs, list)
        assert len(recs) <= 5
        assert all(isinstance(item, tuple) and len(item) == 2 for item in recs)
    
    def test_recommend_all_users(self, cf_recommender):
        """测试为所有用户推荐"""
        recs_dict = cf_recommender.recommend_all_users(top_k=3, method='user_cf')
        assert isinstance(recs_dict, dict)
        assert len(recs_dict) > 0
    
    def test_exclude_rated(self, cf_recommender, sample_user_item_matrix):
        """测试排除已评分物品"""
        recs = cf_recommender.recommend(1, top_k=10, method='user_cf', exclude_rated=True)
        rated_items = set(sample_user_item_matrix.loc[1].dropna().index)
        recommended_items = set(item_id for item_id, _ in recs)
        
        # 推荐列表不应包含已评分物品
        assert len(rated_items & recommended_items) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
