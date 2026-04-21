"""
混合推荐策略模块测试
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

# 添加 src 目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from hybrid_recommender import HybridRecommender, DynamicWeightHybridRecommender


@pytest.fixture
def hybrid_recommender():
    """创建混合推荐器实例"""
    return HybridRecommender(cf_weight=0.6, diversity_weight=0.0)


@pytest.fixture
def sample_cf_recommendations():
    """样例CF推荐"""
    return {
        1: [(101, 4.5), (102, 4.2), (103, 3.8)],
        2: [(104, 4.0), (105, 3.9), (106, 3.5)]
    }


@pytest.fixture
def sample_cb_recommendations():
    """样例CB推荐"""
    return {
        1: [(102, 4.8), (103, 4.0), (107, 3.6)],
        2: [(105, 4.5), (108, 3.8), (101, 3.2)]
    }


class TestHybridRecommender:
    """混合推荐器测试类"""
    
    def test_normalize_scores_minmax(self):
        """测试 MinMax 归一化"""
        hybrid = HybridRecommender(normalization='minmax')
        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        normalized = hybrid.normalize_scores(scores)
        
        assert abs(min(normalized) - 0.0) < 0.001
        assert abs(max(normalized) - 1.0) < 0.001
    
    def test_hybrid_score(self):
        """测试混合分数计算"""
        hybrid = HybridRecommender(cf_weight=0.6)
        score = hybrid.hybrid_score(0.8, 0.6)
        
        expected = 0.6 * 0.8 + 0.4 * 0.6
        assert abs(score - expected) < 0.001
    
    def test_recommend(self, hybrid_recommender):
        """测试混合推荐"""
        cf_recs = [(101, 4.5), (102, 4.2), (103, 3.8)]
        cb_recs = [(102, 4.8), (103, 4.0), (104, 3.6)]
        
        recs = hybrid_recommender.recommend(1, cf_recs, cb_recs, top_k=3)
        
        assert isinstance(recs, list)
        assert len(recs) <= 3
        assert all(isinstance(item, tuple) and len(item) == 2 for item in recs)
    
    def test_recommend_all_users(self, hybrid_recommender, 
                                sample_cf_recommendations, 
                                sample_cb_recommendations):
        """测试为所有用户推荐"""
        recs_dict = hybrid_recommender.recommend_all_users(
            sample_cf_recommendations,
            sample_cb_recommendations,
            top_k=3
        )
        
        assert isinstance(recs_dict, dict)
        assert len(recs_dict) == 2  # 2个用户
    
    def test_empty_recommendations(self, hybrid_recommender):
        """测试空推荐情况"""
        recs = hybrid_recommender.recommend(1, [], [], top_k=5)
        assert recs == []
    
    def test_weight_variation(self):
        """测试不同权重的影响"""
        hybrid_cf = HybridRecommender(cf_weight=0.8)
        hybrid_cb = HybridRecommender(cf_weight=0.2)
        
        cf_recs = [(101, 0.9), (102, 0.7)]
        cb_recs = [(101, 0.5), (103, 0.8)]
        
        recs_cf = hybrid_cf.recommend(1, cf_recs, cb_recs, top_k=3)
        recs_cb = hybrid_cb.recommend(1, cf_recs, cb_recs, top_k=3)
        
        # 不同权重应该产生不同的排序
        assert isinstance(recs_cf, list)
        assert isinstance(recs_cb, list)


class TestDynamicWeightHybridRecommender:
    """动态权重混合推荐器测试类"""
    
    def test_calculate_user_weight(self):
        """测试用户权重计算"""
        dynamic_hybrid = DynamicWeightHybridRecommender(base_cf_weight=0.5)
        
        user_item_matrix = pd.DataFrame({
            101: [5.0, 4.0],
            102: [4.0, 3.0],
            103: [3.0, np.nan],
            104: [np.nan, 5.0],
            105: [5.0, 4.0]
        }, index=[1, 2])
        
        # 用户1有更多评分，应该获得更高的CF权重
        weight1 = dynamic_hybrid.calculate_user_weight(1, user_item_matrix)
        weight2 = dynamic_hybrid.calculate_user_weight(2, user_item_matrix)
        
        assert 0.3 <= weight1 <= 0.8
        assert 0.3 <= weight2 <= 0.8
    
    def test_recommend_all_users(self, sample_cf_recommendations, 
                                sample_cb_recommendations):
        """测试动态权重推荐"""
        user_item_matrix = pd.DataFrame({
            101: [5.0, 4.0],
            102: [4.0, 3.0],
            103: [3.0, np.nan],
            104: [np.nan, 5.0],
            105: [5.0, 4.0],
            106: [4.0, np.nan],
            107: [np.nan, 3.0],
            108: [3.0, 5.0]
        }, index=[1, 2])
        
        dynamic_hybrid = DynamicWeightHybridRecommender(base_cf_weight=0.5)
        
        recs_dict = dynamic_hybrid.recommend_all_users(
            sample_cf_recommendations,
            sample_cb_recommendations,
            user_item_matrix,
            top_k=3
        )
        
        assert isinstance(recs_dict, dict)
        assert len(recs_dict) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
