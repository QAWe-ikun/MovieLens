"""
混合推荐策略模块
实现线性加权融合和多样性优化重排
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class HybridRecommender:
    """混合推荐器"""
    
    def __init__(self, cf_weight: float = 0.5, 
                 diversity_weight: float = 0.0,
                 normalization: str = 'minmax'):
        """
        初始化混合推荐器
        
        Args:
            cf_weight: 协同滤波权重 (0-1)，CB权重为 1-cf_weight
            diversity_weight: 多样性权重 (0-1)，用于重排
            normalization: 分数归一化方法 ('minmax', 'zscore', None)
        """
        self.cf_weight = cf_weight
        self.cb_weight = 1.0 - cf_weight
        self.diversity_weight = diversity_weight
        self.normalization = normalization
        
    def normalize_scores(self, scores: List[float]) -> List[float]:
        """
        归一化分数到 [0, 1] 范围
        
        Args:
            scores: 原始分数列表
            
        Returns:
            normalized_scores: 归一化后的分数
        """
        if not scores:
            return scores
        
        scores_array = np.array(scores)
        
        if self.normalization == 'minmax':
            min_score = np.min(scores_array)
            max_score = np.max(scores_array)
            if max_score - min_score > 0:
                return ((scores_array - min_score) / (max_score - min_score)).tolist()
            else:
                return [0.5] * len(scores)
        
        elif self.normalization == 'zscore':
            mean = np.mean(scores_array)
            std = np.std(scores_array)
            if std > 0:
                normalized = (scores_array - mean) / std
                # 缩放到 [0, 1]
                return ((normalized - normalized.min()) / (normalized.max() - normalized.min())).tolist()
            else:
                return [0.5] * len(scores)
        
        else:
            return scores
    
    def hybrid_score(self, cf_score: float, cb_score: float) -> float:
        """
        计算混合分数
        
        Args:
            cf_score: 协同滤波分数
            cb_score: 基于内容分数
            
        Returns:
            hybrid_score: 混合分数
        """
        return self.cf_weight * cf_score + self.cb_weight * cb_score
    
    def calculate_diversity_penalty(self, item_id: int, 
                                   recommended_items: List[int],
                                   item_similarity_matrix: Optional[np.ndarray] = None) -> float:
        """
        计算多样性惩罚项 - 降低与已推荐物品过于相似的物品分数
        
        Args:
            item_id: 当前物品ID
            recommended_items: 已推荐物品列表
            item_similarity_matrix: 物品相似度矩阵
            
        Returns:
            penalty: 多样性惩罚值 (0-1)
        """
        if not recommended_items or item_similarity_matrix is None:
            return 0.0
        
        # 简化实现：如果没有相似度矩阵，返回默认值
        # 实际应用中应该计算与已推荐物品的平均相似度
        return 0.0
    
    def rerank_with_diversity(self, candidates: List[Tuple[int, float]],
                             item_similarity_matrix: Optional[np.ndarray] = None,
                             top_k: int = 10) -> List[Tuple[int, float]]:
        """
        使用多样性优化重排推荐列表
        
        Args:
            candidates: 候选物品列表 [(item_id, score), ...]
            item_similarity_matrix: 物品相似度矩阵
            top_k: 最终推荐数量
            
        Returns:
            reranked_items: 重排后的推荐列表
        """
        if not candidates:
            return []
        
        if self.diversity_weight == 0:
            return candidates[:top_k]
        
        # MMR (Maximal Marginal Relevance) 算法
        selected = []
        remaining = candidates.copy()
        
        while len(selected) < top_k and remaining:
            best_item = None
            best_score = -float('inf')
            
            for item_id, original_score in remaining:
                # 相关性分数
                relevance = original_score
                
                # 多样性惩罚（与已选物品的相似度）
                diversity_penalty = 0.0
                if selected and item_similarity_matrix is not None:
                    similarities = []
                    for sel_item_id, _ in selected:
                        # 这里简化处理
                        similarities.append(0.5)
                    diversity_penalty = np.mean(similarities)
                
                # MMR 分数
                mmr_score = (1 - self.diversity_weight) * relevance - \
                           self.diversity_weight * diversity_penalty
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_item = (item_id, original_score)
            
            if best_item:
                selected.append(best_item)
                remaining.remove(best_item)
        
        return selected
    
    def recommend(self, user_id: int,
                 cf_recommendations: List[Tuple[int, float]],
                 cb_recommendations: List[Tuple[int, float]],
                 top_k: int = 10,
                 item_similarity_matrix: Optional[np.ndarray] = None) -> List[Tuple[int, float]]:
        """
        生成混合推荐
        
        Args:
            user_id: 用户ID
            cf_recommendations: 协同滤波推荐 [(item_id, score), ...]
            cb_recommendations: 基于内容推荐 [(item_id, score), ...]
            top_k: 推荐数量
            item_similarity_matrix: 物品相似度矩阵（用于多样性重排）
            
        Returns:
            hybrid_recommendations: 混合推荐列表
        """
        if not cf_recommendations and not cb_recommendations:
            return []
        
        # 将推荐列表转换为字典
        cf_dict = dict(cf_recommendations)
        cb_dict = dict(cb_recommendations)
        
        # 获取所有候选物品
        all_items = set(cf_dict.keys()) | set(cb_dict.keys())
        
        # 归一化分数
        if cf_dict:
            cf_items = list(cf_dict.keys())
            cf_scores = list(cf_dict.values())
            cf_normalized = self.normalize_scores(cf_scores)
            cf_dict = dict(zip(cf_items, cf_normalized))
        
        if cb_dict:
            cb_items = list(cb_dict.keys())
            cb_scores = list(cb_dict.values())
            cb_normalized = self.normalize_scores(cb_scores)
            cb_dict = dict(zip(cb_items, cb_normalized))
        
        # 计算混合分数
        candidates = []
        for item_id in all_items:
            cf_score = cf_dict.get(item_id, 0.0)
            cb_score = cb_dict.get(item_id, 0.0)
            
            hybrid_score = self.hybrid_score(cf_score, cb_score)
            candidates.append((item_id, hybrid_score))
        
        # 按混合分数排序
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # 多样性重排
        if self.diversity_weight > 0:
            candidates = self.rerank_with_diversity(
                candidates, item_similarity_matrix, top_k
            )
        
        return candidates[:top_k]
    
    def recommend_all_users(self,
                           cf_recommendations_dict: Dict[int, List[Tuple[int, float]]],
                           cb_recommendations_dict: Dict[int, List[Tuple[int, float]]],
                           top_k: int = 10,
                           item_similarity_matrix: Optional[np.ndarray] = None) -> Dict[int, List[Tuple[int, float]]]:
        """
        为所有用户生成混合推荐

        Args:
            cf_recommendations_dict: {user_id: [(item_id, score), ...]}
            cb_recommendations_dict: {user_id: [(item_id, score), ...]}
            top_k: 推荐数量
            item_similarity_matrix: 物品相似度矩阵

        Returns:
            hybrid_recommendations_dict: {user_id: [(item_id, score), ...]}
        """
        from tqdm import tqdm
        
        hybrid_recommendations_dict = {}

        # 获取所有用户
        all_users = set(cf_recommendations_dict.keys()) | set(cb_recommendations_dict.keys())

        for user_id in tqdm(all_users, desc="  推荐进度 (Hybrid)", total=len(all_users)):
            cf_recs = cf_recommendations_dict.get(user_id, [])
            cb_recs = cb_recommendations_dict.get(user_id, [])

            hybrid_recs = self.recommend(
                user_id, cf_recs, cb_recs, top_k, item_similarity_matrix
            )

            if hybrid_recs:
                hybrid_recommendations_dict[user_id] = hybrid_recs

        return hybrid_recommendations_dict


class DynamicWeightHybridRecommender:
    """动态权重混合推荐器 - 根据用户特征调整权重"""
    
    def __init__(self, base_cf_weight: float = 0.5):
        """
        初始化动态权重混合推荐器
        
        Args:
            base_cf_weight: 基础协同滤波权重
        """
        self.base_cf_weight = base_cf_weight
    
    def calculate_user_weight(self, user_id: int,
                             user_item_matrix: pd.DataFrame) -> float:
        """
        根据用户活跃度计算个性化权重
        
        假设：
        - 活跃用户（评分多）：协同滤波更准确，增加CF权重
        - 冷门用户（评分少）：基于内容更可靠，增加CB权重
        
        Args:
            user_id: 用户ID
            user_item_matrix: 用户-物品评分矩阵
            
        Returns:
            cf_weight: 该用户的CF权重
        """
        if user_id not in user_item_matrix.index:
            return self.base_cf_weight
        
        # 计算用户评分数量
        user_ratings = user_item_matrix.loc[user_id].dropna()
        num_ratings = len(user_ratings)
        
        # 使用sigmoid函数映射权重
        # 评分越多，CF权重越高
        import math
        cf_weight = self.base_cf_weight + 0.3 * (1 - 1 / (1 + num_ratings / 20))
        
        # 限制在 [0.3, 0.8] 范围
        cf_weight = max(0.3, min(0.8, cf_weight))
        
        return cf_weight
    
    def recommend_all_users(self,
                           cf_recommendations_dict: Dict[int, List[Tuple[int, float]]],
                           cb_recommendations_dict: Dict[int, List[Tuple[int, float]]],
                           user_item_matrix: pd.DataFrame,
                           top_k: int = 10) -> Dict[int, List[Tuple[int, float]]]:
        """
        为所有用户生成动态权重混合推荐

        Args:
            cf_recommendations_dict: CF推荐
            cb_recommendations_dict: CB推荐
            user_item_matrix: 用户-物品评分矩阵
            top_k: 推荐数量

        Returns:
            hybrid_recommendations_dict: 混合推荐
        """
        from tqdm import tqdm
        
        hybrid_recommendations_dict = {}

        all_users = set(cf_recommendations_dict.keys()) | set(cb_recommendations_dict.keys())

        for user_id in tqdm(all_users, desc="  推荐进度 (Dynamic Hybrid)", total=len(all_users)):
            cf_recs = cf_recommendations_dict.get(user_id, [])
            cb_recs = cb_recommendations_dict.get(user_id, [])

            # 计算动态权重
            cf_weight = self.calculate_user_weight(user_id, user_item_matrix)

            # 创建临时混合推荐器
            hybrid_rec = HybridRecommender(cf_weight=cf_weight)

            hybrid_recs = hybrid_rec.recommend(user_id, cf_recs, cb_recs, top_k)

            if hybrid_recs:
                hybrid_recommendations_dict[user_id] = hybrid_recs

        return hybrid_recommendations_dict


if __name__ == "__main__":
    # 测试混合推荐
    hybrid = HybridRecommender(cf_weight=0.6, diversity_weight=0.2)
    
    # 模拟推荐结果
    cf_recs = [(101, 4.5), (102, 4.2), (103, 3.8), (104, 3.5)]
    cb_recs = [(102, 4.8), (103, 4.0), (105, 3.9), (106, 3.6)]
    
    print("协同滤波推荐:", cf_recs)
    print("基于内容推荐:", cb_recs)
    
    hybrid_recs = hybrid.recommend(1, cf_recs, cb_recs, top_k=5)
    print("\n混合推荐结果:", hybrid_recs)
