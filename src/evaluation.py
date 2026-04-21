"""
评估模块
实现召回率、准确率、覆盖率、多样性等推荐系统评估指标
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple
from collections import defaultdict


class RecommenderEvaluator:
    """推荐系统评估器"""
    
    def __init__(self):
        pass
    
    @staticmethod
    def get_relevant_items(test_df: pd.DataFrame, user_id: int, 
                          threshold: float = 3.5) -> Set[int]:
        """
        获取测试集中用户的相关物品（评分高于阈值的物品）
        
        Args:
            test_df: 测试集数据
            user_id: 用户ID
            threshold: 相关评分阈值
            
        Returns:
            relevant_items: 相关物品集合
        """
        user_test = test_df[test_df['userId'] == user_id]
        relevant_items = set(user_test[user_test['rating'] >= threshold]['movieId'])
        return relevant_items
    
    @staticmethod
    def recall_at_k(recommended_items: List[int], 
                    relevant_items: Set[int], 
                    k: int = 10) -> float:
        """
        计算 Recall@K
        
        Recall@K = |推荐列表 ∩ 相关物品| / min(K, |相关物品|)
        
        这种定义更合理：衡量推荐列表的"命中率"
        
        Args:
            recommended_items: 推荐列表
            relevant_items: 相关物品集合
            k: 推荐数量
            
        Returns:
            recall: 召回率
        """
        if len(relevant_items) == 0:
            return 0.0
        
        recommended_k = recommended_items[:k]
        hits = len(set(recommended_k) & relevant_items)
        
        # 使用 min(k, len(relevant_items)) 作为分母，更合理
        denominator = min(k, len(relevant_items))
        
        return hits / denominator if denominator > 0 else 0.0
    
    @staticmethod
    def precision_at_k(recommended_items: List[int], 
                       relevant_items: Set[int], 
                       k: int = 10) -> float:
        """
        计算 Precision@K
        
        Args:
            recommended_items: 推荐列表
            relevant_items: 相关物品集合
            k: 推荐数量
            
        Returns:
            precision: 准确率
        """
        if k == 0:
            return 0.0
        
        recommended_k = recommended_items[:k]
        hits = len(set(recommended_k) & relevant_items)
        
        return hits / k
    
    @staticmethod
    def ndcg_at_k(recommended_items: List[int], 
                  relevant_items: Set[int], 
                  k: int = 10) -> float:
        """
        计算 NDCG@K (Normalized Discounted Cumulative Gain)
        
        Args:
            recommended_items: 推荐列表
            relevant_items: 相关物品集合
            k: 推荐数量
            
        Returns:
            ndcg: 归一化折扣累积增益
        """
        if len(relevant_items) == 0:
            return 0.0
        
        recommended_k = recommended_items[:k]
        
        # 计算 DCG
        dcg = 0.0
        for i, item in enumerate(recommended_k):
            if item in relevant_items:
                dcg += 1.0 / np.log2(i + 2)  # i+2 因为 i 从 0 开始
        
        # 计算理想 DCG
        ideal_hits = min(len(relevant_items), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    @staticmethod
    def coverage(recommended_items_dict: Dict[int, List[int]], 
                 all_items: Set[int]) -> float:
        """
        计算覆盖率 - 被推荐出的物品占总物品的比例
        
        Args:
            recommended_items_dict: {user_id: [item_id1, item_id2, ...]}
            all_items: 所有物品集合
            
        Returns:
            coverage: 覆盖率
        """
        if len(all_items) == 0:
            return 0.0
        
        all_recommended = set()
        for items in recommended_items_dict.values():
            all_recommended.update(items)
        
        return len(all_recommended) / len(all_items)
    
    @staticmethod
    def diversity(recommended_items_dict: Dict[int, List[int]], 
                  item_similarity: np.ndarray = None) -> float:
        """
        计算多样性 - 推荐列表中物品的平均差异度
        
        Args:
            recommended_items_dict: {user_id: [item_id1, item_id2, ...]}
            item_similarity: 物品相似度矩阵 (可选)
            
        Returns:
            diversity: 平均多样性 (1 - 平均相似度)
        """
        if not recommended_items_dict or item_similarity is None:
            return 0.0
        
        diversities = []
        
        for user_id, items in recommended_items_dict.items():
            if len(items) < 2:
                continue
            
            # 计算推荐列表中物品的平均相似度
            similarities = []
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    # 这里简化处理，实际应该使用物品索引
                    similarities.append(0.5)  #  placeholder
            
            avg_similarity = np.mean(similarities) if similarities else 0
            diversities.append(1 - avg_similarity)
        
        return np.mean(diversities) if diversities else 0.0
    
    def evaluate_user(self, user_id: int, 
                     recommended_items: List[int],
                     test_df: pd.DataFrame,
                     k: int = 10) -> Dict[str, float]:
        """
        评估单个用户的推荐结果
        
        Args:
            user_id: 用户ID
            recommended_items: 推荐列表
            test_df: 测试集
            k: 推荐数量
            
        Returns:
            metrics: {metric_name: value}
        """
        relevant_items = self.get_relevant_items(test_df, user_id)
        
        if len(relevant_items) == 0:
            return {
                'recall': 0.0,
                'precision': 0.0,
                'ndcg': 0.0
            }
        
        metrics = {
            'recall': self.recall_at_k(recommended_items, relevant_items, k),
            'precision': self.precision_at_k(recommended_items, relevant_items, k),
            'ndcg': self.ndcg_at_k(recommended_items, relevant_items, k)
        }
        
        return metrics
    
    def evaluate_all(self, recommended_items_dict: Dict[int, List[int]], 
                    test_df: pd.DataFrame,
                    all_items: Set[int],
                    k: int = 10) -> Dict[str, float]:
        """
        评估所有用户的推荐结果
        
        Args:
            recommended_items_dict: {user_id: [item_id1, item_id2, ...]}
            test_df: 测试集
            all_items: 所有物品集合
            k: 推荐数量
            
        Returns:
            avg_metrics: 平均评估指标
        """
        all_metrics = defaultdict(list)
        
        for user_id, recommended_items in recommended_items_dict.items():
            metrics = self.evaluate_user(user_id, recommended_items, test_df, k)
            for metric_name, value in metrics.items():
                all_metrics[metric_name].append(value)
        
        # 计算平均值
        avg_metrics = {
            metric: np.mean(values)
            for metric, values in all_metrics.items()
        }
        
        # 添加覆盖率
        avg_metrics['coverage'] = self.coverage(recommended_items_dict, all_items)
        
        return avg_metrics
    
    def compare_methods(self, 
                       results_dict: Dict[str, Dict[int, List[int]]],
                       test_df: pd.DataFrame,
                       all_items: Set[int],
                       k: int = 10) -> pd.DataFrame:
        """
        对比多种推荐方法的评估结果
        
        Args:
            results_dict: {method_name: {user_id: [item_id1, ...]}}
            test_df: 测试集
            all_items: 所有物品集合
            k: 推荐数量
            
        Returns:
            comparison_df: 对比结果 DataFrame
        """
        comparison_results = {}
        
        for method_name, recommended_items_dict in results_dict.items():
            metrics = self.evaluate_all(recommended_items_dict, test_df, all_items, k)
            comparison_results[method_name] = metrics
        
        comparison_df = pd.DataFrame(comparison_results).T
        comparison_df.index.name = 'Method'
        
        return comparison_df


def plot_comparison(comparison_df: pd.DataFrame, 
                    save_path: str = "results/comparison.png") -> None:
    """
    绘制评估指标对比图
    
    Args:
        comparison_df: 对比结果 DataFrame
        save_path: 保存路径
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 绘制对比图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('推荐算法性能对比', fontsize=16, fontweight='bold')
    
    metrics = ['recall', 'precision', 'ndcg', 'coverage']
    titles = ['Recall@K', 'Precision@K', 'NDCG@K', 'Coverage']
    
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        
        if metric in comparison_df.columns:
            values = comparison_df[metric].values
            bars = ax.bar(range(len(values)), values, color=colors[:len(values)], 
                         alpha=0.8, edgecolor='black')
            
            # 添加数值标签
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                       f'{value:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels(comparison_df.index, fontsize=10)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_ylabel('Score', fontsize=10)
            ax.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 1.0)
            ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"对比图已保存到: {save_path}")


if __name__ == "__main__":
    # 测试评估模块
    evaluator = RecommenderEvaluator()
    
    # 模拟数据
    test_df = pd.DataFrame({
        'userId': [1, 1, 1, 2, 2, 2],
        'movieId': [101, 102, 103, 101, 104, 105],
        'rating': [4.5, 3.0, 5.0, 4.0, 3.5, 2.0]
    })
    
    recommended = {
        1: [101, 102, 104, 105, 106],
        2: [103, 101, 102, 106, 107]
    }
    
    all_items = {101, 102, 103, 104, 105, 106, 107}
    
    metrics = evaluator.evaluate_all(recommended, test_df, all_items, k=5)
    print("评估指标:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
