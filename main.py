"""
混合推荐系统主程序
整合数据加载、模型训练、评估和可视化
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 添加 src 目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import MovieLensDataLoader
from src.collaborative_filtering import CollaborativeFilteringRecommender
from src.content_based import ContentBasedRecommender
from src.hybrid_recommender import HybridRecommender, DynamicWeightHybridRecommender
from src.evaluation import RecommenderEvaluator, plot_comparison


# 配置参数
CONFIG = {
    'test_size': 0.2,              # 测试集比例
    'top_k': 10,                   # 推荐数量
    'cf_alpha': 0.5,               # 协同滤波权重（混合推荐）
    'n_components': 50,            # SVD 降维维度
    'similarity_metric': 'cosine', # 相似度计算方法
    'min_ratings': 5,              # 最小评分数量过滤
    'random_state': 42             # 随机种子
}


def main():
    """主程序入口"""
    print("=" * 70)
    print("混合推荐系统 - Hybrid Recommender System")
    print("=" * 70)
    
    start_time = time.time()
    
    # 创建结果目录
    os.makedirs("results", exist_ok=True)
    
    # ==================== 1. 数据加载与预处理 ====================
    print("=" * 70)
    print("步骤 1: 数据加载与预处理")
    print("=" * 70)
    
    loader = MovieLensDataLoader(data_dir="data")
    ratings, movies, tags = loader.load_data()
    
    # 预处理：过滤低活跃度用户和物品
    filtered_ratings = loader.preprocess_ratings(min_ratings=CONFIG['min_ratings'])
    
    # 划分训练集和测试集
    train_df, test_df = loader.create_train_test_split(
        test_size=CONFIG['test_size'],
        random_state=CONFIG['random_state']
    )
    
    # 构建用户-物品矩阵
    user_item_matrix = loader.build_user_item_matrix(train_df)
    
    print(f"\n数据统计:")
    print(f"  - 用户数: {user_item_matrix.shape[0]}")
    print(f"  - 电影数: {user_item_matrix.shape[1]}")
    print(f"  - 评分数: {len(train_df)}")
    print(f"  - 稀疏度: {1 - len(train_df) / (user_item_matrix.shape[0] * user_item_matrix.shape[1]):.4f}")
    
    # 获取所有物品集合
    all_items = set(user_item_matrix.columns)
    
    # ==================== 2. 训练协同过滤模型 ====================
    print("\n" + "=" * 70)
    print("步骤 2: 训练协同过滤模型")
    print("=" * 70)
    
    cf_recommender = CollaborativeFilteringRecommender(
        n_components=CONFIG['n_components'],
        similarity_metric=CONFIG['similarity_metric']
    )
    cf_recommender.fit(user_item_matrix)
    
    # ==================== 3. 训练基于内容推荐模型 ====================
    print("\n" + "=" * 70)
    print("步骤 3: 训练基于内容推荐模型")
    print("=" * 70)
    
    cb_recommender = ContentBasedRecommender()
    cb_recommender.fit(movies)
    
    # ==================== 4. 生成推荐结果 ====================
    print("\n" + "=" * 70)
    print("步骤 4: 生成推荐结果")
    print("=" * 70)
    
    print("\n为用户生成推荐 (User-CF)...")
    cf_recommendations = cf_recommender.recommend_all_users(
        top_k=CONFIG['top_k'],
        method='user_cf'
    )
    
    print("\n为用户生成推荐 (Item-CF)...")
    icf_recommendations = cf_recommender.recommend_all_users(
        top_k=CONFIG['top_k'],
        method='item_cf'
    )
    
    print("\n为用户生成推荐 (Content-Based)...")
    cb_recommendations = cb_recommender.recommend_all_users(
        user_item_matrix=user_item_matrix,
        top_k=CONFIG['top_k']
    )
    
    # ==================== 5. 混合推荐 ====================
    print("\n" + "=" * 70)
    print("步骤 5: 混合推荐策略")
    print("=" * 70)
    
    # 固定权重混合
    hybrid_recommender = HybridRecommender(
        cf_weight=CONFIG['cf_alpha'],
        diversity_weight=0.0
    )
    
    print("\n生成混合推荐 (Hybrid, α=0.5)...")
    hybrid_recommendations = hybrid_recommender.recommend_all_users(
        cf_recommendations_dict=cf_recommendations,
        cb_recommendations_dict=cb_recommendations,
        top_k=CONFIG['top_k']
    )
    
    # 动态权重混合
    dynamic_hybrid_recommender = DynamicWeightHybridRecommender(
        base_cf_weight=CONFIG['cf_alpha']
    )
    
    print("\n生成动态权重混合推荐 (Dynamic Hybrid)...")
    dynamic_hybrid_recommendations = dynamic_hybrid_recommender.recommend_all_users(
        cf_recommendations_dict=cf_recommendations,
        cb_recommendations_dict=cb_recommendations,
        user_item_matrix=user_item_matrix,
        top_k=CONFIG['top_k']
    )
    
    # ==================== 6. 评估推荐结果 ====================
    print("\n" + "=" * 70)
    print("步骤 6: 评估推荐结果")
    print("=" * 70)
    
    evaluator = RecommenderEvaluator()
    
    # 准备评估数据
    results_dict = {
        'User-CF': {uid: [item for item, _ in recs] for uid, recs in cf_recommendations.items()},
        'Item-CF': {uid: [item for item, _ in recs] for uid, recs in icf_recommendations.items()},
        'Content-Based': {uid: [item for item, _ in recs] for uid, recs in cb_recommendations.items()},
        f'Hybrid (α={CONFIG["cf_alpha"]})': {uid: [item for item, _ in recs] for uid, recs in hybrid_recommendations.items()},
        'Dynamic Hybrid': {uid: [item for item, _ in recs] for uid, recs in dynamic_hybrid_recommendations.items()}
    }
    
    print("\n计算评估指标...")
    comparison_df = evaluator.compare_methods(
        results_dict,
        test_df,
        all_items, # type: ignore
        k=CONFIG['top_k']
    )
    
    print("\n" + "=" * 70)
    print("📊 评估结果对比")
    print("=" * 70)
    print(comparison_df.round(4))
    
    # 保存评估结果
    comparison_df.to_csv("results/evaluation_metrics.csv")
    print("\n✅ 评估结果已保存到: results/evaluation_metrics.csv")
    
    # ==================== 7. 可视化对比结果 ====================
    print("\n" + "=" * 70)
    print("步骤 7: 可视化对比结果")
    print("=" * 70)
    
    plot_comparison(comparison_df, save_path="results/comparison.png")
    
    # 绘制详细对比图
    plot_detailed_comparison(comparison_df)
    
    # ==================== 8. 保存推荐示例 ====================
    print("\n" + "=" * 70)
    print("步骤 8: 保存推荐示例")
    print("=" * 70)
    
    # 选择几个用户展示推荐结果
    sample_users = list(results_dict['User-CF'].keys())[:5]
    
    recommendation_examples = []
    for user_id in sample_users:
        user_recs = {
            'user_id': user_id,
            'user_cf': results_dict['User-CF'].get(user_id, [])[:5],
            'content_based': results_dict['Content-Based'].get(user_id, [])[:5],
            'hybrid': results_dict[f'Hybrid (α={CONFIG["cf_alpha"]})'].get(user_id, [])[:5]
        }
        recommendation_examples.append(user_recs)
    
    # 保存推荐示例
    rec_df = pd.DataFrame(recommendation_examples)
    rec_df.to_csv("results/recommendation_examples.csv", index=False)
    print("\n✅ 推荐示例已保存到: results/recommendation_examples.csv")
    
    # ==================== 9. 总结 ====================
    print("\n" + "=" * 70)
    print("推荐系统运行完成")
    print("=" * 70)
    
    elapsed_time = time.time() - start_time
    print(f"\n⏱️ 总运行时间: {elapsed_time:.2f} 秒")
    
    print(f"\n生成文件:")
    print(f"  - results/evaluation_metrics.csv (评估指标)")
    print(f"  - results/comparison.png (对比图)")
    print(f"  - results/recommendation_examples.csv (推荐示例)")
    
    # 找出最佳方法
    best_method = comparison_df['recall'].idxmax()
    best_recall = comparison_df['recall'].max()
    print(f"\n最佳方法 (按召回率): {best_method} (Recall@{CONFIG['top_k']} = {best_recall:.4f})")
    
    print("\n" + "=" * 70)


def plot_detailed_comparison(comparison_df: pd.DataFrame):
    """绘制详细的对比图"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('推荐算法详细对比', fontsize=16, fontweight='bold')
    
    methods = comparison_df.index.tolist()
    x_pos = np.arange(len(methods))
    
    # Recall 对比
    ax1 = axes[0]
    colors1 = plt.cm.Blues(np.linspace(0.4, 0.8, len(methods))) # type: ignore
    bars1 = ax1.bar(x_pos, comparison_df['recall'].values, color=colors1, alpha=0.8, edgecolor='black')
    for bar, value in zip(bars1, comparison_df['recall'].values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{value:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(methods, rotation=15, ha='right')
    ax1.set_title('Recall@K', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Recall', fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    
    # Precision 对比
    ax2 = axes[1]
    colors2 = plt.cm.Greens(np.linspace(0.4, 0.8, len(methods))) # type: ignore
    bars2 = ax2.bar(x_pos, comparison_df['precision'].values, color=colors2, alpha=0.8, edgecolor='black')
    for bar, value in zip(bars2, comparison_df['precision'].values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{value:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(methods, rotation=15, ha='right')
    ax2.set_title('Precision@K', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Precision', fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    
    # NDCG 对比
    ax3 = axes[2]
    colors3 = plt.cm.Oranges(np.linspace(0.4, 0.8, len(methods))) # type: ignore
    bars3 = ax3.bar(x_pos, comparison_df['ndcg'].values, color=colors3, alpha=0.8, edgecolor='black')
    for bar, value in zip(bars3, comparison_df['ndcg'].values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{value:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(methods, rotation=15, ha='right')
    ax3.set_title('NDCG@K', fontsize=12, fontweight='bold')
    ax3.set_ylabel('NDCG', fontsize=10)
    ax3.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("results/detailed_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ 详细对比图已保存到: results/detailed_comparison.png")


if __name__ == "__main__":
    main()
