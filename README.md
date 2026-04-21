# 🎬 混合推荐系统 (Hybrid Recommender System)

基于 MovieLens 数据集的混合推荐系统，实现并对比协同滤波与基于内容的推荐算法。

## 📋 项目简介

本项目实现了三种推荐策略：
1. **协同滤波 (Collaborative Filtering)**: 基于用户/物品相似度矩阵
2. **基于内容的推荐 (Content-Based)**: 利用电影标签(TF-IDF特征)
3. **混合推荐 (Hybrid)**: 加权融合前两种策略，提升推荐多样性

## 🚀 快速开始

### 环境配置

```bash
# 创建虚拟环境
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 安装依赖
pip install -r requirements.txt
```

### 运行项目

```bash
# 下载 MovieLens 数据集并运行推荐系统
python main.py
```

### 查看数据探索

```bash
jupyter notebook notebooks/exploration.ipynb
```

## 📁 项目结构

```
hybrid-recommender-system/
├── main.py                         # 主程序入口
├── data/                           # 数据集目录
├── src/
│   ├── data_loader.py              # 数据加载与预处理
│   ├── collaborative_filtering.py  # 协同滤波推荐
│   ├── content_based.py            # 基于内容的推荐
│   ├── hybrid_recommender.py       # 混合推荐策略
│   └── evaluation.py               # 评估指标
├── tests/                          # 单元测试
├── notebooks/                      # Jupyter 数据探索
└── results/                        # 推荐结果与图表
```

## 📊 算法说明

### 协同滤波 (Collaborative Filtering)
- **用户-用户协同**: 计算用户间余弦相似度，找到相似用户进行推荐
- **物品-物品协同**: 计算物品间余弦相似度，基于历史评分推荐
- 使用 SVD 矩阵分解降维提升效率

### 基于内容推荐 (Content-Based)
- 利用电影的 genres 标签构建 TF-IDF 特征向量
- 计算电影间的余弦相似度
- 根据用户已评分电影推荐相似电影

### 混合策略 (Hybrid)
- **线性加权**: `score = α × CF_score + (1-α) × CB_score`
- 可调节权重参数 α (默认 0.5)
- 支持多样性优化重排

## 📈 评估指标

- **Recall@K**: 召回率 - 推荐列表中用户实际喜欢的物品比例
- **Precision@K**: 准确率 - 推荐列表中相关物品的比例
- **Coverage**: 覆盖率 - 被推荐出的物品占总物品的比例
- **Diversity**: 多样性 - 推荐列表中物品的平均差异度

## 🔧 配置参数

在 `main.py` 中可以调整：

```python
CONFIG = {
    'test_size': 0.2,           # 测试集比例
    'top_k': 10,                # 推荐数量
    'cf_alpha': 0.5,            # 协同滤波权重
    'n_components': 50,         # SVD 降维维度
    'similarity_metric': 'cosine'  # 相似度计算方法
}
```

## 📝 输出结果

运行后在 `results/` 目录下生成：
- `recommendation_results.csv`: 各算法推荐结果
- `evaluation_metrics.csv`: 评估指标对比
- `recall_comparison.png`: 召回率对比图
- `precision_comparison.png`: 准确率对比图

## 🧪 运行测试

```bash
pytest tests/ -v
```

## 📚 数据集

使用 [MovieLens Latest Dataset](https://grouplens.org/datasets/movielens/)，包含：
- `ratings.csv`: 用户评分数据
- `movies.csv`: 电影信息与标签
- `tags.csv`: 用户标签数据