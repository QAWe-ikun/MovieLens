"""
数据加载与预处理模块
负责下载、解析 MovieLens 数据集，构建训练/测试集
"""

import os
import zipfile
import requests
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict


class MovieLensDataLoader:
    """MovieLens 数据加载器"""
    
    # MovieLens 100k 数据集下载链接
    DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    
    def __init__(self, data_dir: str = "data"):
        """
        初始化数据加载器
        
        Args:
            data_dir: 数据存储目录
        """
        self.data_dir = data_dir
        self.ratings_df = None
        self.movies_df = None
        self.tags_df = None
        
        os.makedirs(data_dir, exist_ok=True)
    
    def download_data(self) -> None:
        """下载 MovieLens 数据集"""
        zip_path = os.path.join(self.data_dir, "ml-latest-small.zip")
        extract_path = os.path.join(self.data_dir, "ml-latest-small")
        
        # 检查是否已存在数据
        if os.path.exists(extract_path) and os.listdir(extract_path):
            print(f"数据集已存在: {extract_path}")
            return
        
        print("正在下载 MovieLens 数据集...")
        response = requests.get(self.DATA_URL, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(zip_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="下载") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        print("正在解压...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.data_dir)
        
        print(f"数据集已下载到: {extract_path}")
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        加载评分、电影和标签数据
        
        Returns:
            ratings_df: 评分数据 (userId, movieId, rating, timestamp)
            movies_df: 电影数据 (movieId, title, genres)
            tags_df: 标签数据 (userId, movieId, tag, timestamp)
        """
        if self.ratings_df is None:
            self.download_data()
            
            extract_path = os.path.join(self.data_dir, "ml-latest-small")
            
            # 加载数据
            self.ratings_df = pd.read_csv(os.path.join(extract_path, "ratings.csv"))
            self.movies_df = pd.read_csv(os.path.join(extract_path, "movies.csv"))
            self.tags_df = pd.read_csv(os.path.join(extract_path, "tags.csv"))
            
            print(f"加载完成:")
            print(f"  - 评分数据: {len(self.ratings_df)} 条")
            print(f"  - 电影数据: {len(self.movies_df)} 部")
            print(f"  - 标签数据: {len(self.tags_df)} 条")
        
        return self.ratings_df, self.movies_df, self.tags_df
    
    def preprocess_ratings(self, min_ratings: int = 5) -> pd.DataFrame:
        """
        预处理评分数据，过滤低活跃度用户和物品
        
        Args:
            min_ratings: 最小评分数量
            
        Returns:
            过滤后的评分数据
        """
        if self.ratings_df is None:
            self.load_data()
        
        # 过滤评分数少于 min_ratings 的用户
        user_counts = self.ratings_df.groupby('userId').size()
        valid_users = user_counts[user_counts >= min_ratings].index
        
        # 过滤被评分次数少于 min_ratings 的电影
        movie_counts = self.ratings_df.groupby('movieId').size()
        valid_movies = movie_counts[movie_counts >= min_ratings].index
        
        # 应用过滤
        filtered_df = self.ratings_df[
            (self.ratings_df['userId'].isin(valid_users)) &
            (self.ratings_df['movieId'].isin(valid_movies))
        ]
        
        print(f"过滤后: {len(filtered_df)} 条评分")
        return filtered_df
    
    def create_train_test_split(self, test_size: float = 0.2, 
                                 random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        创建训练集和测试集（按用户分组）
        
        Args:
            test_size: 测试集比例
            random_state: 随机种子
            
        Returns:
            train_df: 训练集
            test_df: 测试集
        """
        if self.ratings_df is None:
            self.load_data()
        
        train_df = pd.DataFrame()
        test_df = pd.DataFrame()
        
        # 对每个用户进行分割
        for userId, group in self.ratings_df.groupby('userId'):
            train, test = train_test_split(
                group, 
                test_size=test_size, 
                random_state=random_state
            )
            train_df = pd.concat([train_df, train])
            test_df = pd.concat([test_df, test])
        
        print(f"训练集: {len(train_df)} 条, 测试集: {len(test_df)} 条")
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
    
    def build_user_item_matrix(self, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """
        构建用户-物品评分矩阵
        
        Args:
            ratings_df: 评分数据
            
        Returns:
            user_item_matrix: 用户-物品矩阵 (用户为行，物品为列)
        """
        user_item_matrix = ratings_df.pivot_table(
            index='userId',
            columns='movieId',
            values='rating'
        )
        return user_item_matrix
    
    def get_user_interactions(self, ratings_df: pd.DataFrame) -> Dict[int, set]:
        """
        获取每个用户的交互物品集合
        
        Args:
            ratings_df: 评分数据
            
        Returns:
            user_interactions: {userId: {movieId1, movieId2, ...}}
        """
        user_interactions = {}
        for userId, group in ratings_df.groupby('userId'):
            user_interactions[userId] = set(group['movieId'].values)
        return user_interactions
    
    def get_item_users(self, ratings_df: pd.DataFrame) -> Dict[int, set]:
        """
        获取每个物品的交互用户集合
        
        Args:
            ratings_df: 评分数据
            
        Returns:
            item_users: {movieId: {userId1, userId2, ...}}
        """
        item_users = {}
        for movieId, group in ratings_df.groupby('movieId'):
            item_users[movieId] = set(group['userId'].values)
        return item_users


if __name__ == "__main__":
    # 测试数据加载
    loader = MovieLensDataLoader()
    ratings, movies, tags = loader.load_data()
    
    print("\n评分数据样例:")
    print(ratings.head())
    
    print("\n电影数据样例:")
    print(movies.head())
    
    print("\n用户-物品矩阵形状:")
    matrix = loader.build_user_item_matrix(ratings)
    print(matrix.shape)
