import numpy as np
import os
import pandas as pd

def load_rating(args):
    print('reading rating file ...')

    # reading rating file
    rating_file = '../data/' + args.dataset + '/ratings_final'
    if os.path.exists(rating_file + '.npy'):
        rating_np = np.load(rating_file + '.npy')
    else:
        rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int64)
        np.save(rating_file + '.npy', rating_np)

    n_user = len(set(rating_np[:, 0]))
    n_item = len(set(rating_np[:, 1]))
    train_data, eval_data, test_data = dataset_split(rating_np, args)

    return n_user, n_item, train_data, eval_data, test_data


def classify_ratings(data):
    # 将数据转换为DataFrame以便于处理
    df = pd.DataFrame(data, columns=['UserID', 'MovieID', 'Rating', 'Timestamp'])

    # 计算每个用户的平均评分
    user_avg_ratings = df.groupby('UserID')['Rating'].mean().to_dict()

    # 初始化正样本和负样本列表
    positive_samples = []
    negative_samples = []

    # 遍历数据，分类正负样本
    for index, row in df.iterrows():
        user_id = row['UserID']
        rating = row['Rating']
        avg_rating = user_avg_ratings[user_id]

        # 判断是正样本还是负样本
        if rating > avg_rating:
            positive_samples.append([user_id, row['MovieID'], rating, row['Timestamp']])
        else:
            negative_samples.append([user_id, row['MovieID'], rating, row['Timestamp']])

    return positive_samples, negative_samples