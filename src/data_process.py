import numpy as np
import pandas as pd

rating_file ='../data/ml-20m/'+'ratings.csv'
rating_pd = pd.read_csv(rating_file)
rating_pd.head()
rating_pd .to_csv('../data/ratings.txt', sep='\t', index=False, header=False)
rating_np = np.loadtxt('../data/ratings.txt', delimiter='\t',dtype=np.int64)

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

positive_samples, negative_samples = classify_ratings(rating_np)
np.save('../data/data_mid/positive_samples.npy', positive_samples)
np.save('../data/data_mid/negative_samples.npy', negative_samples)
###保存正负数据

def dataset_split(rating_np, ratio):
    print('splitting dataset ...')

    # train:eval:test = 6:2:2
    eval_ratio = 0.2
    test_ratio = 0.2
    n_ratings = rating_np.shape[0]

    eval_indices = np.random.choice(list(range(n_ratings)), size=int(n_ratings * eval_ratio), replace=False)
    left = set(range(n_ratings)) - set(eval_indices)
    test_indices = np.random.choice(list(left), size=int(n_ratings * test_ratio), replace=False)
    train_indices = list(left - set(test_indices))
    if ratio < 1:
        train_indices = np.random.choice(list(train_indices), size=int(len(train_indices) * ratio), replace=False)

    train_data = rating_np[train_indices]
    eval_data = rating_np[eval_indices]
    test_data = rating_np[test_indices]

    return train_data, eval_data, test_data


