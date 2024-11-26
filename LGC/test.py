
import pandas as pd
import csv
movie_path = '../data/ml-1m/movies.dat'
rating_path = '../data/ml-1m/ratings.dat'

with open(movie_path, 'r',encoding='utf-8') as file:
    content = file.read()


def convert_dat_to_csv(dat_data, csv_file_path):
    """
    Converts dat format data to CSV format.

    Args:
        dat_data (str): The content of the dat file as a string.
        csv_file_path (str): The path where the CSV file will be saved.
    """
    # 分割数据为行
    lines = dat_data.strip().split('\n')

    # 创建CSV文件并写入数据
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file, delimiter='::')

        # 写入标题行
        writer.writerow(['movieId', 'title', 'genres'])

        # 遍历每一行数据
        for line in lines:
            if line.strip():  # 确保不处理空行
                # 分割每行为movieId, title, genres
                parts = line.split('::')
                if len(parts) == 3:
                    movieId, title, genres = parts
                    writer.writerow([movieId, title, genres])

convert_dat_to_csv(content, '../data/ml-1m/movies.csv')
#print(content)
