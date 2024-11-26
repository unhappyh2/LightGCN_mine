# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。
import torch
import numpy as np


# 按装订区域中的绿色按钮以运行脚本。
rating_np = np.loadtxt('./data/ratings.txt', delimiter='\t',dtype=np.int64)
print(rating_np[:10])