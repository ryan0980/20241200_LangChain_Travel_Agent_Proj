import os
import shutil
import kagglehub

# 下载数据集
path = kagglehub.dataset_download("raj713335/tbo-hotels-dataset")

# 指定目标目录
target_dir = "G:/Code/Projects/GWU/24_FA/AML/Final_proj"

# 确保目标目录存在
os.makedirs(target_dir, exist_ok=True)

# 移动文件到目标目录
for file_name in os.listdir(path):
    shutil.move(os.path.join(path, file_name), target_dir)

print("数据集已下载并移动到:", target_dir)
