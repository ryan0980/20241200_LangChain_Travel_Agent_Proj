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



'''

Country Code to which this hotel belongs.
这家酒店所属的国家/地区代码。


countyName
Country Name where this hotel belongs.
该酒店所属的国家/地区名称。


cityCode
The city Code where the hotel is located.
酒店所在的城市代码。


cityName
The city where the hotel is located.
酒店所在的城市。


HotelCode
A unique identifier for each hotel.
每家酒店的唯一标识符。


HotelName
The name of the hotel.
酒店的名称。


HotelRating
The star rating of the hotel, ranging from 1 to 5.
酒店的星级，从1星到5星。


Address
The address of the hotel.
酒店的地址。


Attractions
The Attractions nearby to the hotel.
酒店附近的景点。


Description
The detailed Description of the hotel.
酒店的详细描述。
'''