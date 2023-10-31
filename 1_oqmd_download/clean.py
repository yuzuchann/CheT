import pandas as pd

# 读取CSV文件
df = pd.read_csv('all_materials_data.csv')

# 删除重复行
df = df.drop_duplicates()

# 将结果保存到新的CSV文件
df.to_csv('your_file_no_duplicates.csv', index=False)
