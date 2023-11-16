'''
该文件用于处理材料化学式，使其改为字典类型，键值分别为原子序号和元素含量。
'''
import re
import torch
from collections import Counter
import periodictable
import csv
import argparse
import pandas as pd

def parse_formula(formula):
    # 使用正则表达式来匹配元素符号和数量
    matches = re.findall(r'([A-Z][a-z]*)(\d*)', formula)
    # 计算每种元素的数量
    counts = Counter()
    for element, count in matches:
        if count == '':
            count = 1
        else:
            count = int(count)
        counts[element] += count
    # 计算总数量
    total_count = sum(counts.values())
    elements = []
    proportions = []
    for element, count in counts.items():
        # 使用 periodictable 库来获取元素的原子序号
        atomic_number = getattr(periodictable, element).number
        elements.append(atomic_number)
        proportions.append(count / total_count)
    return torch.tensor(elements, dtype=torch.long), torch.tensor(proportions)  # 使用dtype=torch.long来创建长整型张量

import gemmi

def get_space_group_sym2num(symbol):
    if symbol == 'Unknown':
        return -1
    else:
        sg = gemmi.SpaceGroup(symbol)
        return sg.number

df = pd.read_csv('1m.csv')

df = df.dropna()

# 找出 'Material_Name' 列中非字符串类型的数据
non_str_rows = df['Material_Name'].apply(lambda x: not isinstance(x, str))

# 删除这些行
df = df[~non_str_rows]

if df['Space_Group'].isna().any():
    print("Warning: 'Space_Group' 列中存在 nan 值。这些值将被替换为默认值。")
    # 使用默认值填充 nan 值
    df['Space_Group'] = df['Space_Group'].fillna('Unknown')

df['atomic_numbers'], df['proportions'] = zip(*df['Material_Name'].map(parse_formula))
df['space_group_sym'] = df['Space_Group'].apply(get_space_group_sym2num)

df = df.drop(columns=['Material_Name'])
df = df.drop(columns=['Space_Group'])

import torch
from torch.nn.utils.rnn import pad_sequence

# 将列表转换为 PyTorch tensor
df['atomic_numbers'] = df['atomic_numbers'].apply(torch.tensor)
df['proportions'] = df['proportions'].apply(torch.tensor)

# 计算每个列表的长度，并保存到 'valid_len' 列
df['valid_len'] = df['atomic_numbers'].apply(len) + 1

# 填充列表
df['atomic_numbers'] = pad_sequence(df['atomic_numbers'], batch_first=True, padding_value=0).tolist()
df['proportions'] = pad_sequence(df['proportions'], batch_first=True, padding_value=0).tolist()

# 在每个列表的开始处插入一个零
df['atomic_numbers'] = df['atomic_numbers'].apply(lambda x: [0] + x)
df['proportions'] = df['proportions'].apply(lambda x: [0] + x)

df.to_csv('new_file.csv', index=False)