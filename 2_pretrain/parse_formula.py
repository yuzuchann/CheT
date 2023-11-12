'''
该文件用于处理材料化学式，使其改为字典类型，键值分别为原子序号和元素含量。
'''
import re
import torch
from collections import Counter
import periodictable
import csv
import argparse

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

def parse_to_csv(input_file, output_file):
    with open(input_file, 'r') as csv_in, open(output_file, 'w', newline='') as csv_out:
        reader = csv.DictReader(csv_in)
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(csv_out, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            row['Materials_Name'] = parse_formula(row['Materials_Name'])
            writer.writerow(row)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a CSV file.')
    parser.add_argument('-i', type=str, required=True, help='Input CSV file')
    parser.add_argument('-o', type=str, required=True, help='Output CSV file')
    args = parser.parse_args()
    parse_to_csv(args.i, args.o)