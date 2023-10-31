import re
from collections import Counter
import periodictable
import gemmi

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
    # 获取元素符号列表和归一化后的摩尔比例数值列表，cls标记直接定为200，方便与pad的0进行区分
    elements = [200]
    proportions = [200]
    for element, count in counts.items():
        # 使用 periodictable 库来获取元素的原子序号
        atomic_number = getattr(periodictable, element).number
        elements.append(atomic_number)
        proportions.append(count / total_count)
    return elements, proportions



def get_space_group_sym2num(symbol):
    if symbol == 'UNKNOWN': return 2
    else:
        sg = gemmi.SpaceGroup(symbol)
        out = sg.number
        if out == 225:return 1
        else: return 0
        return sg.number



def replace_random_element(lst):
    new_lst = lst.copy()
    index = random.randint(1, len(lst)-1)
    original_value = new_lst[index]
    p = random.random()
    if p < 0.8:
        new_lst[index] = 199
    elif p < 0.9:
        pass
    else:
        new_lst[index] = random.randint(1, 120)
    index_lst = [0] * len(lst)
    index_lst[index] = 1
    return new_lst, [index], [original_value]

def predo(filename):
    df = pd.read_csv(filename)
    result = df.iloc[:, 0].apply(parse_formula)
    df.iloc[:, 0] = result.apply(lambda x: x[0])
    df = df.rename(columns={'name':'elements'})
    df.insert(1, 'portions', result.apply(lambda x: x[1]))
    df.insert(2, 'valid_lens', result.apply(lambda x: len(x[1])))
    df = df[df.iloc[:, 2] >= 2]
    df.iloc[:, 3] = df.iloc[:, 3].apply(get_space_group_sym2num)
    df.insert(4, 'P/W', result.apply(lambda x: replace_random_element(x[0])))
    max_len = 15
    df.insert(3, 'max_len', max_len)
    df[['all_element_ids', 'all_pred_positions','all_mlm_labels']] = pd.DataFrame(df.iloc[:, 5].tolist(), index=df.index)
    df['all_element_ids'] = df['all_element_ids'].apply(lambda x: x + [0] * (1+max_len - len(x)))
    df['portions'] = df['portions'].apply(lambda x: x + [0] * (1+max_len - len(x)))
    return df