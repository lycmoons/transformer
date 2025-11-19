import pandas as pd
import re
from collections import Counter
from typing import Dict, Tuple, List
import torch
from torch import Tensor
import os

# 读取训练数据
def get_train_data() -> pd.DataFrame:
    file_path = os.path.join("data", "wmt14_translate_de-en_train.csv")
    print('成功读取数据')
    return pd.read_csv(file_path, lineterminator="\n")



# 读取测试数据
def get_test_data() -> pd.DataFrame:
    return pd.read_csv("data\wmt14_translate_de-en_test.csv", lineterminator='\n')



# 数据清洗
def clean_text(df: pd.DataFrame) -> pd.DataFrame:
    def clean(text):
        text = text.lower()
        text = re.sub(r'([^\w\s])', r' \1 ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    df['de'] = df['de'].apply(clean)
    df['en'] = df['en'].apply(clean)
    print('数据清洗完成')
    return df



# 对每个序列进行 token 划分，转为列表数据
def split_token(df: pd.DataFrame) -> Tuple[List[List[str]], List[List[str]]]:
    src_tokens_list = [s.split() for s in df['en'].tolist()]
    tgt_tokens_list = [s.split() for s in df['de'].tolist()]
    print('序列分词完成')
    return src_tokens_list, tgt_tokens_list



# 构建词表
def build_vocab(tokens_list: List[List[str]]) -> Tuple[Dict[str, int], Dict[int, str], int]:
    token_counter = Counter(tok for sent in tokens_list for tok in sent)

    token_to_idx = {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3}
    for i, (tok, _) in enumerate(token_counter.most_common(), start=4):
        token_to_idx[tok] = i

    idx_to_token = {i: w for w, i in token_to_idx.items()}

    print('词表构建完成')
    return token_to_idx, idx_to_token, len(token_to_idx)



# 将 token 序列转换成 idx，并进行适当的填充
def encode_tokens(tokens_list: List[List[str]], token_to_idx: Dict[str, int], max_len: int, encode_type: str) -> List[List[int]]:
    output = []
    for tokens in tokens_list:
        tokens = tokens[:max_len - 1]
        idxs = [token_to_idx.get(token, token_to_idx['<unk>']) for token in tokens]
        if encode_type == 'src' or encode_type == 'label':
            idxs = idxs + [token_to_idx['<eos>']]
        elif encode_type == 'target':
            idxs = [token_to_idx['<bos>']] + idxs
        if len(idxs) < max_len:
            idxs += [token_to_idx['<pad>']] * (max_len - len(idxs))
        output.append(idxs)

    print('序列编码完成')
    return torch.tensor(output)



# 对标签进行平滑化处理，label: (batch_size, seq_len)
def label_smoothing(label: Tensor, vocab_size: int, smoothing_rate: float) -> Tensor:
    confidence = 1 - smoothing_rate
    smoothing_value = smoothing_rate / (vocab_size - 1)
    one_hot_label = torch.nn.functional.one_hot(label, vocab_size)
    smoothed_label = one_hot_label * (confidence - smoothing_value) + smoothing_value

    print('标签平滑处理完成')
    return smoothed_label.float()



# 创建一个 causal mask
def create_causal_mask(seq_len: int):
    len_mask = torch.arange(seq_len).unsqueeze(0).expand(seq_len, seq_len)
    valid_len_mask = torch.arange(seq_len).unsqueeze(1).expand(seq_len, seq_len)

    print('causal mask 创建完成')
    return len_mask <= valid_len_mask



# 创建一个 padding mask，x: (batch_size, seq_len)
def create_padding_mask(x: Tensor, padding_id: int = 0):
    print('padding mask 创建完成')
    return (x != padding_id)