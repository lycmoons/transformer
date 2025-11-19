import pandas as pd
import torch
from torch import Tensor
import sentencepiece as spm
import numpy as np



# 读取训练数据
def get_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path, lineterminator="\n")



# 生成语料文件
def build_corpus(df: pd.DataFrame, en_corpus_path: str, de_corpus_path: str):
    en_sentences = df["en"].astype(str)
    de_sentences = df["de"].astype(str)
    with open(en_corpus_path, "w", encoding="utf-8") as f:
        for line in en_sentences:
            f.write(line.strip() + "\n")
    with open(de_corpus_path, "w", encoding="utf-8") as f:
        for line in de_sentences:
            f.write(line.strip() + "\n")



# 训练 BPE 模型构建词表
def train_bpe(input_file: str, output_file_prefix: str, vocab_size: int):
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=output_file_prefix,
        vocab_size=vocab_size,
        model_type='bpe',
        pad_id=3,
        pad_piece="<pad>",
    )



# 对字符串数据进行编码
def tokenize_and_pad(df: pd.DataFrame, en_bpe_prefix: str, de_bpe_prefix: str,
                     max_len: int, pad_id: int, bos_id: int, eos_id: int):
    en_sentences = df["en"].astype(str)
    de_sentences = df["de"].astype(str)
    en_sp = spm.SentencePieceProcessor(model_file=f'{en_bpe_prefix}.model')
    de_sp = spm.SentencePieceProcessor(model_file=f'{de_bpe_prefix}.model')
    src_input = []
    tgt_input = []
    label = []
    for line in en_sentences:
        ids = en_sp.encode(line.strip())[:max_len - 1] + [eos_id]
        if len(ids) < max_len:
            ids = ids + [pad_id] * (max_len - len(ids))
        src_input.append(ids)

    for line in de_sentences:
        ids = de_sp.encode(line.strip())[:max_len - 1]
        t = [bos_id] + ids
        l = ids + [eos_id]
        if len(t) < max_len:
            t = t + [pad_id] * (max_len - len(t))
            l = l + [pad_id] * (max_len - len(l))
        tgt_input.append(t)
        label.append(l)
    return np.array(src_input), np.array(tgt_input), np.array(label)
            


# 对标签进行平滑化处理，label: (batch_size, seq_len)
def label_smoothing(label: Tensor, vocab_size: int, smoothing_rate: float) -> Tensor:
    confidence = 1 - smoothing_rate
    smoothing_value = smoothing_rate / (vocab_size - 1)
    one_hot_label = torch.nn.functional.one_hot(label, vocab_size)
    smoothed_label = one_hot_label * (confidence - smoothing_value) + smoothing_value
    return smoothed_label.float()



# 创建一个 causal mask
def create_causal_mask(seq_len: int) -> Tensor:
    len_mask = torch.arange(seq_len).unsqueeze(0).expand(seq_len, seq_len)
    valid_len_mask = torch.arange(seq_len).unsqueeze(1).expand(seq_len, seq_len)
    return len_mask <= valid_len_mask



# 创建一个 padding mask，x: (batch_size, seq_len)
def create_padding_mask(x: Tensor, padding_id: int = 3) -> Tensor:
    return (x != padding_id)