import torch
from torch import nn, Tensor
import math


''' 位置编码层 '''
class PositionEncoder(nn.Module):
    def __init__(self, dropout: float, device = None):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.device = device

    # x: (batch_size, seq_len, embed_size)
    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, embed_size = x.shape
        pos_map = torch.arange(seq_len, device=self.device).unsqueeze(1).expand(seq_len, embed_size)
        dim_map = torch.arange(embed_size, device=self.device).unsqueeze(0).expand(seq_len, embed_size)
        angle = pos_map / torch.pow(10000, 2 * dim_map / embed_size)
        pe = torch.zeros(seq_len, embed_size, device=self.device)
        pe[:, 0::2] = torch.sin(angle[:, 0::2])
        pe[:, 1::2] = torch.cos(angle[:, 1::2])
        pe = pe.unsqueeze(0).expand(batch_size, seq_len, embed_size)
        return self.dropout(x + pe)



''' 多头注意力层 '''
class MultiHeadAttention(nn.Module):
    def __init__(self, query_size: int, key_size: int, value_size: int, hidden_size: int, num_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.linear_q = nn.Linear(query_size, hidden_size)
        self.linear_k = nn.Linear(key_size, hidden_size)
        self.linear_v = nn.Linear(value_size, hidden_size)
        self.linear_o = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=-1)

    # query: (batch_size, num_querys, query_size)
    # key: (batch_size, num_pairs, key_size)
    # value: (batch_size, num_pairs, value_size)
    # mask: (batch_size, num_querys, num_pairs)
    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None) -> Tensor:
        batch_size = query.shape[0]
        num_querys = query.shape[1]
        num_pairs = key.shape[1]

        query = self.linear_q(query)
        key = self.linear_k(key)
        value = self.linear_v(value)

        query = query.reshape(batch_size, num_querys, self.num_heads, self.head_dim).permute(0, 2, 1, 3).reshape(batch_size * self.num_heads, num_querys, self.head_dim)
        key = key.reshape(batch_size, num_pairs, self.num_heads, self.head_dim).permute(0, 2, 3, 1).reshape(batch_size * self.num_heads, self.head_dim, num_pairs)
        value = value.reshape(batch_size, num_pairs, self.num_heads, self.head_dim).permute(0, 2, 1, 3).reshape(batch_size * self.num_heads, num_pairs, self.head_dim)
        score = torch.bmm(query, key) / math.sqrt(self.head_dim)

        if mask is not None:
            mask = mask.unsqueeze(1).expand(batch_size, self.num_heads, num_querys, num_pairs).reshape(batch_size * self.num_heads, num_querys, num_pairs)
            score = score.masked_fill(mask == False, float('-inf'))

        weight = self.softmax(score)
        output = torch.bmm(weight, value)
        output = output.reshape(batch_size, self.num_heads, num_querys, self.head_dim).permute(0, 2, 1, 3).reshape(batch_size, num_querys, self.hidden_size)
        return self.linear_o(output)



''' 前馈网络层 '''
class PositionWiseFFN(nn.Module):
    def __init__(self, input_size: int, ffn_size: int):
        super().__init__()
        self.linear1 = nn.Linear(input_size, ffn_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(ffn_size, input_size)

    # x: (batch_size, seq_len, model_size)
    def forward(self, x: Tensor) -> Tensor:
        return self.linear2(self.relu(self.linear1(x)))



''' 残差连接与归一化层 '''
class ResAndNorm(nn.Module):
    def __init__(self, norm_size: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(norm_size)
        self.dropout = nn.Dropout(dropout)

    # x: 原始数据
    # y: 输出数据
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return self.norm(x + self.dropout(y))



''' 编码器块 '''
class EncoderBlock(nn.Module):
    def __init__(self, model_size: int, num_heads: int, dropout: float, ffn_size: int):
        super().__init__()
        self.attention = MultiHeadAttention(model_size, model_size, model_size, model_size, num_heads)
        self.res_norm1 = ResAndNorm(model_size, dropout)
        self.ffn = PositionWiseFFN(model_size, ffn_size)
        self.res_norm2 = ResAndNorm(model_size, dropout)

    # x: (batch_size, seq_len, model_size)
    def forward(self, x: Tensor, padding_mask: Tensor = None) -> Tensor:
        output = self.res_norm1(x, self.attention(x, x, x, padding_mask))
        return self.res_norm2(output, self.ffn(output))


''' 解码器块 '''
class DecoderBlock(nn.Module):
    def __init__(self, model_size: int, num_heads: int, dropout: float, ffn_size: int):
        super().__init__()
        self.self_attention = MultiHeadAttention(model_size, model_size, model_size, model_size, num_heads)
        self.res_norm1 = ResAndNorm(model_size, dropout)
        self.cross_attention = MultiHeadAttention(model_size, model_size, model_size, model_size, num_heads)
        self.res_norm2 = ResAndNorm(model_size, dropout)
        self.ffn = PositionWiseFFN(model_size, ffn_size)
        self.res_norm3 = ResAndNorm(model_size, dropout)

    # x: (batch_size, seq_len, model_size)
    # y: (batch_size, seq_len, model_size)
    # causal_mask: (batch_size, seq_len, seq_len)
    # padding_mask: (batch_size, seq_len, seq_len)
    def forward(self, x: Tensor, y: Tensor, causal_mask: Tensor = None, padding_mask: Tensor = None) -> Tensor:
        output = self.res_norm1(x, self.self_attention(x, x, x, causal_mask))
        output = self.res_norm2(output, self.cross_attention(output, y, y, padding_mask))
        return self.res_norm3(output, self.ffn(output))



''' 带有 padding mask 的交叉熵损失 '''
class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    # pred: (batch_size, seq_len, vocab_size)
    # label: (batch_size, seq_len, vocab_size)
    # mask: (batch_size, seq_len)
    def forward(self, pred: Tensor, label: Tensor, mask: Tensor) -> Tensor:
        log_prob = torch.log_softmax(pred, dim=-1)
        loss = -(label * log_prob).sum(dim=-1)
        loss = loss * mask
        return loss.sum() / mask.sum()



# 学习率调度器
class LRScheduler:
    def __init__(self, optimizer, warmup_steps, model_size):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.lr_mul = model_size ** (-0.5)
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = self.lr_mul * min(self.step_num ** (-0.5), self.step_num * (self.warmup_steps ** -1.5))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr


# TODO 设计 BLEU 函数