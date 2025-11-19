import yaml
from utils import get_data, tokenize_and_pad
import numpy as np


# 获取超参数
with open("/data/transformer/config.yaml", "r") as f:
    params = yaml.safe_load(f)
print('成功获取超参数')


# 获取训练数据
df = get_data(params['train_csv_path'])
print('成功读取数据')


# 数据预处理
src_input, tgt_input, label = tokenize_and_pad(
    df, params['en_bpe_prefix'],
    params['de_bpe_prefix'],
    params['max_len'],
    params['pad_id'],
    params['bos_id'],
    params['eos_id']
)
print('成功完成序列转id并填充')


# 保存数据处理结果
np.save('/data/processed_data/src_input.npy', src_input)
print('成功保存编码器输入数据')
np.save('/data/processed_data/tgt_input.npy', tgt_input)
print('成功保存解码器输入数据')
np.save('/data/processed_data/label.npy', label)
print('成功保存标签数据')