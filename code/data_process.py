import yaml
from utils import get_data, tokenize_and_pad, tokenize4test
import numpy as np
import json


# 获取超参数
with open("/data/transformer/config.yaml", "r") as f:
    params = yaml.safe_load(f)
print('成功获取超参数')


# 获取训练数据
df = get_data(params['train_csv_path'])
print('成功读取数据')


# 数据预处理
src_input, tgt_input, label = tokenize_and_pad(
    df,
    params['en_bpe_prefix'],
    params['de_bpe_prefix'],
    params['max_len'],
    params['pad_id'],
    params['bos_id'],
    params['eos_id']
)
print('成功完成序列转id并填充')


# 保存数据处理结果
np.save(params['src_input_path'], src_input)
print('成功保存编码器输入数据')
np.save(params['tgt_input_path'], tgt_input)
print('成功保存解码器输入数据')
np.save(params['label_path'], label)
print('成功保存标签数据')


# 获取测试集数据
test_df = get_data(params['test_csv_path'])
print('成功读取数据')


# 数据预处理
test_src_input, test_label = tokenize4test(
    test_df,
    params['en_bpe_prefix'],
    params['de_bpe_prefix'],
    params['max_len'],
    params['pad_id'],
    params['eos_id']
)
print('成功完成序列转id并填充')


# 保存数据处理结果
np.save(params['test_src_input_path'], test_src_input)
print('成功保存测试编码器输入数据')
with open(params['test_label_path'], "w") as j:
    json.dump(test_label, j)
print('成功保存测试标签数据')
