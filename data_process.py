import yaml
import utils
import os
import numpy as np



# 读取超参数
with open("config.yaml", "r") as f:
    params = yaml.safe_load(f)



# 创建文件夹
os.makedirs('processed_data', exist_ok=True)



# 数据预处理，得到编码器、解码器的输入，以及最终用于计算损失函数的平滑标签
df = utils.get_train_data()
df = utils.clean_text(df)
src_tokens_list, tgt_tokens_list = utils.split_token(df)
src_token_to_idx, src_idx_to_token, src_vocab_size = utils.build_vocab(src_tokens_list)
tgt_token_to_idx, tgt_idx_to_token, tgt_vocab_size = utils.build_vocab(tgt_tokens_list)
label = utils.encode_tokens(tgt_tokens_list, tgt_token_to_idx, params['max_len'], 'label')
src_input = utils.encode_tokens(src_tokens_list, src_token_to_idx, params['max_len'], 'src')   # (batch_size, seq_len)
tgt_input = utils.encode_tokens(tgt_tokens_list, tgt_token_to_idx, params['max_len'], 'tgt')   # (batch_size, seq_len)



# 保存结果
np.save(os.path.join('processed_data', 'label.npy'), np.array(label))
np.save(os.path.join('processed_data', 'src_input.npy'), np.array(src_input))
np.save(os.path.join('processed_data', 'tgt_input.npy'), np.array(tgt_input))
np.save(os.path.join('processed_data', 'vocab_size.npy'), np.array([src_vocab_size, tgt_vocab_size]))

# 字典类型数据读取：np.load("***.npy", allow_pickle=True).item()
np.save(os.path.join('processed_data', 'src_token_to_idx.npy'), src_token_to_idx)
np.save(os.path.join('processed_data', 'src_idx_to_token.npy'), src_idx_to_token)
np.save(os.path.join('processed_data', 'tgt_token_to_idx.npy'), tgt_token_to_idx)
np.save(os.path.join('processed_data', 'tgt_idx_to_token.npy'), tgt_idx_to_token)