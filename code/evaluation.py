import yaml
import json
import torch
import numpy as np
from model import Transformer
from utils import create_padding_mask
from modules import BLEUCalculator
import sentencepiece as spm


# 获取超参数
with open("/data/transformer/config.yaml", "r") as f:
    params = yaml.safe_load(f)


# 获取测试集编码器输入
test_src_input = np.load(params['test_src_input_path'])


# 获取测试集标签
with open(params['test_label_path'], "r") as j:
    test_label = json.load(j)


# 加载需要评估的模型
model_epoch = 10
state_dict = torch.load(f"{params['outputs_dir']}model_{model_epoch}.pth", map_location='cpu')
new_state_dict = {}
for k, v in state_dict.items():
    new_key = k.replace("module.", "")
    new_state_dict[new_key] = v
model = Transformer(
    params['num_encoder_blocks'],
    params['num_decoder_blocks'],
    params['vocab_size'],
    params['vocab_size'],
    params['model_size'],
    params['dropout'],
    params['num_heads'],
    params['ffn_size'],
)
model.load_state_dict(new_state_dict)


# 使用 GPU 进行模型推理工作
device = torch.device('cuda:0')
model = model.to(device)


# 对每一条编码器输入预测其翻译序列
model.eval()
predict = []
with torch.no_grad():
    for i in range(test_src_input.shape[0]):
        print(f'开始推理第 {i + 1} 条样本')
        x = torch.tensor(test_src_input[i]).reshape(1, params['max_len']).to(device)
        src_padding_mask = create_padding_mask(x).unsqueeze(1).to(device)
        predict.append(model.predict(x, src_padding_mask, params['bos_id'], params['eos_id'], params['max_len']))


# 计算整个测试集的 BLEU 值
bleu = BLEUCalculator()
test_size = len(predict)
sp = spm.SentencePieceProcessor(model_file=f"{params['de_bpe_prefix']}.model")
predictions = []
references = []
for i in range(test_size):
    predictions.append(sp.decode(predict[i]))
    references.append([sp.decode(test_label[i])])
bleu_score = bleu(predictions, references)
print(f'整个测试数据集的 BLEU 指标为：{bleu_score}')


# 10 epoch: 25.9838
# 20 epoch: 28.1370
# 30 epoch: 14.8891