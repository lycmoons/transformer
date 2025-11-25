import yaml
from model import Transformer
import torch
import sentencepiece as spm
from utils import create_padding_mask



# 获取超参数
with open("/data/transformer/config.yaml", "r") as f:
    params = yaml.safe_load(f)



# 加载模型
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


# 序列的 tokenizer
en_sp = spm.SentencePieceProcessor(model_file=f"{params['en_bpe_prefix']}.model")
de_sp = spm.SentencePieceProcessor(model_file=f"{params['de_bpe_prefix']}.model")

with torch.no_grad():
    while True:
        line = input('请输入需要翻译的英文：')
        if(line == 'exit'):
            print('程序退出')
            break
        ids = en_sp.encode(line.strip())[:params['max_len'] - 1] + [params['eos_id']]
        if len(ids) < params['max_len']:
            ids = ids + [params['pad_id']] * (params['max_len'] - len(ids))
        src_input = torch.tensor([ids]).to(device)
        src_padding_mask = create_padding_mask(src_input).unsqueeze(1).to(device)
        predict_ids = model.predict(src_input, src_padding_mask, params['bos_id'], params['eos_id'], params['max_len'])
        print(f'德文机器翻译：{de_sp.decode(predict_ids)}')