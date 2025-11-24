import yaml
from utils import get_data, build_corpus, train_bpe

# 获取超参数
with open("/data/transformer/config.yaml", "r") as f:
    params = yaml.safe_load(f)
print('成功获取超参数')


# 获取训练数据
df = get_data(params['train_csv_path'])
print('成功读取数据')


# 构建语料库
build_corpus(df, params['en_corpus_path'], params['de_corpus_path'])
print('成功构建语料库')


# 训练 BPE 模型
train_bpe(params['en_corpus_path'], params['en_bpe_prefix'], params['vocab_size'])
print('成功训练英语BPE模型')
train_bpe(params['de_corpus_path'], params['de_bpe_prefix'], params['vocab_size'])
print('成功训练德语BPE模型')