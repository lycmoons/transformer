import yaml
import utils
import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformer import Transformer
from modules import MaskedCrossEntropyLoss, LRScheduler
from torch.optim import Adam
from torch.amp import autocast, GradScaler



# 读取超参数
with open("/data/transformer/config.yaml", "r") as f:
    params = yaml.safe_load(f)
print('成功获取超参数')



# 设计数据的懒加载模式
class LazyDataset(Dataset):
    def __init__(self, src_input_path, tgt_input_path, label_path):
        super().__init__()
        self.src_input = np.load(src_input_path, mmap_mode='r')
        self.tgt_input = np.load(tgt_input_path, mmap_mode='r')
        self.label = np.load(label_path, mmap_mode='r')

    def __len__(self):
        return self.label.shape[0]
    
    def __getitem__(self, index):
        return torch.tensor(self.src_input[index]), torch.tensor(self.tgt_input[index]), torch.tensor(self.label[index])



# 使用 Dataset 和 DataLoader 封装数据
dataset = LazyDataset(params['src_input_path'], params['tgt_input_path'], params['label_path'])
dataloader = DataLoader(
    dataset,
    batch_size=params['batch_size'],
    shuffle=True, 
    num_workers=params['num_workers'],
    pin_memory=True,
    persistent_workers=True,
    drop_last=True
)
print('成功创建数据加载器')



# 获取可用设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# 创建模型实例
model = Transformer(
    params['num_encoder_blocks'],
    params['num_decoder_blocks'],
    params['vocab_size'],
    params['vocab_size'],
    params['model_size'],
    params['dropout'],
    params['num_heads'],
    params['ffn_size'],
    device
)
print('成功创建模型实例')


# 创建优化指标、优化器、学习率调度器
criterion = MaskedCrossEntropyLoss()
print('成功创建优化指标')
optimizer = Adam(
    model.parameters(),
    params['learning_rate'],
    (params['beta1'], params['beta2']),
    params['eps']
)
print('成功创建优化器')
scheduler = LRScheduler(optimizer, params['warmup_step'], params['model_size'])
print('成功创建学习率调度器')



# 解码器输入的 causal mask
tgt_causal_mask = utils.create_causal_mask(params['max_len']).unsqueeze(0).repeat(params['batch_size'], 1, 1)
print('成功生成causal mask')


# 指定使用可用设备进行模型训练
model = model.to(device)
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
tgt_causal_mask = tgt_causal_mask.to(device, non_blocking=True)
print('成功将模型放置GPU上')



# 创建文件夹用于保存模型结果
os.makedirs('outputs', exist_ok=True)



# 设计训练过程
scaler = GradScaler()
lr_histories = []
loss_histories = []
model.train()
for epoch in range(1, params['num_epochs'] + 1):
    print(f'========== epoch {epoch} start ==========')
    epoch_loss = 0.0
    for step, (x, y, l) in enumerate(dataloader):
        print(f'step: {step}')
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        l = l.to(device, non_blocking=True)
        sl = utils.label_smoothing(l.cpu(), params['vocab_size'], params['smoothing_rate']).to(device)
        src_padding_mask = utils.create_padding_mask(x).unsqueeze(1).repeat(1, params['max_len'], 1)
        label_padding_mask = utils.create_padding_mask(l)
        optimizer.zero_grad()
        with autocast(device_type='cuda', dtype=torch.float16):
            output = model(x, y, src_padding_mask, tgt_causal_mask)
            loss = criterion(output, sl, label_padding_mask)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        lr_histories.append(scheduler.step())
        epoch_loss += loss.item()
    loss_histories.append(epoch_loss / params['batch_size'])

    # 每 10 个 epoch 保存一下中间结果
    if epoch % 10 == 0:
        torch.save(model.state_dict(), os.path.join('outputs', f'model_{epoch}.pth'))
        np.save(os.path.join('outputs', f'lr_histories_{epoch}.npy'), np.array(lr_histories))
        np.save(os.path.join('outputs', f'loss_histories_{epoch}.npy'), np.array(loss_histories))
        lr_histories = []
        loss_histories = []