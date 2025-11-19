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
with open("config.yaml", "r") as f:
    params = yaml.safe_load(f)



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
src_input_path = os.path.join('processed_data', 'src_input.npy')
tgt_input_path = os.path.join('processed_data', 'tgt_input.npy')
label_path = os.path.join('processed_data', 'label.npy')
dataset = LazyDataset(src_input_path, tgt_input_path, label_path)
dataloader = DataLoader(
    dataset,
    batch_size=params['batch_size'],
    shuffle=True, 
    #num_workers=params['num_workers'],
    pin_memory=True,
    #persistent_workers=True,
    drop_last=True
)



# 获取可用设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# 定义模型、损失函数、优化器、学习率调度器
vocab_size_path = os.path.join('processed_data', 'vocab_size.npy')
vocab_size = np.load(vocab_size_path)
model = Transformer(
    params['num_encoder_blocks'],
    params['num_decoder_blocks'],
    vocab_size[0],
    vocab_size[1],
    params['model_size'],
    params['dropout'],
    params['num_heads'],
    params['ffn_size'],
    device
)
criterion = MaskedCrossEntropyLoss()
optimizer = Adam(
    model.parameters(),
    params['learning_rate'],
    (params['beta1'], params['beta2']),
    params['eps']
)
scheduler = LRScheduler(optimizer, params['warmup_step'], params['model_size'])



# 解码器输入的 causal mask
tgt_causal_mask = utils.create_causal_mask(params['max_len']).unsqueeze(0).repeat(params['batch_size'], 1, 1)



# 指定使用可用设备进行模型训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
tgt_causal_mask = tgt_causal_mask.to(device, non_blocking=True)



# 创建文件夹用于保存模型结果
os.makedirs('outputs', exist_ok=True)



# 设计训练过程
scaler = GradScaler()
lr_histories = []
loss_histories = []
model.train()
for epoch in range(1, params['num_epochs'] + 1):
    epoch_loss = 0.0
    for step, (x, y, l) in enumerate(dataloader):
        print(step)
        print(x.shape)
        print(y.shape)
        print(l.shape)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        l = l.to(device, non_blocking=True)
        sl = utils.label_smoothing(l.cpu(), vocab_size[1], params['smoothing_rate']).to(device)
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
        print(f'成功完成第 {step} 步优化')
    loss_histories.append(epoch_loss / params['batch_size'])
    print(f'成功完成第 {epoch} 轮优化')

    # 每 10 个 epoch 保存一下中间结果
    if epoch % 10 == 0:
        torch.save(model.state_dict(), os.path.join('outputs', f'model_{epoch}.pth'))
        np.save(os.path.join('outputs', f'lr_histories_{epoch}.npy'), np.array(lr_histories))
        np.save(os.path.join('outputs', f'loss_histories_{epoch}.npy'), np.array(loss_histories))
        lr_histories = []
        loss_histories = []