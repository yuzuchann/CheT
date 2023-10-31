import torch
from torch import nn
from d2l import torch as d2l
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import re

def _get_batch_loss_bert(net, loss, vocab_size, tokens_X, pred_positions_X, mlm_Y, portions_X, valid_lens_x, sgp_y, sgp_weight):
    # 前向传播
    _, mlm_Y_hat, sgp_Y_hat = net(tokens_X, portions_X,
                                  valid_lens_x.reshape(-1),
                                  pred_positions_X)
    # 计算遮蔽语言模型损失
    mlm_l = loss(mlm_Y_hat.reshape(-1, vocab_size), mlm_Y.reshape(-1))
    mlm_l = mlm_l.sum()
    # 计算下一句子预测任务的损失
    #weights = torch.tensor([0.62, 0.38])
    #weights = weights.to(devices[0])
    #loss2 = nn.CrossEntropyLoss(weight=weights)
    sgp_l = loss(sgp_Y_hat, sgp_y)
    l = mlm_l + sgp_l*sgp_weight
    return mlm_l, sgp_l, l

def training(train_iter, net, loss, vocab_size, devices, num_steps,sgp_weight):
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    trainer = torch.optim.Adam(net.parameters(), lr=0.2)
    step, timer = 0, d2l.Timer()
    animator = d2l.Animator(xlabel='step', ylabel='loss',
                            xlim=[1, num_steps], legend=['mlm', 'sgp'])
    # 遮蔽语言模型损失的和，下一句预测任务损失的和，句子对的数量，计数
    metric = d2l.Accumulator(4)
    num_steps_reached = False
    while step < num_steps and not num_steps_reached:
        #all_element_ids,all_pred_positions,all_mlm_labels,portions,valid_lens,spacegroup
        for tokens_X, pred_positions_X, mlm_Y, portions_X, valid_lens_x, sgp_y in train_iter:
            tokens_X = tokens_X.to(devices[0])
            portions_X = portions_X.to(devices[0])
            valid_lens_x = valid_lens_x.to(devices[0])
            pred_positions_X = pred_positions_X.to(devices[0])
            mlm_Y, sgp_y = mlm_Y.to(devices[0]), sgp_y.to(devices[0])
            trainer.zero_grad()
            timer.start()
            mlm_l, nsp_l, l = _get_batch_loss_bert(
                net, loss, vocab_size, tokens_X, pred_positions_X, mlm_Y, portions_X, valid_lens_x, sgp_y,sgp_weight)
            l.backward()
            trainer.step()
            metric.add(mlm_l, nsp_l, tokens_X.shape[0], 1)
            timer.stop()
            animator.add(step + 1,
                         (metric[0] / metric[3], metric[1] / metric[3]))
            step += 1
            if step == num_steps:
                num_steps_reached = True
                break

    print(f'MLM loss {metric[0] / metric[3]:.3f}, '
          f'SGP loss {metric[1] / metric[3]:.3f}')
    print(f'{metric[2] / timer.sum():.1f} samples/sec on '
          f'{str(devices)}')

# 将 DataFrame 中的数据转换为张量
def df2tensor(batch_size,df):    
    all_element_ids = torch.tensor(df['all_element_ids'].values.tolist())
    all_pred_positions = torch.tensor(df['all_pred_positions'].values.tolist())
    all_mlm_labels = torch.tensor(df['all_mlm_labels'].values.tolist())
    portions = torch.tensor(df['portions'].values.tolist())
    valid_lens = torch.tensor(df['valid_lens'].values.tolist())
    spacegroup = torch.tensor(df['spacegroup'].values.tolist())

    # 使用 TensorDataset 创建数据集
    dataset = TensorDataset(all_element_ids,all_pred_positions,all_mlm_labels,portions,valid_lens,spacegroup)

    # 使用 DataLoader 加载数据
    train_iter = DataLoader(dataset, batch_size=batch_size)
    return train_iter