import torch
from torch import nn
from d2l import torch as d2l
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import ast

class CheTEncoder(nn.Module):
    """CheT Encoder."""
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout=0.2,
                 max_len=100, key_size=32, query_size=32, value_size=32,
                 **kwargs):
        super(CheTEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.element_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.proportion_embedding = nn.Linear(1, num_hiddens)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f"{i}", d2l.EncoderBlock(
                key_size, query_size, value_size, num_hiddens, norm_shape,
                ffn_num_input, ffn_num_hiddens, num_heads, dropout, True))

    def forward(self, element_numbers, proportions, valid_lens):
        original_shape = proportions.shape
        proportions = proportions.view(-1, 1)
        X = self.proportion_embedding(proportions)
        X = X.view(*original_shape[:-1], -1,self.num_hiddens)
        X += self.element_embedding(element_numbers)
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X

class ClassPred(nn.Module):
    def __init__(self, num_inputs, num_hiddens, num_class, **kwargs):
        super(ClassPred, self).__init__(**kwargs)
        self.mlp = nn.Sequential(nn.Linear(num_inputs, num_hiddens), # 定义一个多层感知器，包括线性层，ReLU激活函数，层归一化和线性层
                                 nn.ReLU(),
                                 nn.LayerNorm(num_hiddens),
                                 nn.Linear(num_hiddens, num_class))

    def forward(self, X):
        # X的形状：(batchsize,num_hiddens)
        return self.mlp(X)
    
class RegressPred(nn.Module):
    def __init__(self, num_inputs, num_hiddens, **kwargs):
        super(RegressPred, self).__init__(**kwargs)
        self.mlp = nn.Sequential(nn.Linear(num_inputs, num_hiddens), # 定义一个多层感知器，包括线性层，ReLU激活函数，层归一化和线性层
                                 nn.ReLU(),
                                 nn.LayerNorm(num_hiddens),
                                 nn.Linear(num_hiddens, 1))

    def forward(self, X):
        # X的形状：(batchsize,num_hiddens)
        return self.mlp(X)
    
class CheTModel(nn.Module):
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=11, key_size=32, query_size=32, value_size=32,
                 hid_in_features=32, mlm_in_features=32,
                 sgp_in_features=32):
        super(CheTModel, self).__init__()
        self.encoder = CheTEncoder(vocab_size, num_hiddens, norm_shape,
                    ffn_num_input, ffn_num_hiddens, num_heads, num_layers,
                    dropout, max_len=max_len, key_size=key_size,
                    query_size=query_size, value_size=value_size)
        self.hidden = nn.Sequential(nn.Linear(hid_in_features, num_hiddens),
                                    nn.Tanh())
        #self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)
        
        self.sgp = ClassPred(hid_in_features,500,240)
        self.bgp = RegressPred(hid_in_features,500)
        self.sp = RegressPred(hid_in_features,500)
        

    def forward(self, tokens, segments, valid_lens=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        sgp_label = self.sgp(self.hidden(encoded_X[:, 0, :]))
        bpp_label = self.bgp(self.hidden(encoded_X[:, 0, :]))
        sp_label = self.sp(self.hidden(encoded_X[:, 0, :]))
        
        return sgp_label, bpp_label, sp_label
    

def load_data(df, test_size,batch_size):

    # 划分训练集和验证集
    train_df, val_df = train_test_split(df, test_size=test_size)

    # 创建训练集
    train_element_ids = torch.tensor(train_df['atomic_numbers'].values.tolist())
    train_proportions = torch.tensor(train_df['proportions'].values.tolist())
    train_valid_lens = torch.tensor(train_df['valid_len'].values.tolist())
    train_spacegroup = torch.tensor(train_df['space_group_sym'].values.tolist())
    train_bandgaps = torch.tensor(train_df['Band_Gap'].values.tolist())
    train_stablility = torch.tensor(train_df['Stability'].values.tolist())

    train_dataset = TensorDataset(train_element_ids,train_proportions,train_valid_lens,train_spacegroup,train_bandgaps,train_stablility)
    train_iter = DataLoader(train_dataset, batch_size=batch_size)

    # 创建验证集
    val_element_ids = torch.tensor(val_df['atomic_numbers'].values.tolist())
    val_proportions = torch.tensor(val_df['proportions'].values.tolist())
    val_valid_lens = torch.tensor(val_df['valid_len'].values.tolist())
    val_spacegroup = torch.tensor(val_df['space_group_sym'].values.tolist())
    val_bandgaps = torch.tensor(val_df['Band_Gap'].values.tolist())
    val_stablility = torch.tensor(val_df['Stability'].values.tolist())

    val_dataset = TensorDataset(val_element_ids,val_proportions,val_valid_lens,val_spacegroup,val_bandgaps,val_stablility)
    val_iter = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_iter,val_iter

def _get_batch_loss_bert(net,element_ids,proportions,valid_lens,spacegroup,bandgaps,stablility):
    # 前向传播
    sgp_label, bpp_label, sp_label = net(element_ids,proportions,valid_lens=valid_lens.reshape(-1))
    
    # 计算遮蔽语言模型损失
    
    loss1 = nn.CrossEntropyLoss()
    loss2 = nn.MSELoss()
    
    sgp_l = loss1(sgp_label, spacegroup)#cls
    #print(sgp_label, spacegroup,bpp_label)
    bpp_l = loss2(bpp_label, bandgaps)#reg
    sp_l = loss2(sp_label, stablility)#reg
    
    l = sgp_l + bpp_l + sp_l
    
    return sgp_l,bpp_l,sp_l, l

def training(train_iter,val_iter, net, devices, num_steps,valstep,anime=False):
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    trainer = torch.optim.Adam(net.parameters(), lr=0.005)
    step, timer = 0, d2l.Timer()
    if anime == True:
        animator = d2l.Animator(xlabel='step', ylabel='loss', xlim=[1, num_steps],ylim=[0, 5], legend=['sgp', 'bpp','sp'])
    # 遮蔽语言模型损失的和，下一句预测任务损失的和，句子对的数量，计数
    metric = d2l.Accumulator(5)
    trainlog,vallog=[],[]
    num_steps_reached = False
    while step < num_steps and not num_steps_reached:
        #all_element_ids,all_pred_positions,all_mlm_labels,portions,valid_lens,spacegroup
        for element_ids,proportions,valid_lens,spacegroup,bandgaps,stablility in train_iter:
            element_ids = element_ids.to(devices[0])
            proportions = proportions.to(devices[0])
            valid_lens = valid_lens.to(devices[0])
            spacegroup = spacegroup.to(devices[0])
            bandgaps = bandgaps.to(devices[0])
            stablility = stablility.to(devices[0])
            trainer.zero_grad()
            timer.start()
            sgp_l,bpp_l,sp_l, l = _get_batch_loss_bert(
                net, element_ids,proportions,valid_lens,spacegroup,bandgaps,stablility)
            l.backward()
            trainer.step()
            metric.add(sgp_l,bpp_l,sp_l, element_ids.shape[0], 1)
            timer.stop()
            if anime == True:
                animator.add(step + 1,(metric[0] / metric[4], metric[1] / metric[4],metric[2] / metric[4]))
            elif step % (num_steps//20 )== 0:
                print(str(step)+'/'+str(num_steps))
            trainlog.append([metric[0] / metric[4], metric[1] / metric[4],metric[2] / metric[4]])
            if step % valstep == 0:
                vallog.append(validate(val_iter,net,devices))
            step += 1
            if step == num_steps:
                num_steps_reached = True
                break

    print(f'sgp loss {metric[0] / metric[4]:.3f}, '
          f'bpp loss {metric[1] / metric[4]:.3f}, '
          f'sp loss {metric[2] / metric[4]:.3f}')
    print(f'{metric[3] / timer.sum():.1f} samples/sec on '
          f'{str(devices)}')
    return trainlog,vallog

def validate(val_iter, net, devices):
    net.eval()  # 切换到评估模式
    metric = d2l.Accumulator(5)  # '验证损失的总和', '计数'
    for element_ids,proportions,valid_lens,spacegroup,bandgaps,stablility in val_iter:
        element_ids = element_ids.to(devices[0])
        proportions = proportions.to(devices[0])
        valid_lens = valid_lens.to(devices[0])
        spacegroup = spacegroup.to(devices[0])
        bandgaps = bandgaps.to(devices[0])
        stablility = stablility.to(devices[0])
        with torch.no_grad():
            sgp_l,bpp_l,sp_l, _ = _get_batch_loss_bert(
                net, element_ids,proportions,valid_lens,spacegroup,bandgaps,stablility)
        metric.add(sgp_l,bpp_l,sp_l, element_ids.shape[0],1)
    # 返回验证损失
    return metric[0] / metric[4],metric[1] / metric[4],metric[2] / metric[4]

def load_csv(filename):
    df = pd.read_csv(filename)
    df['atomic_numbers'] = df['atomic_numbers'].apply(ast.literal_eval)
    df['proportions'] = df['proportions'].apply(ast.literal_eval)
    return df