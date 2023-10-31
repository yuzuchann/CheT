import torch
from torch import nn
from d2l import torch as d2l

class MaskLM(nn.Module): # 定义一个MaskLM类，继承自nn.Module
    """BERT的掩蔽语言模型任务""" # 类的文档字符串
    def __init__(self, vocab_size, num_hiddens, num_inputs=32, **kwargs): # 初始化方法，接受词汇表大小，隐藏层大小和输入大小作为参数
        super(MaskLM, self).__init__(**kwargs) # 调用父类的初始化方法
        self.mlp = nn.Sequential(nn.Linear(num_inputs, num_hiddens), # 定义一个多层感知器，包括线性层，ReLU激活函数，层归一化和线性层
                                 nn.ReLU(),
                                 nn.LayerNorm(num_hiddens),
                                 nn.Linear(num_hiddens, vocab_size))

    def forward(self, X, pred_positions): # 前向传播方法，接受输入X和预测位置pred_positions作为参数
        num_pred_positions = pred_positions.shape[1] # 获取预测位置的数量
        pred_positions = pred_positions.reshape(-1) # 将预测位置展平
        batch_size = X.shape[0] # 获取批量大小
        batch_idx = torch.arange(0, batch_size) # 创建一个从0到batch_size-1的整数序列
        # 假设batch_size=2，num_pred_positions=3
        # 那么batch_idx是np.array（[0,0,0,1,1,1]）
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions) # 重复batch_idx中的每个元素num_pred_positions次
        masked_X = X[batch_idx, pred_positions] # 获取掩蔽位置的输入值
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1)) # 将masked_X重塑为(batch_size, num_pred_positions, -1)形状
        mlm_Y_hat = self.mlp(masked_X) # 将masked_X输入到多层感知器中，得到预测结果mlm_Y_hat
        return mlm_Y_hat # 返回预测结果mlm_Y_hat