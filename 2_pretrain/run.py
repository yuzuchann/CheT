from CHET import CheTEncoder,ClassPred,RegressPred,CheTModel,load_data,_get_batch_loss_bert,training,validate,load_csv
from d2l import torch as d2l
import matplotlib.pyplot as plt
import numpy as np

devices = d2l.try_all_gpus()

df = load_csv('new_file.csv')

vocab_size = 210
num_hiddens = 32
ffn_num_hiddens = 40
num_heads = 4
norm_shape = [32]
ffn_num_input = 32
num_layers = 12
dropout = 0

batch_size = 128
test_size = 0.001
step_per_val = 20
total_step = 1000
num_embed = 32
step_per_val = 20
total_step = 20000

encoder = CheTEncoder(vocab_size,num_hiddens, norm_shape, ffn_num_input,
                      ffn_num_hiddens, num_heads, num_layers, dropout)

net = CheTModel(vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=10, key_size=num_embed, query_size=num_embed, value_size=num_embed,
                 hid_in_features=num_embed, mlm_in_features=num_embed,
                 sgp_in_features=num_embed)

train_iter,val_iter = load_data(df, test_size,batch_size)

trainlog,vallog = training(train_iter,val_iter, net, devices, total_step, step_per_val)

trainlog,vallog
train_losses,val_losses = np.array(trainlog),np.array(vallog)
list_tasks = ['sgp','bgp','sp']

plt.figure()
colors = plt.cm.get_cmap('prism', len(list_tasks))
for i, task in enumerate(list_tasks):
    plt.plot(train_losses[:, i], label=f'Train {task}', color=colors(i))
for i, task in enumerate(list_tasks):
    plt.plot(np.arange(0, train_losses.shape[0], step_per_val), val_losses[:, i], label=f'Val {task}', linestyle='--', color=colors(i))
plt.legend()
plt.title('Training and Validation Losses')
plt.xlabel('Step')
plt.ylabel('Loss')

last_tloss = sum(train_losses[-1])
last_vloss = sum(val_losses[-1])

plt.savefig(f'{total_step}_{last_tloss}_{last_vloss}.png')