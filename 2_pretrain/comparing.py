from CHET import CheTEncoder,ClassPred,RegressPred,CheTModel,load_data,_get_batch_loss_bert,training,validate,load_csv
from d2l import torch as d2l
import matplotlib.pyplot as plt
import numpy as np
import csv

devices = d2l.try_all_gpus()
df = load_csv('new_file.csv')

batch_size = 128
test_size = 0.001
seed = 0
train_iter,val_iter = load_data(df, test_size,batch_size,seed = 0)

vocab_size = 210
num_hiddens = 32
ffn_num_hiddens = 40
num_heads = 4
norm_shape = [32]
ffn_num_input = 32
num_layers = 12
dropout = 0
num_embed = 32
lr = 0.005
step_per_val = 20
total_step = 20000

def run(vocab_size,num_hiddens,ffn_num_hiddens,num_heads,norm_shape,ffn_num_input,num_layers,dropout,num_embed,lr,step_per_val,total_step):
    encoder = CheTEncoder(vocab_size,num_hiddens, norm_shape, ffn_num_input,
                          ffn_num_hiddens, num_heads, num_layers, dropout)

    net = CheTModel(vocab_size, num_hiddens, norm_shape, ffn_num_input,
                     ffn_num_hiddens, num_heads, num_layers, dropout,
                     max_len=10, key_size=num_embed, query_size=num_embed, value_size=num_embed,
                     hid_in_features=num_embed, mlm_in_features=num_embed,
                     sgp_in_features=num_embed)



    trainlog,vallog = training(train_iter,val_iter, net, devices, total_step, step_per_val,lr=lr)

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

    plt.savefig(f'picture/{total_step}_{last_tloss}_{last_vloss}.png')

    data = {
        'seed':seed,
        'vocab_size': vocab_size,
        'num_hiddens': num_hiddens,
        'ffn_num_hiddens': ffn_num_hiddens,
        'num_heads': num_heads,
        'norm_shape': norm_shape,
        'ffn_num_input': ffn_num_input,
        'num_layers': num_layers,
        'dropout': dropout,
        'batch_size': batch_size,
        'test_size': test_size,
        'step_per_val': step_per_val,
        'total_step': total_step,
        'num_embed': num_embed,
        'lr': lr,
        'last_tloss': last_tloss,
        'last_vloss': last_vloss
    }

    # 指定你要写入的csv文件名
    csv_file = "training_log.csv"

    # 检查文件是否已存在
    try:
        with open(csv_file, 'x', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=data.keys())
            writer.writeheader()
    except FileExistsError:
        pass

    # 将数据写入csv文件
    with open(csv_file, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data.keys())
        writer.writerow(data)

for i in range(12):
    exp = 1 - i*0.5
    lr = 10**(exp)
    print(f'lr={lr}')
    run(vocab_size,num_hiddens,ffn_num_hiddens,num_heads,norm_shape,ffn_num_input,num_layers,dropout,num_embed,lr,step_per_val,total_step)