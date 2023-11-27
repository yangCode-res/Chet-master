import os
import pickle
import random

import torch
import numpy as np

from models.model2 import Model
from utils import load_adj, EHRDataset, load_cate__parent_dict
from metrics2 import evaluate_codes, evaluate_hf
from utils import load_cate_adj, load_cate_dict
def historical_hot(code_x, code_num, lens):
    result = np.zeros((len(code_x), code_num), dtype=int)
    for i, (x, l) in enumerate(zip(code_x, lens)):
        result[i] = x[l - 1]
    return result
state_path='27'
state_dict=torch.load('./data/params/mimic3/m/{}.pt'.format(state_path))
print(state_path)
if __name__ == '__main__':
    seed = 6669
    dataset = 'mimic3'  # 'mimic3' or 'eicu'
    task = 'm'  # 'm' or 'h' or'e'
    use_cuda = True
    device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')
    print(device)
    code_size = 48  # mn的嵌入大小s
    graph_size = 32  # R的嵌入大小32
    hidden_size = 150  # rnn hidden size
    t_attention_size = 32
    t_output_size = hidden_size
    batch_size = 32
    epochs = 300
    torch.cuda.current_device()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    dataset_path = os.path.join('data', dataset, 'standard')
    train_path = os.path.join(dataset_path, 'train')
    valid_path = os.path.join(dataset_path, 'valid')
    test_path = os.path.join(dataset_path, 'test')

    code_adj = load_adj(dataset_path, device=device)
    cate_adj = load_cate_adj(dataset_path, device=device)
    cate_dict = load_cate_dict(dataset_path, dataset)
    parent_dict = load_cate__parent_dict(dataset_path, dataset)
    code_num = len(code_adj)
    print('this is code num', code_num)
    print('loading train data ...')
    train_data = EHRDataset(train_path, label=task, batch_size=batch_size, shuffle=True, device=device)
    print('loading valid data ...')
    valid_data = EHRDataset(valid_path, label=task, batch_size=batch_size, shuffle=False, device=device)
    print('loading test data ...')
    test_data = EHRDataset(test_path, label=task, batch_size=batch_size, shuffle=False, device=device)
    print(test_data)
    valid_historical = historical_hot(valid_data.code_x, code_num, valid_data.visit_lens)
    test_historical = historical_hot(test_data.code_x, code_num, test_data.visit_lens)
    #test_historical = historical_hot(train_data.code_x, code_num, train_data.visit_lens)

    task_conf = {
        'm': {
            'dropout': 0.45,
            'output_size': code_num,
            'evaluate_fn': evaluate_codes,
            'lr': {
                'init_lr': 0.01,
                'milestones': [20, 30],
                'lrs': [1e-3, 1e-4]
            }
        },
        'h': {
            'dropout': 0.0,
            'output_size': 1,
            'evaluate_fn': evaluate_hf,
            'lr': {
                'init_lr': 0.01,
                'milestones': [2, 3, 20],
                'lrs': [1e-3, 1e-4, 1e-5]
            }
        },
        'e': {
            'dropout': 0.0,
            'output_size': 1,
            'evaluate_fn': evaluate_hf,
            'lr': {
                'init_lr': 0.01,
                'milestones': [2, 3, 20],
                'lrs': [1e-3, 1e-4, 1e-5]
            }
        }
    }
    output_size = task_conf[task]['output_size']
    activation = torch.nn.Sigmoid()
    loss_fn = torch.nn.BCELoss()
    evaluate_fn = task_conf[task]['evaluate_fn']
    dropout_rate = task_conf[task]['dropout']
    param_path = os.path.join('data', 'params', dataset, task)
    if not os.path.exists(param_path):
        os.makedirs(param_path)

    model = Model(code_num=code_num, code_size=code_size,
                  adj=code_adj, graph_size=graph_size, hidden_size=hidden_size, t_attention_size=t_attention_size,
                  t_output_size=t_output_size, cate_adj=cate_adj,
                  output_size=output_size, dropout_rate=dropout_rate, activation=activation, cate_dict=cate_dict,
                  parent_dict=parent_dict).to(device)
    model.load_state_dict(state_dict)
    for name ,param in model.named_parameters():
        print(name)
    print(model.parameters())
    # pickle.dump(model.embedding_layer.c_embeddings,open('c_embeddings.pkl','wb'))
    # pickle.dump(model.embedding_layer.cate_embeddings,open('cate_embeddings.pkl','wb'))
    valid_loss, f1_score = evaluate_fn(model, test_data, loss_fn, output_size, test_historical)