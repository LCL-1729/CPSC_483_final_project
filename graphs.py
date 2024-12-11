from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.loader import DataLoader as GeoDataLoader
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from ogb.graphproppred import Evaluator
from sklearn.preprocessing import scale
from sklearn.metrics import f1_score
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

russell_file = './content/Processed_RUSSELL.csv'
sp_file = './content/Processed_S&P.csv'
dji_file = './content/Processed_DJI.csv'
nasdaq_file = './content/Processed_NASDAQ.csv'
nyse_file = './content/Processed_NYSE.csv'

def load_data(file_path):
    return pd.read_csv(file_path, index_col='Date')

data_russell = load_data(russell_file)
data_sp = load_data(sp_file)
data_dji = load_data(dji_file)
data_nasdaq = load_data(nasdaq_file)
data_nyse = load_data(nyse_file)

cols_to_keep = [
    'Close', 'Volume', 'mom', 'mom1', 'mom2', 'mom3', 
    'ROC_5', 'ROC_10', 'ROC_15', 'ROC_20', 'EMA_10', 'EMA_20', 'EMA_50', 'EMA_200'
]

def merge_dataframes(dfs):
    for i, df_ in enumerate(dfs):
        for col in df_.columns:
            if col not in cols_to_keep:
                del df_[col]
        df_.rename(columns={"Close": f"Close{i+1}"}, inplace=True)

merge_dataframes([data_sp, data_dji, data_nasdaq, data_nyse])
merged_df = pd.concat([data_russell, data_sp, data_dji, data_nasdaq, data_nyse], axis=1)
del merged_df['Name']

predict_day = 1
split_date = '2016-04-21'
seq_len = 60

target = pd.DataFrame()
true_target = pd.DataFrame()

target["Close"] = (merged_df['Close'][predict_day:] / merged_df['Close'][:-predict_day].values).astype(int)
for i in range(1,5):
    target[f"Close{i}"] = (merged_df[f"Close{i}"][predict_day:] / merged_df[f"Close{i}"][:-predict_day].values).astype(int)

true_target["Close"] = merged_df['Close'][predict_day:] / merged_df['Close'][:-predict_day].values
for i in range(1,5):
    true_target[f"Close{i}"] = merged_df[f"Close{i}"][predict_day:] / merged_df[f"Close{i}"][:-predict_day].values

merged_df = merged_df[:-predict_day]
target.index = merged_df.index
true_target.index = merged_df.index

merged_df = merged_df[200:]
target = target[200:]
true_target = true_target[200:]
merged_df = merged_df.fillna(0)

def scale_data(data_):
    for col in data_.columns:
        data_[col] = scale(data_[col])
    return data_

merged_df = scale_data(merged_df)

train_data_all = merged_df[merged_df.index < split_date]
train_target_all = target[target.index < split_date]
train_true_target_all = true_target[true_target.index < split_date]

train_data = train_data_all[:int(0.85 * train_data_all.shape[0])]
train_target = train_target_all[:int(0.85 * train_target_all.shape[0])]
train_true_target = train_true_target_all[:int(0.85 * train_target_all.shape[0])]

val_data = train_data_all[int(0.85 * train_data_all.shape[0]) - seq_len:]
val_target = train_target_all[int(0.85 * train_target_all.shape[0]) - seq_len:]
val_true_target = train_true_target_all[int(0.85 * train_target_all.shape[0]) - seq_len:]

test_data = merged_df[merged_df.index >= split_date]
test_target = target[target.index >= split_date]
test_true_target = true_target[true_target.index >= split_date]

all_data_arrays = [
    np.array(train_data), np.array(train_target), 
    np.array(test_data), np.array(test_target), 
    np.array(val_data), np.array(val_target)
]

def make_sequences(total_data, total_target, total_ret, data_, target_, ret_, length):
    for i in range(data_.shape[0] - length + 1):
        total_data.append(data_[i: i + length])
        total_target.append(target_[i + length - 1])
        total_ret.append(ret_[i + length - 1])
    return total_data, total_target, total_ret

def prepare_sequences(arrays, length):
    train_X, train_y, train_ret = [], [], []
    test_X, test_y, test_ret = [], [], []
    val_X, val_y, val_ret = [], [], []

    train_X, train_y, train_ret = make_sequences(train_X, train_y, train_ret, arrays[0], arrays[1], np.array(train_true_target), length)
    test_X, test_y, test_ret = make_sequences(test_X, test_y, test_ret, arrays[2], arrays[3], np.array(test_true_target), length)
    val_X, val_y, val_ret = make_sequences(val_X, val_y, val_ret, arrays[4], arrays[5], np.array(val_true_target), length)

    train_X = np.array(train_X).reshape(-1, length, merged_df.shape[1], 1)
    test_X = np.array(test_X).reshape(-1, length, merged_df.shape[1], 1)
    val_X = np.array(val_X).reshape(-1, length, merged_df.shape[1], 1)

    return train_X, np.array(train_y), test_X, np.array(test_y), val_X, np.array(val_y), np.array(train_ret), np.array(val_ret), np.array(test_ret)

train_data_seq, train_target_seq, test_data_seq, test_target_seq, val_data_seq, val_target_seq, train_return_seq, val_return_seq, test_return_seq = prepare_sequences(all_data_arrays, seq_len)

return_true_test = test_return_seq - 1
return_true_val = val_return_seq - 1
return_true_train = train_return_seq - 1

combined = np.append(train_return_seq, val_return_seq, axis=0)
for i in range(5):
    combined[:,i].sort()

low_thresh = combined[int(0.35*combined.shape[0])]
mid_thresh = combined[int(0.5*combined.shape[0])]
high_thresh = combined[int(0.65*combined.shape[0])]

plt.hist(combined, alpha=0.7, label='Combined')
plt.hist(test_return_seq, alpha=0.7, label='Test')
plt.xlabel("Return")
plt.ylabel("Frequency")
plt.title("Distribution of Returns")
plt.legend()
plt.savefig("./returns_distribution.png", dpi=300, bbox_inches='tight')
plt.close()

def returns_to_classes(r):
    c1 = r > low_thresh
    c2 = r > high_thresh
    return (c1.astype(int) + c2.astype(int))

train_return_seq = returns_to_classes(train_return_seq)
val_return_seq = returns_to_classes(val_return_seq)
test_return_seq = returns_to_classes(test_return_seq)

def returns_to_one_hot(r):
    out = np.zeros((r.shape[0], 3*5))
    for i in range(r.shape[0]):
        for k in range(5):
            out[i,k*3+r[i,k]] = 1
    return out

train_target_seq = returns_to_one_hot(train_return_seq)
val_target_seq = returns_to_one_hot(val_return_seq)
test_target_seq = returns_to_one_hot(test_return_seq)

dim_feature = merged_df.shape[1]
total_stock_index = 5

train_X = torch.Tensor(np.transpose(train_data_seq.reshape(-1, seq_len, dim_feature),(0,2,1)))
train_Y = torch.Tensor(train_target_seq.reshape(-1,3*total_stock_index))
train_loader = DataLoader(TensorDataset(train_X, train_Y), batch_size=64, shuffle=True)

val_X = torch.Tensor(np.transpose(val_data_seq.reshape(-1, seq_len, dim_feature),(0,2,1)))
val_Y = torch.Tensor(val_target_seq.reshape(-1,3*total_stock_index))
val_loader = DataLoader(TensorDataset(val_X,val_Y), batch_size=32, shuffle=False)

test_X = torch.Tensor(test_data_seq.reshape(-1, seq_len, dim_feature))
test_Y = torch.Tensor(test_target_seq.reshape(-1,3*total_stock_index))
test_loader = DataLoader(TensorDataset(test_X,test_Y), batch_size=32, shuffle=False)

cov_mat = np.corrcoef(all_data_arrays[0].T)
threshold_edge = 0.6
total_edge = np.sum(cov_mat > threshold_edge)
graph_info = np.zeros((total_edge,2))
c=0
for i in range(dim_feature):
    for j in range(dim_feature):
        if cov_mat[i,j]>threshold_edge:
            graph_info[c]=[i,j]
            c+=1

def form_graph_rep(size_batch):
    rep = graph_info.reshape(-1,2)
    for i in range(size_batch-1):
        rep = np.append(rep,graph_info.reshape(-1,2)+dim_feature*(i+1),axis=0)
    return np.transpose(rep,(1,0))

class GAT(nn.Module):
    def __init__(self,
                 hidden_dim:list,
                 mid_dim:list,
                 num_layers:int,
                 dropout:float,
                 num_stock:int,
                 conv_state_1: bool=False,
                 conv_state_2: bool=False,
                 gat_state:str='arma',
                 filter_time=[8,1,8,3,1],
                 pooling_state="cnn"):
        super(GAT, self).__init__()
        self.mid_dim = mid_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_stock = num_stock
        self.conv_state_1 = conv_state_1
        self.conv_state_2 = conv_state_2
        self.gat_state = gat_state
        self.filter_time=filter_time
        self.pool_state=pooling_state
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for l in range(num_layers):
            if self.gat_state=='gat':
                layer = pyg_nn.GATv2Conv(in_channels=hidden_dim[l],
                                         out_channels=hidden_dim[l+1],
                                         add_self_loops=False,
                                         dropout=dropout)
            elif self.gat_state=='gcn':
                layer = pyg_nn.GCNConv(in_channels=hidden_dim[l],
                                       out_channels=hidden_dim[l+1],
                                       add_self_loops=False)
            elif self.gat_state=='deepgcn':
                pass
            elif self.gat_state=='arma':
                layer=pyg_nn.ARMAConv(in_channels=hidden_dim[l],
                                      out_channels=hidden_dim[l+1],
                                      dropout=dropout)
            self.convs.append(layer)
            self.bns.append(nn.BatchNorm1d(hidden_dim[l+1]))

        self.conv2d1=nn.Conv2d(in_channels=1,out_channels=self.filter_time[0],kernel_size=(1,5))
        self.conv2d2=nn.Conv2d(in_channels=self.filter_time[0],out_channels=self.filter_time[1],kernel_size=(1,5))
        self.conv2d3=nn.Conv2d(in_channels=1,out_channels=self.filter_time[2],kernel_size=(1,3))
        self.conv2d4=nn.Conv2d(in_channels=self.filter_time[2],out_channels=self.filter_time[3]*self.num_stock,kernel_size=(1,3))
        self.conv2d5=nn.Conv2d(in_channels=1,out_channels=self.filter_time[4],kernel_size=(dim_feature,1))
        self.pooling1=nn.MaxPool2d((1,2))
        self.pooling2=nn.MaxPool2d((1,2))
        self.fc1 = nn.Linear(hidden_dim[-2], self.num_stock*mid_dim[0])
        self.fc1_ = nn.Linear(hidden_dim[-1],mid_dim[0])
        self.drop1 = nn.Dropout(p=self.dropout)
        self.fc3 = nn.Linear(mid_dim[1], mid_dim[2])

    def forward(self, x, graph_data,batch) -> torch.Tensor:
        if self.conv_state_1:
            x=x.reshape(-1,1,dim_feature,seq_len)
            x = F.relu(self.conv2d1(x))
            if not self.conv_state_2:
              x= self.pooling1(x)
              x=F.relu(self.conv2d2(x))
              x= self.pooling2(x)
        x=x.reshape(-1,self.hidden_dim[0]*self.filter_time[1])
        graph_data=graph_data.to(torch.int64)
        batch=batch.to(torch.int64)

        for l, conv in enumerate(self.convs):
            x = conv(x, graph_data)
            if l != self.num_layers - 1:
                x = self.bns[l](x)
                x = F.relu(x)

        if self.pool_state=="cnn":
          x = x.reshape(-1,1,dim_feature,self.hidden_dim[-2])
          x = self.conv2d5(x)
          x = x.reshape(-1,self.hidden_dim[-2])
        if self.conv_state_2:
            x=x.reshape(-1,1,1,self.hidden_dim[-2])
            x=F.relu(self.conv2d3(x))
            x= self.pooling1(x)
            x=F.relu(self.conv2d4(x))
            x= self.pooling2(x)
            x=x.reshape(-1,self.num_stock,self.hidden_dim[-1])
            x = F.relu(self.fc1_(x))
        else:
            x = F.relu(self.fc1(x))
            x=x.reshape(-1,self.num_stock,self.mid_dim[0])
        x = self.drop1(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        x = x.flatten(start_dim=1)
        return x

def train(model, device, loader, optimizer, loss_fn, pbar, num_stock=5):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    pbar.reset(len(loader))
    pbar.set_description('Training')
    for step,(x_data,target) in enumerate(loader):
        batch_size=target.size(dim=0)
        graph_data=torch.Tensor(form_graph_rep(batch_size))
        batch = torch.Tensor(np.repeat(np.arange(batch_size),dim_feature))
        all_labels.append(target.detach())
        batch = batch.to(device)
        x_data = x_data.to(device)
        graph_data = graph_data.to(device)
        target = target.to(device)
        preds = model(x_data,graph_data,batch)
        all_preds.append(preds.detach().cpu())
        if num_stock==5:
          loss = loss_fn(preds.reshape(-1,5,3), target.reshape(-1,5,3))
        else:
          loss = loss_fn(preds, target.to(torch.float32))
        total_loss += loss.item() * batch_size
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.update()
    all_preds = (torch.cat(all_preds).view(-1) > 0).to(torch.int32)
    all_labels = torch.cat(all_labels).view(-1)
    acc = (all_preds == all_labels).sum() / len(all_labels)
    avg_loss = total_loss / len(all_labels)
    return avg_loss, acc

def evaluate_model(model, device, loader, evaluator, pbar, num_stock:int, f_1_state: bool=False):
    model.eval()
    y_true = []
    y_pred = []
    y_true_f1 = []
    pbar.reset(total=len(loader))
    pbar.set_description('Evaluating')
    test_total=0
    test_correct=0
    test_loss=0
    for step,(x_data,target) in enumerate(loader):
        batch_size=target.size(dim=0)
        graph_data=torch.Tensor(form_graph_rep(batch_size))
        batch = torch.Tensor(np.repeat(np.arange(batch_size),dim_feature))
        batch = batch.to(device)
        x_data = x_data.to(device)
        graph_data = graph_data.to(device)
        target = target.to(device)
        with torch.no_grad():
            pred = model(x_data,graph_data,batch).detach().cpu()
            target= target.detach().cpu()
            if num_stock==5:
              pred=pred.reshape(-1,5,3)
              target=target.reshape(-1,5,3)
            _, predicted = torch.max(pred, 2)
            _,labels = torch.max(target,2)
            _, predicted_f1 = torch.max(pred[:,:,(0,2)], 2)
            y_true.append(predicted.detach())
            y_pred.append(labels.detach())
            y_true_f1.append(predicted_f1.detach())
            test_total += labels.size(0)*num_stock
            test_correct += (predicted == labels).sum().item()
            if num_stock==5:
                loss = loss_fn(pred.reshape(-1,5,3).to(torch.float32), target.reshape(-1,5,3).to(torch.float32))
            else:
                loss = loss_fn(pred, labels.to(torch.float32))
            test_loss += loss.item()
        pbar.update()
    total_length= sum(y_.shape[0] for y_ in y_true_f1)
    predicted_f1_np = np.zeros((total_length,5))
    labels_np = np.zeros((total_length,5))
    idx=0
    for i in range(len(y_true_f1)):
      predicted_f1_np[idx:idx+y_true_f1[i].shape[0],:]=y_true_f1[i].numpy()
      labels_np[idx:idx+y_pred[i].shape[0],:]=y_pred[i].numpy()
      idx+=y_true_f1[i].shape[0]
    if f_1_state:
      f1_=np.zeros(5)
      labels=(0.5*labels_np).astype(int)
      for i in range(5):
          f1_[i]=f1_score(labels[:,i],predicted_f1_np[:,i],average='macro')
    acc = test_correct/test_total
    score= test_loss/len(loader)
    return acc, score if not f_1_state else f1_, y_true, y_pred

def sharpe(actions, actual_returns, labels, multi=True):
    trading_res=np.zeros_like(actual_returns)
    count=0
    count_short=0
    count_long=0
    count_short_l=0
    count_long_l=0
    if multi:
        for i in range(len(actions)):
            for j in range(len(actions[i])):
                for k in range(5):
                    if labels[i][j,k]==0:
                      count_short_l+=1
                    if labels[i][j,k]==2:
                      count_long_l+=1
                    if actions[i][j,k]==0:
                      count_short+=1
                      trading_res[count,k]=-actual_returns[count,k]
                    if actions[i][j,k]==2:
                      count_long+=1
                      trading_res[count,k]=actual_returns[count,k]
                count+=1
        sharpe_comb=np.sqrt(252)*np.mean(np.sum(trading_res,axis=1))/np.std(np.sum(trading_res,axis=1))
        sharpe_1=np.sqrt(252)*np.mean(trading_res,axis=0)/np.std(trading_res,axis=0)
        ceq=np.mean(np.sum(trading_res,axis=1))-0.5*(np.std(np.sum(trading_res,axis=1))**2)
        ceq_1=np.mean(trading_res,axis=0)-0.5*(np.std(trading_res,axis=0)**2)
        long_sharpe=np.sqrt(252)*np.mean(np.sum(actual_returns,axis=1))/np.std(np.sum(actual_returns,axis=1))
    else:
        sharpe_comb=np.sqrt(252)*np.mean(trading_res)/np.std(trading_res)
        long_sharpe=np.sqrt(252)*np.mean(actual_returns)/np.std(actual_returns)
        sharpe_1=0
        ceq=0
        ceq_1=0
    return count_short,count_long,count_short_l,count_long_l,sharpe_comb,sharpe_1,long_sharpe,ceq,ceq_1

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gat_state", type=str, default="arma", choices=["gat", "gcn", "deepgcn", "arma"], help="Type of graph convolution to use.")
args = parser.parse_args()

model = GAT(
    hidden_dim=[seq_len-48,12,12,12,12,12,12],
    num_layers=0,
    mid_dim=[20,20,3],
    dropout=0.5,
    num_stock=5,
    conv_state_1=True,
    conv_state_2=False,
    gat_state=args.gat_state,
    pooling_state="cnn"
).to(device)

evaluator = Evaluator(name='ogbg-molhiv')
weights = [0.4,0.1,0.5]
class_weights = torch.FloatTensor(weights).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
loss_fn = nn.CrossEntropyLoss()

num_epochs = 300
train_losses = []
train_sharpe_list = []
val_sharpe_list = []
test_sharpe_list = []
ceq_list = []
ceq_list_ind=np.zeros((num_epochs,5))
sharpe_list_ind=np.zeros((num_epochs,5))
score_arr = np.zeros((num_epochs,5))
pbar = tqdm()

for epoch in range(1, 1 + num_epochs):
    loss, acc = train(model, device, train_loader, optimizer, loss_fn, pbar,num_stock=5)
    acc_train, score_train,y_true,y_pred = evaluate_model(model, device, train_loader, evaluator, pbar,num_stock=5,f_1_state=True)
    _,_,_,_,train_sharpe,_,_,_,_=sharpe(y_true,return_true_train,y_pred,True)
    acc_val,score_val,y_true,y_pred = evaluate_model(model, device, val_loader, evaluator, pbar,num_stock=5,f_1_state=True)
    _1,_2,_,_,val_sharpe,val_1,all_v_sharpe,ceq_0_v,ceq_1_v=sharpe(y_true,return_true_val,y_pred,True)
    acc_test,score_test,y_true,y_pred = evaluate_model(model, device, test_loader, evaluator, pbar,num_stock=5,f_1_state=True)
    _3,_4,_,_,test_sharpe,test_1,all_t_sharpe,ceq_0_t,ceq_1_t=sharpe(y_true,return_true_test,y_pred,True)
    train_sharpe_list.append(train_sharpe)
    val_sharpe_list.append(val_sharpe)
    test_sharpe_list.append(test_sharpe)
    ceq_list.append(ceq_0_t)
    ceq_list_ind[epoch-1,:]=ceq_1_t
    sharpe_list_ind[epoch-1,:]=test_1
    score_arr[epoch-1,:]=score_test

epochs = np.arange(1, num_epochs + 1)
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_sharpe_list, label='Train Sharpe', marker='o')
plt.plot(epochs, test_sharpe_list, label='Test Sharpe', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Sharpe Ratio')
plt.title(f'Train and Test Sharpe Ratios for {args.gat_state} Over Epochs')
plt.grid(True)
plt.legend()
plt.savefig('./sharpe_ratios.png')
plt.close()
