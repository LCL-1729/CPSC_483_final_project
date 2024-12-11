from __future__ import annotations

# from Python standard library
import copy
from pprint import pprint

# third-party libraries
import matplotlib.pyplot as plt
import networkx as nx
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric as pyg
import torch_geometric.nn as pyg_nn
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda' # if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

# DATA Processing

import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from os.path import join
from sklearn.metrics import accuracy_score as accuracy, f1_score, mean_absolute_error as mae
import os

file_fir='./content/Processed_RUSSELL.csv'
file_fir1='./content/Processed_S&P.csv'
file_fir2='./content/Processed_DJI.csv'
file_fir3='./content/Processed_NASDAQ.csv'
file_fir4='./content/Processed_NYSE.csv'

def load_data(file_fir):
    try:
        df_raw = pd.read_csv(file_fir, index_col='Date') # parse_dates=['Date'])
    except IOError:
        print("IO ERROR")
    return df_raw


'''
file_fir='/content/Processed_DJI.csv'
file_fir1='/content/Processed_DJI.csv'
file_fir2='/content/Processed_NASDAQ.csv'
file_fir3='/content/Processed_NASDAQ.csv'
file_fir4='/content/Processed_NASDAQ.csv'
'''
#df_raw = pd.read_csv(file_fir, index_col='Date') # parse_dates=['Date'])
data=load_data(file_fir)
data
data2=load_data(file_fir2)
data2
data1=load_data(file_fir1)
data1
data3=load_data(file_fir3)
data4=load_data(file_fir4)
diff_list=['Close', 'Volume', 'mom', 'mom1', 'mom2', 'mom3', 'ROC_5', 'ROC_10', 'ROC_15', 'ROC_20', 'EMA_10', 'EMA_20', 'EMA_50', 'EMA_200']
#join(file_fir, file_fir1, file_fir2)
def merge_df(df_list):
  for i in range(len(df_list)):
    for j in df_list[i].columns:
      if j not in diff_list:

        del df_list[i][j]
      df_list[i].rename(columns={"Close": "Close"+str(i+1)}, inplace=True)

merge_df([data1,data2,data3,data4])
print(data1)
print(data)
print(data.keys())
df=pd.concat([data, data1,data2,data3,data4], axis=1)
#data.join(data1, how='inner',on=index)

print(df)
print(df.columns)
print(df['Close'])
print(df["Close1"])
print(len(df.columns))
#data=df


data=df
del data['Name']

predict_day = 1
split_rate=0.85
target_split_day='2016-04-21'
seq_len = 60
target=pd.DataFrame()
true_target = pd.DataFrame()

target["Close"] = (data['Close'][predict_day:] / data['Close'][:-predict_day].values).astype(int)
target["Close1"]=(data['Close1'][predict_day:] / data['Close1'][:-predict_day].values).astype(int)
target["Close2"]=(data['Close2'][predict_day:] / data['Close2'][:-predict_day].values).astype(int)
target["Close3"]=(data['Close3'][predict_day:] / data['Close3'][:-predict_day].values).astype(int)
target["Close4"]=(data['Close4'][predict_day:] / data['Close4'][:-predict_day].values).astype(int)

true_target["Close"] = data['Close'][predict_day:] / data['Close'][:-predict_day].values
true_target["Close1"] = data['Close1'][predict_day:] / data['Close1'][:-predict_day].values
true_target["Close2"] = data['Close2'][predict_day:] / data['Close2'][:-predict_day].values
true_target["Close3"] = data['Close3'][predict_day:] / data['Close3'][:-predict_day].values
true_target["Close4"] = data['Close4'][predict_day:] / data['Close4'][:-predict_day].values

data = data[:-predict_day]
target.index = data.index
true_target.index = data.index

# Becasue of using 200 days Moving Average as one of the features
data = data[200:]
target = target[200:]
true_target = true_target[200:]
data = data.fillna(0)

print(target,true_target)
number_feature = data.shape[1]
samples_in_each_stock = data.shape[0]
# scale data
def scale_data(data_):
  for i in data_.columns:
    data_[i]=scale(data_[i])
  return data_
data=scale_data(data)





train_data1 = data[data.index < target_split_day ]
train_target1 = target[target.index < target_split_day]
train_true_target1 = true_target[true_target.index < target_split_day]

train_data = train_data1[:int(split_rate * train_data1.shape[0])]
train_target = train_target1[:int(split_rate* train_target1.shape[0])]
train_true_target = train_true_target1[:int(split_rate * train_target1.shape[0])]

valid_data = train_data1[int(split_rate * train_data1.shape[0]) - seq_len:]
valid_target = train_target1[int(split_rate * train_target1.shape[0]) - seq_len:]
valid_true_target = train_true_target1[int(split_rate * train_target1.shape[0]) - seq_len:]


test_data = data[data.index >= target_split_day]
test_target = target[target.index >= target_split_day]
test_true_target = true_target[true_target.index >= target_split_day]
print(test_true_target)
print(train_true_target)
print(valid_true_target)
a = [np.array(train_data), np.array(train_target), np.array(test_data), np.array(test_target), np.array(valid_data), np.array(valid_target)]

def cnn_data_sequence_separately(tottal_data, tottal_target, tottal_return, data, target, t_return, seque_len):
    for index in range(data.shape[0] - seque_len + 1):
        #print(index,data[index: index + seque_len])
        tottal_data.append(data[index: index + seque_len])
        #print(tottal_data)
        #print(index,target[ index + seque_len])

        tottal_target.append(target[index + seque_len - 1])
        #print(tottal_target)
        tottal_return.append(t_return[index + seque_len - 1])


    return tottal_data, tottal_target,tottal_return


def cnn_data_sequence(data_warehouse, seq_len):
    tottal_train_data = []
    tottal_train_target = []
    tottal_valid_data = []
    tottal_valid_target = []
    tottal_test_data = []
    tottal_test_target = []
    tottal_train_return = []
    tottal_valid_return = []
    tottal_test_return = []


    value=a
    tottal_train_data, tottal_train_target, tottal_train_return= cnn_data_sequence_separately(tottal_train_data, tottal_train_target,tottal_train_return,
                                                                          value[0], value[1],np.array(train_true_target), seq_len)

    tottal_test_data, tottal_test_target, tottal_test_return= cnn_data_sequence_separately(tottal_test_data, tottal_test_target, tottal_test_return,
                                                                        value[2], value[3],np.array(test_true_target), seq_len)
    tottal_valid_data, tottal_valid_target,tottal_valid_return = cnn_data_sequence_separately(tottal_valid_data, tottal_valid_target,tottal_valid_return,
                                                                          value[4], value[5],np.array(valid_true_target), seq_len)

    tottal_train_data = np.array(tottal_train_data)
    tottal_train_target = np.array(tottal_train_target)
    tottal_test_data = np.array(tottal_test_data)
    tottal_test_target = np.array(tottal_test_target)
    tottal_valid_data = np.array(tottal_valid_data)
    tottal_valid_target = np.array(tottal_valid_target)
    tottal_train_return = np.array(tottal_train_return)
    tottal_valid_return = np.array(tottal_valid_return)
    tottal_test_return = np.array(tottal_test_return)

    tottal_train_data = tottal_train_data.reshape(tottal_train_data.shape[0], tottal_train_data.shape[1],
                                                  tottal_train_data.shape[2], 1)
    tottal_test_data = tottal_test_data.reshape(tottal_test_data.shape[0], tottal_test_data.shape[1],
                                                tottal_test_data.shape[2], 1)
    tottal_valid_data = tottal_valid_data.reshape(tottal_valid_data.shape[0], tottal_valid_data.shape[1],
                                                  tottal_valid_data.shape[2], 1)

    return tottal_train_data, tottal_train_target, tottal_test_data, tottal_test_target, tottal_valid_data, tottal_valid_target,tottal_train_return,tottal_valid_return,tottal_test_return
cnn_train_data, cnn_train_target, cnn_test_data, cnn_test_target, cnn_valid_data, cnn_valid_target,cnn_train_return,cnn_valid_return,cnn_test_return = cnn_data_sequence(
            a, seq_len)

return_true_test=cnn_test_return-1
return_true_valid=cnn_valid_return-1
return_true_train=cnn_train_return-1

print(np.sum(return_true_test>0)/(338*5))
print(np.sum(return_true_test<0)/(338*5))

print(cnn_train_return.shape)
print(cnn_valid_return.shape)
print(cnn_test_return.shape)


c = np.append(cnn_train_return,cnn_valid_return,axis=0)
print(c.shape)
for i in range(5):
  c[:,i].sort()
print(c)
low_threshold=c[int(0.35*c.shape[0])]
low_threshold_=c[int(0.5*c.shape[0])]
high_threshold=c[int(0.65*c.shape[0])]
print(low_threshold,high_threshold,low_threshold_)

import matplotlib.pyplot as plt
print(plt.get_backend())
plt.hist(c, alpha=0.7, label='Combined Data')
plt.hist(cnn_test_return, alpha=0.7, label='Test Data')

plt.xlabel("Return Value")
plt.ylabel("Frequency")
plt.title("Distribution of Returns")

plt.legend()
plt.savefig("./returns_distribution.png", dpi=300, bbox_inches='tight')
plt.show()

def return_to_label(return_np):

  new_1 = return_np>low_threshold
  #new_1 = return_np>1.000000
  #new_2 = return_np>1.000
  new_2 = return_np>high_threshold
  new3=new_1.astype(int)+new_2.astype(int)
  return new3
cnn_train_return=return_to_label(cnn_train_return)
cnn_test_return=return_to_label(cnn_test_return)
cnn_valid_return=return_to_label(cnn_valid_return)

print(cnn_valid_return)

threshold_edge=0.6
print(cnn_train_data.shape,cnn_train_target.shape)
train_data_size=cnn_train_data.shape[0]
cov_mat=np.corrcoef(a[0].T)

# Graph formation

print(cov_mat.shape)
print(cnn_train_data.shape)
total_edge=np.sum(cov_mat>threshold_edge)
graph_info=np.ndarray(shape=(total_edge,2))
c=0
for i in range(number_feature):
  for j in range(number_feature):
    if cov_mat[i,j]>threshold_edge:
      graph_info[c]=[i,j]
      c+=1
#print(graph_info.shape)
#this also transpose
def form_graph_rep(size_batch):
    graph_info_rep=graph_info.reshape(-1,2)
    for i in range(size_batch-1):
      graph_info_rep=np.append(graph_info_rep,graph_info.reshape(-1,2)+number_feature*(i+1),axis=0)
    #print(graph_info_rep,graph_info_rep.shape)
    return np.transpose(graph_info_rep,(1,0))
print(form_graph_rep(3).shape)

print(cnn_train_data.shape)
print(cnn_train_target.shape)
print(cnn_test_data.shape)
print(cnn_test_target.shape)
print(cnn_valid_data.shape)
print(cnn_valid_target.shape)



dim_feature=cnn_train_data.shape[2]
print(dim_feature)
dim_target=cnn_train_target.shape[1]
print(dim_target)
tensor_x = torch.Tensor(np.transpose(cnn_train_data.reshape(-1,seq_len,dim_feature),(0,2,1))) # transform to torch tensor
#tensor_g = torch.Tensor(graph_info_rep)
tensor_y = torch.Tensor(cnn_train_target.reshape(-1,dim_target))
print(tensor_x.shape, tensor_y.shape)

my_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
my_dataloader = DataLoader(my_dataset, batch_size=32, shuffle=False)

tensor_x = torch.Tensor(np.transpose(cnn_valid_data.reshape(-1,seq_len,dim_feature),(0,2,1))) # transform to torch tensor
tensor_y = torch.Tensor(cnn_valid_target.reshape(-1,dim_target))

my_dataset_val = TensorDataset(tensor_x,tensor_y) # create your datset
my_dataloader_val = DataLoader(my_dataset_val, batch_size=32, shuffle=False)
tensor_x = torch.Tensor(cnn_test_data.reshape(-1,seq_len,dim_feature)) # transform to torch tensor
tensor_y = torch.Tensor(cnn_test_target.reshape(-1,dim_target))

my_dataset_test = TensorDataset(tensor_x,tensor_y) # create your datset
my_dataloader_test = DataLoader(my_dataset_test, batch_size=32, shuffle=False)

# Model

from torch.nn import LayerNorm, Linear, ReLU

class GAT(torch.nn.Module):
    def __init__(self,

                 hidden_dim: list, #hidden layer for gat/gcn channel
                 mid_dim:list, #fc layer
                 num_layers: int,
                 dropout: float,
                 num_stock: int,
                 conv_state_1: bool = False,
                 conv_state_2: bool = False,
                 gat_state = "gat",
                 filter_time=[8,1,8,3,1], #conc filter time
                 pooling_state="cnn"
                 ):
        """
        Args

        - hidden_dim: int, dimensions of hidden layers
        - num_layers: int, # of hidden layers
        - dropout: float, probability of dropout
        """
        super(GAT, self).__init__()

        # save all of the info
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



        # a list of GATv2 layers, with dropout
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for l in range(num_layers):
            if self.gat_state=='gat':
              layer = pyg_nn.GATv2Conv(in_channels=hidden_dim[l],
                                       out_channels=hidden_dim[l+1],add_self_loops=False,
                                       dropout=dropout)
            elif self.gat_state=='gcn':



              layer = pyg_nn.GCNConv(in_channels=hidden_dim[l],
                                     out_channels=hidden_dim[l+1],add_self_loops=False)
            elif self.gat_state=='deepgcn':
              conv = pyg_nn.GENConv(hidden_dim[l], hidden_dim[l+1], aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')
              norm = LayerNorm(hidden_dim[l+1])
              act = ReLU(inplace=True)

              layer =pyg_nn.DeepGCNLayer(conv, norm, act, block='res+', dropout=dropout,
                                  ckpt_grad=i % 3)
              #self.layers.append(layer)
            elif self.gat_state=='arma':
              layer=pyg_nn.ARMAConv(in_channels=hidden_dim[l],
                                     out_channels=hidden_dim[l+1],dropout=dropout)





            self.convs.append(layer)

            self.bns.append(nn.BatchNorm1d(hidden_dim[l+1]))

        # fully-connected final layer
        self.conv2d1=nn.Conv2d(in_channels=1,out_channels=self.filter_time[0],kernel_size=(1,5))
        self.conv2d2=nn.Conv2d(in_channels=self.filter_time[0],out_channels=self.filter_time[1],kernel_size=(1,5))
        self.conv2d3=nn.Conv2d(in_channels=1,out_channels=self.filter_time[2],kernel_size=(1,3))
        self.conv2d4=nn.Conv2d(in_channels=self.filter_time[2],out_channels=self.filter_time[3]*self.num_stock,kernel_size=(1,3))
        self.conv2d5=nn.Conv2d(in_channels=1,out_channels=self.filter_time[4],kernel_size=(number_feature,1))
        self.pooling1=nn.MaxPool2d((1,2))
        self.pooling2=nn.MaxPool2d((1,2))
        self.pooling3=nn.MaxPool2d((1,3))

        self.fc1 = nn.Linear(hidden_dim[-2], self.num_stock*mid_dim[0])
        #self.fc1 = nn.Linear(hidden_dim, 5*mid_dim[0])
        self.fc1_ = nn.Linear(hidden_dim[-1],mid_dim[0])
        self.fc2 = nn.Linear(mid_dim[0], mid_dim[1])
        self.drop1 = nn.Dropout(p=self.dropout)
        self.drop2 = nn.Dropout(p=self.dropout)
        self.fc3 = nn.Linear(mid_dim[1], mid_dim[2])


    def forward(self, x ,graph_data,batch) -> torch.Tensor:
        """
        Args
        - data: pyg.data.Batch, a batch of graphs

        Returns: torch.Tensor, shape [batch_size], unnormalized classification
            probability for each graph
        """
        #print(x.shape)
        if self.conv_state_1:
            x=x.reshape(-1,1,dim_feature,seq_len)


            x = self.conv2d1(x)
            x = F.relu(x)

            if not self.conv_state_2:
              x= self.pooling1(x)
              x=self.conv2d2(x)
              x = F.relu(x)

              x= self.pooling2(x)
             #here the dimension is batch*filter_time[1]* graph_node * hidden_dim[0](time) here to avoid shuffleing ,we set filtertime[1]=1


        x=x.reshape(-1,self.hidden_dim[0]*self.filter_time[1])



        graph_data=graph_data.to(torch.int64)
        batch=batch.to(torch.int64)
        #print(x.shape,'lall')

        for l, conv in enumerate(self.convs):
            #print(conv,x.dim(),x.shape,graph_data,graph_data.shape)
            #print(x,x.shape,edge_index,edge_index.shape,edge_attr,edge_attr.shape,batch)

            x = conv(x, graph_data)



            if l != self.num_layers - 1:

                x = self.bns[l](x)

                x = F.relu(x)
        if self.pool_state=="mean":

          x = pyg_nn.global_mean_pool(x, batch=batch)
        if self.pool_state=="max":

          x = pyg_nn.global_max_pool(x, batch=batch)
        #print(x.shape,'lll')
        if self.pool_state=="cnn":

          x = x.reshape(-1,1,number_feature,self.hidden_dim[-2])
          x = self.conv2d5(x)
          x = x.reshape(-1,self.hidden_dim[-2])

        #print(x.shape,'lal')

        if self.conv_state_2:

            x=x.reshape(-1,1,1,self.hidden_dim[-2])
            x=self.conv2d3(x)

            x = F.relu(x)
            x= self.pooling1(x)


            x=self.conv2d4(x)

            x = F.relu(x)
            x= self.pooling2(x)


            x=x.reshape(-1,self.num_stock,self.hidden_dim[-1])

            x = F.relu(self.fc1_(x))
            #print(x.shape)
        else:
            #print(x.shape,'not state2')
            x = F.relu(self.fc1(x))
            x=x.reshape(-1,self.num_stock,self.mid_dim[0])
            #print(x.shape,'k')

        x = self.drop1(x)
        '''
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        '''
        x = self.fc3(x)
        x = F.sigmoid(x)
        x = x.flatten(start_dim=1)
        #print(x.shape)
        return x


def train(model: nn.Module, device: torch.device | str,
          data_loader: DataLoader, optimizer: torch.optim.Optimizer,
          loss_fn: nn.Module, pbar: tqdm,num_stock=5) -> float:
    """Trains a GNN model.

    Args
    - model: nn.Module, GNN model, already placed on device
    - device: torch.device
    - data_loader: pyg.loader.DataLoader
    - optimizer: torch.optim.Optimizer
    - loss_fn: nn.Module

    Returns: (loss, acc)
    - loss: float, average per-graph loss across epoch
    - acc: float, accuracy
    """
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    pbar.reset(len(data_loader))
    pbar.set_description('Training')

    for step,(x_data,target) in enumerate(data_loader):

        batch_size=target.size(dim=0)
        graph_data=torch.Tensor(form_graph_rep(batch_size))
        ''''
        print(batch_size,x_data,target)
        print(number_feature)

        print(np.arange(batch_size))
        print(np.repeat(np.arange(batch_size),number_feature))
        '''
        batch = torch.Tensor(np.repeat(np.arange(batch_size),number_feature))
        all_labels.append(target.detach())
        #print(graph_data.shape,"graph")
        batch = batch.to(device)
        x_data = x_data.to(device)
        graph_data = graph_data.to(device)
        target = target.to(device)

        #batch_size = batch.batch.max().item()

        preds = model(x_data,graph_data,batch)
        #print(preds,preds.shape,target.shape,target)


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

def eval(model: nn.Module, device: torch.device, loader: DataLoader,
         evaluator: Evaluator, pbar: tqdm) -> dict[str, float]:
    """Evaluates the AUROC of a model on a dataset split.

    Args
    - model: nn.Module, GNN model, already moved to device
    - device: torch.device
    - loader: DataLoader
    - evaluator: Evaluator from OGB
    - pbar: tqdm, progress bar

    Returns: dict, with one (key, value)
    - key is dataset.eval_metric (which in this case is 'rocauc')
    - value is the AUROC
    """
    model.eval()
    y_true = []
    y_pred = []

    pbar.reset(total=len(loader))
    pbar.set_description('Evaluating')

    for step,(x_data,target) in enumerate(loader):
        y_true.append(target.detach())
        batch_size=target.size(dim=0)
        graph_data=torch.Tensor(form_graph_rep(batch_size))

        batch = torch.Tensor(np.repeat(np.arange(batch_size),number_feature))

        batch = batch.to(device)
        x_data = x_data.to(device)
        graph_data = graph_data.to(device)
        target = target.to(device)


        #batch_size = batch.batch.max().item()
        with torch.no_grad():

            pred = model(x_data,graph_data,batch)


        y_pred.append(pred.detach().cpu())
        pbar.update()


    all_preds = (torch.cat(y_pred).view(-1) > 0).to(torch.int32)
    all_labels = torch.cat(y_true).view(-1)
    acc = (all_preds == all_labels).sum() / len(all_labels)
    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    input_dict = {'y_true': y_true, 'y_pred': y_pred}
    return evaluator.eval(input_dict),acc

total_stock_index=5
def positional_encoding(return_l):
  data_size=return_l.shape[0]
  return_true=np.zeros(shape=[data_size,3*total_stock_index])
  for i in range(data_size):
    for j in range(5):
      return_true[i,j*3+return_l[i,j]]=1
  return return_true

cnn_train_target=positional_encoding(cnn_train_return)

cnn_valid_target=positional_encoding(cnn_valid_return)

cnn_test_target=positional_encoding(cnn_test_return)

print(cnn_train_data.shape, cnn_valid_data.shape, cnn_train_target.shape,cnn_test_target.shape,cnn_train_target.shape,cnn_valid_target.shape )

tensor_x = torch.Tensor(np.transpose(cnn_train_data.reshape(-1,seq_len,dim_feature),(0,2,1))) # transform to torch tensor
#tensor_g = torch.Tensor(graph_info_rep)
tensor_y = torch.Tensor(cnn_train_target.reshape(-1,3*total_stock_index))

my_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
my_dataloader = DataLoader(my_dataset, batch_size=64, shuffle=True)

tensor_x = torch.Tensor(np.transpose(cnn_valid_data.reshape(-1,seq_len,dim_feature),(0,2,1))) # transform to torch tensor
tensor_y = torch.Tensor(cnn_valid_target.reshape(-1,3*total_stock_index))

my_dataset_val = TensorDataset(tensor_x,tensor_y) # create your datset
my_dataloader_val = DataLoader(my_dataset_val, batch_size=32, shuffle=False)
tensor_x = torch.Tensor(cnn_test_data.reshape(-1,seq_len,dim_feature)) # transform to torch tensor
tensor_y = torch.Tensor(cnn_test_target.reshape(-1,3*total_stock_index))

my_dataset_test = TensorDataset(tensor_x,tensor_y) # create your datset
my_dataloader_test = DataLoader(my_dataset_test, batch_size=32, shuffle=False)


def sharpe(action, true_return,label,multi_state=False):
  res=0
  l=len(action)

  count,count_short,count_long,count_short_l,count_long_l=0,0,0,0,0
  trading_res=np.zeros(shape=true_return.shape)

  #print(true_return.shape)



  if multi_state:
      for i in range(l):
        for j in range(len(action[i])):
          for k in range(5):

              if label[i][j,k]==0:
                count_short_l+=1
              if label[i][j,k]==2:
                count_long_l+=1

              if action[i][j,k]==0:
                count_short+=1

                trading_res[count,k]=-true_return[count,k]
              if action[i][j,k]==2:
                count_long+=1

                trading_res[count,k]=+true_return[count,k]

          count+=1
  else:
        for i in range(l):
          for j in range(len(action[i])):
            if label[i][j]==0:
              count_short_l+=1
            if label[i][j]==2:
              count_long_l+=1

            if action[i][j]==0:
              count_short+=1

              trading_res[count]=-true_return[count]
            if action[i][j]==2:
              count_long+=1

              trading_res[count]=true_return[count]

            count+=1
  if multi_state:
      sharpe_comb=np.sqrt(252)*np.mean(np.sum(trading_res,axis=1))/np.std(np.sum(trading_res,axis=1))
      sharpe_1=np.sqrt(252)*np.mean(trading_res,axis=0)/np.std(trading_res,axis=0)
      ceq=np.mean(np.sum(trading_res,axis=1))-0.5*(np.std(np.sum(trading_res,axis=1))**2)
      ceq_1=np.mean(trading_res,axis=0)-0.5*(np.std(trading_res,axis=0)**2)
      long_sharpe=np.sqrt(252)*np.mean(np.sum(true_return,axis=1))/np.std(np.sum(true_return,axis=1))





  else:
      sharpe_comb=np.sqrt(252)*np.mean(trading_res)/np.std(trading_res)
      long_sharpe=np.sqrt(252)*np.mean(true_return)/np.std(true_return)
      sharpe_1=0
  #print(count,count_short,count_long,count_short_l,count_long_l)

  return count_short,count_long,count_short_l,count_long_l,sharpe_comb,sharpe_1 ,long_sharpe,ceq,ceq_1
def generate_random(label,true_return):
    res=0
    l=len(label)

    count,count_short,count_long,count_short_l,count_long_l=len(true_return),0,0,0,0
    for i in range(l):
      for j in range(len(label[i])):
        if label[i][j]==0:
          count_short_l+=1
        if label[i][j]==2:
          count_long_l+=1
    generated=np.random.choice(3,count, p=[0, 1-count_short_l/count-count_long_l/count, count_short_l/count+count_long_l/count])
    random_return=np.multiply((generated-1),true_return)


    return np.sqrt(252)*np.mean(random_return)/np.std(random_return)

from sklearn.metrics import f1_score

def eval_1(model: nn.Module, device: torch.device, loader: DataLoader,
         evaluator: Evaluator, pbar: tqdm,num_stock: int,f_1_state: bool=False) -> dict[str, float]:
    """Evaluates the AUROC of a model on a dataset split.

    Args
    - model: nn.Module, GNN model, already moved to device
    - device: torch.device
    - loader: DataLoader
    - evaluator: Evaluator from OGB
    - pbar: tqdm, progress bar

    Returns: dict, with one (key, value)
    - key is dataset.eval_metric (which in this case is 'rocauc')
    - value is the AUROC
    """
    model.eval()
    y_true = []
    y_pred = []
    y_true_f1 = []

    test_total=0
    test_correct=0
    test_loss=0

    pbar.reset(total=len(loader))
    pbar.set_description('Evaluating')
    total_length=0
    for step,(x_data,target) in enumerate(loader):
        #y_true.append(target.detach())
        batch_size=target.size(dim=0)
        total_length+=batch_size
        graph_data=torch.Tensor(form_graph_rep(batch_size))

        batch = torch.Tensor(np.repeat(np.arange(batch_size),number_feature))

        batch = batch.to(device)
        x_data = x_data.to(device)
        graph_data = graph_data.to(device)
        target = target.to(device)


        #batch_size = batch.batch.max().item()
        with torch.no_grad():

            pred = model(x_data,graph_data,batch).detach().cpu()
            target= target.detach().cpu()
            if num_stock==5:
              pred=pred.reshape(-1,5,3)
              target=target.reshape(-1,5,3)


            #print(pred,target,pred.shape,target.shape)
            _, predicted = torch.max(pred, 2)
            _,labels = torch.max(target,2)

            _, predicted_f1 = torch.max(pred[:,:,(0,2)], 2)
            y_true.append(predicted.detach())
            y_pred.append(labels.detach())
            y_true_f1.append( predicted_f1.detach())




            test_total += labels.size(0)*num_stock

            test_correct += (predicted == labels).sum().item()
            #print(pred.shape,target.shape,labels.shape)
            # find loss
            if num_stock==5:
                loss = loss_fn(pred.reshape(-1,5,3).to(torch.float32), target.reshape(-1,5,3).to(torch.float32))
            else:
                loss = loss_fn(pred, labels.to(torch.float32))


            test_loss += loss.item()



        pbar.update()
    predicted_f1_np=np.zeros(shape=(total_length,5))
    labels_np=np.zeros(shape=(total_length,5))
    #print(total_length)
    count=0
    for i in range(len(y_true)):
      predicted_f1_np[count:count+y_true_f1[i].shape[0],:]=y_true_f1[i].numpy()

      labels_np[count:count+y_pred[i].shape[0],:]=y_pred[i].numpy()
      count+=y_true_f1[i].shape[0]
    #print(count)

    if f_1_state:
      f1_=np.zeros(shape=(5))


      labels=(0.5*labels_np).astype(int)

      f1_[0]=f1_score(labels[:,0],predicted_f1_np[:,0],average='macro')
      f1_[1]=f1_score(labels[:,1],predicted_f1_np[:,1],average='macro')
      f1_[2]=f1_score(labels[:,2],predicted_f1_np[:,2],average='macro')
      f1_[3]=f1_score(labels[:,3],predicted_f1_np[:,3],average='macro')
      f1_[4]=f1_score(labels[:,4],predicted_f1_np[:,4],average='macro')


    #print(test_correct,test_total)
    acc = test_correct/test_total
    score= test_loss/len(loader)



    return acc, score if not f_1_state else f1_,y_true,y_pred


model = GAT(
            hidden_dim=[seq_len-48,12,12,12,12,12,12],
            num_layers=0,
            mid_dim=[20,20,3],
            dropout=0.5,
            num_stock=5,
            conv_state_1=True,
            conv_state_2=False,
            gat_state='arma',
            pooling_state="cnn")
model = model.to(device)

# use the official OGB evaluator, which will compute AUROC for us
evaluator = Evaluator(name='ogbg-molhiv')
weights = [0.4,0.1,0.5] #[ 1 / number of instances for each class]
class_weights = torch.FloatTensor(weights).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
#loss_fn = nn.CrossEntropyLoss(weight=class_weights)
loss_fn = nn.CrossEntropyLoss(weight=torch.as_tensor(np.array(weights)))
loss_fn = nn.CrossEntropyLoss()
num_epochs = 300
best_model = None
best_valid_auroc = 0

# track the train/valid AUROC and train loss across epochs of training

train_losses = []
train_sharpe_l = []
valid_sharpe_l = []
test_sharpe_l = []

ceq_l = []
ceq_l_ind=np.zeros(shape=(num_epochs,5))
sharpe_l_ind=np.zeros(shape=(num_epochs,5))
score_t = np.zeros(shape=(num_epochs,5))

count=0
pbar = tqdm()

for epoch in range(1, 1 + num_epochs):
    loss, acc = train(model, device, my_dataloader, optimizer, loss_fn, pbar,num_stock=5)
    print(loss)

    acc_train, score_train,y_true,y_pred = eval_1(model, device, my_dataloader, evaluator, pbar,num_stock=5,f_1_state=True)
    #print(y_true,return_true_train.shape,y_pred)
    _,_,_,_,train_sharpe,_,_,_,_=sharpe(y_true,return_true_train,y_pred,True)
    acc_val,score_val,y_true,y_pred = eval_1(model, device, my_dataloader_val, evaluator, pbar,num_stock=5,f_1_state=True)
    #print(np.sqrt(252)*np.mean(np.sum(return_true_valid,axis=1))/np.std(np.sum(return_true_valid,axis=1)))
    #print(np.sqrt(252)*np.mean(return_true_valid)/np.std(return_true_valid))
    _1,_2,_,_,val_sharpe,val_1,all_v_sharpe,ceq_0_v,ceq_1_v=sharpe(y_true,return_true_valid,y_pred,True)
    acc_test,score_test,y_true,y_pred = eval_1(model, device, my_dataloader_test, evaluator, pbar,num_stock=5,f_1_state=True)
    _3,_4,_,_,test_sharpe,test_1,all_t_sharpe,ceq_0_t,ceq_1_t=sharpe(y_true,return_true_test,y_pred,True)

    tqdm.write(f'Epoch {epoch:02d}, loss: {loss:.4f}, '
               f'train acc: {100 * acc:.2f}%, '
               f'train sharpe: {train_sharpe:.5f}, '
               f'valid acc: {100 * acc_val:.2f}%, '
               f'valid sharpe: {val_sharpe:.5f}, '
               f'valid count: {_1,_2}, '
               f'valid all long sharpe: {all_v_sharpe:.5f}%, '

               #f'test acc: {100 * acc_test:.2f}%, '
               f'test sharpe: {test_sharpe:.5f}%, '
               f'test count: {_3,_4}%, '
               f'test all long sharpe: {all_t_sharpe:.5f}%, '

               f'test ceq: {ceq_0_t:.5f}, '
               )
    print(score_test)
    #print(test_1,ceq_1_t)

    train_sharpe_l.append(train_sharpe)
    valid_sharpe_l.append(val_sharpe)
    test_sharpe_l.append(test_sharpe)
    ceq_l.append(ceq_0_t)
    ceq_l_ind[epoch-1,:]=ceq_1_t
    sharpe_l_ind[epoch-1,:]=test_1
    score_t[epoch-1,:]=score_test

# Plotting

# Create an array of epoch numbers for plotting
epochs = np.arange(1, num_epochs + 1)

# Plot the raw Sharpe ratios over epochs
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_sharpe_l, label='Train Sharpe', marker='o')
plt.plot(epochs, test_sharpe_l, label='Test Sharpe', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Sharpe Ratio')
plt.title('Train and Test Sharpe Ratios Over Epochs')
plt.grid(True)
plt.legend()
plt.savefig('./sharpe_ratios.png')  # Saves the figure to your local directory
plt.close()
