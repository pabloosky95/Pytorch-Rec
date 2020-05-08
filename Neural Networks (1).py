import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# ### Import movielense data 100k

data= pd.read_csv("C:\\Users\\Pavlos\\Downloads\\ml-100k\\ml-100k\\u.data",
                  sep="\t",
                  header=None,
                  names=["userID","movieID","rating","timestamp"]
                 )
data=data.sort_values(by="userID")
data=data[["userID","movieID","rating"]]
data.head(5)


# ### Split the data into training and validation

data_t, data_v =train_test_split(data, test_size=0.2)

# ### Encode the data

# here is a handy function modified from fast.ai
def proc_col(col, train_col=None):
    """Encodes a pandas column with continous ids. 
    """
    if train_col is not None:
        uniq = train_col.unique()
    else:
        uniq = col.unique()
    name2idx = {o:i for i,o in enumerate(uniq)}
    return name2idx, np.array([name2idx.get(x, -1) for x in col]), len(uniq)


def encode_data(data, train=None):
    """ Encodes rating data with continous user and movie ids. 
    If train is provided, encodes data with the same encoding as train.
    """
    data = data.copy()
    for col_name in ["userID", "movieID"]:
        train_col = None
        if train is not None:
            train_col = train[col_name]
        _,col,_ = proc_col(data[col_name], train_col)
        data[col_name] = col
        data = data[data[col_name] >= 0]
    return data


# encoding the train and validation data
data_t = encode_data(data_t)
data_v = encode_data(data_v, data_t)


# ## 2.0 Embeding


import torch
import torch.nn as nn
import torch.nn.functional as F # contains useful loss functions


# Create the model

class MF(nn.Module):
    def __init__(self,num_users,num_items, emb_size=100):  #emb_size= embeding size , how big is the representation of my user
        super(MF,self).__init__() # not sure why we use this.
        self.user_emb = nn.Embedding(num_users, emb_size) # create an embeding for all users of size 100
        self.item_emb = nn.Embedding(num_items, emb_size) # create an embeding for all items of size 100
        self.user_emb.weight.data.uniform_(0, 0.05) # help with convergence but why?
        self.item_emb.weight.data.uniform_(0, 0.05)
        
    def forward(self,u,v): ## where u=list of users, v= list of movies.
        u=self.user_emb(u) # Firstly the function looks up the embeding for the list of users u in the same way as in [1]
        v=self.item_emb(v) # Similarly looks up the representation vector for movies
        return (u*v).sum(1) # basically what we do here is an element-wise multiplication of the vectors and then sum (dot-product)
            
 # Train the model        
num_users = data_t.userID.nunique()
num_items = data_t.movieID.nunique()
print(num_users, num_items)

# define the model
model = MF(num_users, num_items, emb_size=100)



def train(model, epochs=10, lr=0.01, wd=0.0, unsqueeze=False): # epochs is one cycle through the full training dataset
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd) # weight_decay adds a L2 penalty to the cost which can effectively lead to to smaller model weights
    model.train()
    for i in range(epochs):
        users = torch.LongTensor(data_t.userID.values) # takes the list of users from our training set
        items = torch.LongTensor(data_t.movieID.values) 
        ratings = torch.FloatTensor(data_t.rating.values) 
        if unsqueeze:
            ratings = ratings.unsqueeze(1)
        y_hat = model(users, items) # computes the dot product
        loss = F.mse_loss(ratings, y_hat) # Measures the element-wise mean squared error
        optimizer.zero_grad() 
        '''' In PyTorch, we need to set the gradients to zero before starting to do
        backpropragation because PyTorch accumulates the gradients on subsequent backward passes'''
        loss.backward() # backpropagation 
        optimizer.step() 
        print(loss.item()) 
    test_loss(model, unsqueeze)
    
# Evaluation Function
def test_loss(model, unsqueeze=False):
    model.eval()
    users = torch.LongTensor(data_v.userID.values) 
    items = torch.LongTensor(data_v.movieID.values) 
    ratings = torch.FloatTensor(data_v.rating.values) 
    if unsqueeze:
        ratings = ratings.unsqueeze(1)
    y_hat = model(users, items)
    loss = F.mse_loss(y_hat, ratings)
    print("test loss %.3f " % loss.item())
    
train(model, epochs=10, lr=0.05)


# Add bias
class MF_bias(nn.Module):
    def __init__(self, num_users, num_items, emb_size=100):
        super(MF_bias, self).__init__()
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_emb = nn.Embedding(num_items, emb_size)
        self.item_bias = nn.Embedding(num_items, 1)
        self.user_emb.weight.data.uniform_(0,0.05)
        self.item_emb.weight.data.uniform_(0,0.05)
        self.user_bias.weight.data.uniform_(-0.01,0.01)
        self.item_bias.weight.data.uniform_(-0.01,0.01)
        
    def forward(self, u, v):
        U = self.user_emb(u)
        V = self.item_emb(v)
        b_u = self.user_bias(u).squeeze()
        b_v = self.item_bias(v).squeeze()
        return (U*V).sum(1) +  b_u  + b_v


model = MF_bias(num_users, num_items, emb_size=100) #.cuda()




