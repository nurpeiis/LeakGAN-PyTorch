import torch
from scipy.stats import truncnorm
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#A truncated distribution has its domain (the x-values) restricted to a certain range of values. For example, you might restrict your x-values to between 0 and 100, written in math terminology as {0 > x > 100}. There are several types of truncated distributions:
def truncated_normal(shape, lower=-0.2, upper=0.2):
    size = 1
    for dim in shape:
        size *= dim
    w_truncated = truncnorm.rvs(lower, upper, size=size)
    w_truncated = torch.from_numpy(w_truncated).float()
    w_truncated = w_truncated.view(shape)
    return w_truncated

class Highway(nn.Module):
    #Highway Networks = Gating Function To Highway = y = xA^T + b
    def __init__(self, in_size, out_size):
        super(Highway, self).__init__()
        self.fc1 = nn.Linear(in_size, out_size)
        self.fc2 = nn.Linear(in_size, out_size)
    def forward(self, x):
        #highway = F.sigmoid(highway)*F.relu(highway) + (1. - transform)*pred # sets C = 1 - T
        g = F.relu(fc1)
        t = F.sigmoid(fc2)
        out = g*t + (1. - t)*x
        return out
class Discriminator(nn.Module):
    """
    A CNN for text classification
    num_filters (int): This is the output dim for each convolutional layer, which is the number
          of "filters" learned by that layer.
    """
    def __init__(self, seq_len, num_classes, vocab_size, dis_emb_dim, 
                    filter_sizes, num_filters, start_token, goal_out_size, step_size, dropout_prob, l2_reg_lambda):
        super(Discriminator, self).__init__()
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.dis_emb_dim = dis_emb_dim
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.start_token = start_token
        self.goal_out_size = goal_out_size
        self.step_size = step_size
        self.dropout_prob = dropout_prob
        self.l2_reg_lambda = l2_reg_lambda
        self.num_filters_total = sum(self.num_filters)
        
        #Building up layers
        self._init_embedding()
        self._init_feature_extractor()
        self._init_fully_connected()
        
        
        
        
        #fully connected layer
    def _init_embedding(self):
        """
            emb = A simple lookup table that stores embeddings of a fixed dictionary and size.
            This module is often used to store word embeddings and retrieve them using indices. The input to the module is a list of indices, and the output is the corresponding word embeddings.
            vocab_size = size of dictionary of embeddings
            dis_emb_dim = size of each embedding vector
        """
        self.emb = nn.Embedding(self.vocab_size + 1, self.dis_emb_dim)
        nn.init.uniform(self.emb.weight, -1.0, 1.0) # fills input tensor with values drawn from the uniform distribution from -1 to 1
    def _init_feature_extractor(self):
        """
            this is list of convolution layers each index corresponding to layer at particular step
            in_channels = 1
            out_channels = num_filters
            kernel_size = f_size * embedding_vector
        """
        """
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_f, (f_size, self.dis_emb_dim)) for f_size, num_f in zip (filter_sizes, num_filters)
        ])
        """
        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        for f_size, out_channels in zip(self.filter_sizes, self.num_filters):
            conv = nn.Conv2d(1, out_channels, kernel_size=(f_size, self.dis_emb_dim))
            #Initialize conv extractor's weight with truncated normal 
            conv.weight.data = truncated_normal(conv.weight.data.shape)
            #initialize conv extractor's bias with constant
            nn.init.constant(conv.bias, 0.1)
            convs.append(conv)
            #Initialize two dimensional max pooling layer
            pool = nn.MaxPool2d(self.seq_len - f_size + 1, 1)
            self.pools.append(pool)

        self.highway = Highway(self.num_filters_total, self.num_filters_total)
        #in_features = out_features = sum of num_festures
        self.dropout = nn.Dropout(p = dropout_prob)
        #Randomly zeroes some of the elements of the input tensor with probability p using Bernouli distribution
        #Each channel will be zeroed independently onn every forward call
    def _init_fully_connected(self):
        self.fc = nn.Linear(self.num_filters_total, self.num_classes)
        self.fc.weight.data = truncated_normal(self.fc.weight.data.shape)
        nn.init.constant(self.fc.bias, 0.1)
    def forward(self, x):
        """
        Argument:
            x: shape(batch_size * self.seq_len)
               type(Variable containing torch.LongTensor)
        Return:
            pred: shape(batch_size * 2)
                  For each sequence in the mini batch, output the probability
                  of it belonging to positive sample and negative sample.
            feature: shape(batch_size * self.num_filters_total)
                     Corresponding to f_t in original paper
            score: shape(batch_size, self.num_classes)
              
        """
        #1. Embedding Layer
        #2. Convolution + maxpool layer for each filter size
        #3. Combine all the pooled features into a prediction
        #4. Add highway
        #5. Add dropout. This is when feature should be extracted
        #6. Final unnormalized scores and predictions
        emb = self.emb(x) 
        batch_size, seq_len, emb_dim = emb.data.shape
        emb = emb.view(batch_size, 1, seq_len, emb_dim)#batch_size, 1*seq_len * emb_dim
        pooled_out = []
        for conv, pool in zip(self.convs, self.pools):
            h = F.relu(conv(emb))#[batch_size * num_filter * seq_len]
            pooled_out.append(pool(h)) #[batch_size * num_filter]
        pred = torch.cat(pooled_out, 1) 
        pred = pred.view(-1, self.num_filters_total)# batch_size * sum of num_filtes_sum
        highway = self.highway(pred)
        features = self.dropout(highway)
        score = self.fc(features)
        pred = F.logsoftmax(score, dim=1) #batch * num_classes
        return {"pred":pred, "feature":features, "score": score}
    def l2_loss(self):
        W = self.fc.weight
        b = self.fc.bias
        l2_loss = torch.sum(W*W) + torch.sum(b*b)
        l2_loss = self.l2_reg_lambda * l2_loss
        return l2_loss
