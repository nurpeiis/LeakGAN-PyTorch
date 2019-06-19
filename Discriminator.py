import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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
        self.start_token = start_token
        self.goal_out_size = goal_out_size
        self.l2_reg_lambda = l2_reg_lambda

        
        #Building up layers
        self.emb = nn.Embedding(vocab_size, dis_emb_dim)
        #emb = A simple lookup table that stores embeddings of a fixed dictionary and size.
        #This module is often used to store word embeddings and retrieve them using indices. The input to the module is a list of indices, and the output is the corresponding word embeddings.
        #vocab_size = size of dictionary of embeddings
        #dis_emb_dim = size of each embedding vector

        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_f, (f_size, dis_emb_dim)) for f_size, num_f in zip (filter_sizes, num_filters)
        ])
        #this is list of convolution layers each index corresponding to layer at particular step
        #in_channels = 1
        #out_channels = num_filters
        #kernel_size = f_size * embedding_vector

        self.highway = nn.Linear(sum(num_filters), sum(num_filters))
        #Highway Networks = Gating Function To Highway = y = xA^T + b
        #in_features = out_features = sum of num_festures

        self.dropout = nn.Dropout(p = dropout_prob)
        #Randomly zeroes some of the elements of the input tensor with probability p using Bernouli distribution
        #Each channel will be zeroed independently onn every forward call
        
        self.fc = nn.Linear(sum(num_filters), num_classes)
        #fully connected layer

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
        emb = self.emb(x).unsqueeze(1) #batch_siez, 1*seq_len * emb_dim
        convs = [F.relu(conv(emb)).squeeze(3) for conv in self.convs] #[batch_size * num_filter * seq_len]
        pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs] #[batch_size * num_filter]
        pred = torch.cat(pools, 1) # batch_size * sum of num_filtes_sum
        highway = self.highway(pred)
        highway = F.sigmoid(highway)*F.relu(highway) + (1. - transform)*pred # sets C = 1 - T
        features = self.dropout(highway)
        score = self.fc(features)
        pred = F.logsoftmax(score, dim=1) #batch * num_classes
        return {"pred":pred, "feature":features, "score": score}
