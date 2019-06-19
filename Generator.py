import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, use_cuda):
        return super(Generator, self).__init__()