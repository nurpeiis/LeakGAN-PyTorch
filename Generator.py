from scipy.stats import truncnorm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical

#A truncated distribution has its domain (the x-values) restricted to a certain range of values. For example, you might restrict your x-values to between 0 and 100, written in math terminology as {0 > x > 100}. There are several types of truncated distributions:
def truncated_normal(shape, lower=-0.2, upper=0.2):
    size = 1
    for dim in shape:
        size *= dim
    w_truncated = truncnorm.rvs(lower, upper, size=size)
    w_truncated = torch.from_numpy(w_truncated).float()
    w_truncated = w_truncated.view(shape)
    return w_truncated

class Manager(nn.Module):
    def __init__(self, batch_size, hidden_dim, goal_out_size):
        super(Manager, self).__init__()
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.goal_out_size = goal_out_size
        self.recurrent_unit = nn.LSTMCell(
            self.goal_out_size, #input size
            self.hidden_dim #hidden size
        )
        self.fc = nn.Linear(
            self.hidden_dim, #in_features
            self.goal_out_size #out_features
        )
        self.goal_init = nn.Parameter(torch.zeros(self.batch_size, self.goal_out_size))
        self._init_params()

    def _init_params(self):
        for param in self.parameters():
            nn.init.normal_(param, std=0.1)
        self.goal_init.data = truncated_normal(
            self.goal_init.data.shape
        )
    def forward(self, f_t, h_m_t, c_m_t):
        """
        f_t = feature of CNN from discriminator leaked at time t, it is input into LSTM
        h_m_t = ouput of previous LSTMCell
        c_m_t = previous cell state
        """
        #print("H_M size: {}".format(h_m_t.size()))
        #print("C_M size: {}".format(c_m_t.size()))
        #print("F_t size: {}".format(f_t.size()))
        h_m_tp1, c_m_tp1 = self.recurrent_unit(f_t, (h_m_t, c_m_t))
        sub_goal = self.fc(h_m_tp1)
        sub_goal = torch.renorm(sub_goal, 2, 0, 1.0) #Returns a tensor where each sub-tensor of input along dimension dim is normalized such that the p-norm of the sub-tensor is lower than the value maxnorm
        return sub_goal, h_m_tp1, c_m_tp1
class Worker(nn.Module):
    def __init__(self, batch_size, vocab_size, embed_dim, hidden_dim, 
                    goal_out_size, goal_size):
        super(Worker, self).__init__()
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.goal_out_size = goal_out_size
        self.goal_size = goal_size

        self.emb = nn.Embedding(self.vocab_size, self.embed_dim)
        self.recurrent_unit = nn.LSTMCell(self.embed_dim, self.hidden_dim)
        self.fc = nn.Linear(self.hidden_dim, self.goal_size*self.vocab_size)
        self.goal_change = nn.Parameter(torch.zeros(self.goal_out_size, self.goal_size))
        self._init_params()
        
    def _init_params(self):
        for param in self.parameters():
            nn.init.normal_(param, std=0.1)
    def forward(self, x_t, h_w_t, c_w_t):
        """
            x_t = last word
            h_w_t = last output of LSTM in Worker
            c_w_t = last cell state of LSTM in Worker
        """
        x_t_emb = self.emb(x_t)
        h_w_tp1, c_w_tp1 = self.recurrent_unit(x_t_emb, (h_w_t, c_w_t))
        output_tp1 = self.fc(h_w_tp1)
        output_tp1 = output_tp1.view(self.batch_size, self.vocab_size, self.goal_size)
        return output_tp1, h_w_tp1, c_w_tp1
class Generator(nn.Module):
    def __init__(self, worker_params, manager_params, step_size):
        super(Generator, self).__init__()
        self.step_size = step_size
        self.worker = Worker(**worker_params)
        self.manager = Manager(**manager_params)

    def init_hidden(self):
        h = Variable(torch.zeros(self.worker.batch_size, self.worker.hidden_dim))
        c = Variable(torch.zeros(self.worker.batch_size, self.worker.hidden_dim))
        return h, c

    def forward(self, x_t, f_t, h_m_t, c_m_t, h_w_t, c_w_t, last_goal, real_goal, t, temperature):
        sub_goal, h_m_tp1, c_m_tp1 = self.manager(f_t, h_m_t, c_m_t)
        output, h_w_tp1, c_w_tp1 = self.worker(x_t, h_w_t, c_w_t)
        last_goal_temp = last_goal + sub_goal
        w_t = torch.matmul(
            real_goal, self.worker.goal_change
        )
        w_t = torch.renorm(w_t, 2, 0, 1.0)
        w_t = torch.unsqueeze(w_t, -1)
        logits = torch.squeeze(torch.matmul(output, w_t))
        probs = F.softmax(temperature * logits, dim=1)
        x_tp1 = Categorical(probs).sample()
        return x_tp1, h_m_tp1, c_m_tp1, h_w_tp1, c_w_tp1,\
                last_goal_temp, real_goal, sub_goal, probs, t + 1
