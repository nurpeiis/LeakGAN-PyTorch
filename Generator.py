from scipy.stats import truncnorm

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, worker_params, manager_params, step_size, gpu = False):
        super(Generator, self).__init__()
        self.step_size = step_size
        self.worker = Worker(**worker_params)
        self.manager = Manager(**manager_params)
        self.temperature = 1.5

    def forward(self, index, input, w_h, m_h, feature, real_goal, no_log=False, train=False):
        """
            Pass a token at one time
            - index : index of a current token in the sentence
            - input : [batch_size]
            - w_h (worker hidden layer) : 1 * batch_size * hidden_dim
            - m_h (manager hidden layers) : 1 * batch_size * hidden_dim
            - feature (feature of Discriminator based on the current sentence) : 1 * batch_size * total_num_filters
            - real_goal : batch_size * goal_out_size
            - no_log : if true then logarithmic soft_max
            - train : true if training mode
        """
        emb = self.embeddings(input).unsqueeze(0)  # 1 * batch_size * embed_dim

        # Manager
        mana_out, mana_hidden = self.manager(feature, m_h)  # mana_out: 1 * batch_size * hidden_dim
        mana_out = self.mana2goal(mana_out.permute([1, 0, 2]))  # batch_size * 1 * goal_out_size
        cur_goal = F.normalize(mana_out, dim=-1)
        _real_goal = self.goal2goal(real_goal)  # batch_size * goal_size
        _real_goal = F.normalize(_real_goal, p=2, dim=-1).unsqueeze(-1)  # batch_size * goal_size * 1

        # Worker
        work_out, work_hidden = self.worker(emb, w_h)  # work_out: 1 * batch_size * hidden_dim
        work_out = self.work2goal(work_out).view(-1, self.vocab_size,
                                                 self.goal_size)  # batch_size * vocab_size * goal_size

        # Sample token
        out = torch.matmul(work_out, _real_goal).squeeze(-1)  # batch_size * vocab_size

        # Temperature control
        if index > 1:
            if train:
                temperature = 1.0
            else:
                temperature = self.temperature
        else:
            temperature = self.temperature

        out = temperature * out

        if no_log:
            out = F.softmax(out, dim=-1)
        else:
            out = F.log_softmax(out, dim=-1)

        return out, cur_goal, work_hidden, mana_hidden
        