from scipy.special import expit
from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def init_vars(generator, discriminator, use_cuda=False):
    h_w_t, c_w_t = generator.init_hidden() #worker unit of gen
    h_m_t, c_m_t = generator.init_hidden() #manager unit of gen
    last_goal = Variable(torch.zeros(generator.worker.batch_size, generator.worker.goal_out_size)) #bach_size * goal_out_size
    real_goal = generator.manager.goal_init
    x_t = Variable(nn.init.constant(torch.Tensor(
        generator.worker.batch_size
    ), discriminator.start_token)).long()
    variables_ = [h_w_t, c_w_t, h_m_t, c_m_t, last_goal, real_goal, x_t]
    vs = []
    if use_cuda:
        for var in variables_:
            var = var.cuda(async=True)
            vs.append(var)
    else:
        vs = variables_
    return vs

def recurrent_func(f_type = "pre"):
    """
    There are 3 types of recurrent function:
        1. pre = pretrain
        2. adv = adversarial train
        3. rollout = rollout for evaluate reward

    Each kind of training has its own function
    """
    if f_type == "pre":
        def func(model_dict, real_data, use_cuda, temperature = 1.0):
            """
                Get generator and discriminator
            """
            
            generator = model_dict["generator"]
            discriminator = model_dict["discriminator"]
            '''
            Initialize variables and lists for forward step.
            '''
            h_w_t, c_w_t, h_m_t, c_m_t, last_goal, real_goal, x_t = \
                init_vars(generator, discriminator, use_cuda)
            t = 0
            feature_list = []
            delta_feature_list = []
            prediction_list = []
            real_goal_list = []
            batch_size = generator.worker.batch_size
            seq_len = discriminator.seq_len
            step_size = generator.step_size
            goal_out_size = generator.worker.goal_out_size
            vocab_size = discriminator.vocab_size
            """
                Forward step for pretrainning G & D
            """
            while t < seq_len +1:
                #Extract Feature from D
                if t == 0:
                    cur_sen = Variable(nn.init.constant(
                        torch.zeros(batch_size, seq_len), vocab_size
                    )).long()
                else:
                    cur_sen = real_data[:,:t]
                    cur_sen = cur_sen.contiguous()
                    cur_sen = F.pad(cur_sen.view(-1, t), (0, seq_len - t), value=vocab_size)
                if use_cuda:
                    cur_sen = cur_sen.cuda(async=True)
                f_t= discriminator(cur_sen)["feature"]
                #G forward tep
                x_t, h_m_t, c_m_t, h_w_t, c_w_t, last_goal, real_goal,\
                sub_goal, probs, t_ = generator(
                        x_t, f_t, h_m_t, c_m_t, h_w_t, c_w_t, last_goal,
                        real_goal, t, 1.0
                    )
                if t % step_size == 0:
                    if t>0:
                        real_goal = last_goal
                    last_goal = Variable(torch.zeros(batch_size, goal_out_size))
                    if use_cuda:
                        last_goal = last_goal.cuda(async=True)
                    real_goal_list.append(real_goal)
                """
                Store needed information for calculating loss function
                """
                feature_list.append(f_t)
                prediction_list.append(probs)
                if t > 0:
                    if t % step_size == 0:
                        delta_feature_list.append(f_t-feature_list[t - step_size])
                t = t_