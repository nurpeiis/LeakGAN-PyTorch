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
            delta_feature_list = [] #F(St+c) - F(St) = used to calculate the gradient of manager module
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
            while t < seq_len + 1:
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
                        real_goal, t, temperature
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
            """
            Post process and return variables needed for calculating loss
            """
            if len(real_goal_list) == len(delta_feature_list) + 1:
                real_goal_list = real_goal_list[:-1] #exclude the last element
            prediction_list = prediction_list[:-1]
            real_goal_var = torch.stack(real_goal_list).permute(1,0,2)#stack = turn a list of PyTorch Tensors into one tensor, permute = rotating in regards to z axis
            prediction_var = torch.stack(prediction_list).permute(1,0,2)
            delta_feature_var = torch.stack(delta_feature_list).permute(1,0,2)
            """
            real_goal = g_t, prediction = generator sentence, delta_feature = F(s_(t+c))-F(s_t)
            """
            results = {"real_goal": real_goal_var,"prediction": prediction_var, "delta_feature": delta_feature_var}
            for result in results.values():
                if result.is_contiguous():
                    result = result.contiguous()
            return results
        return func

    #Adversarial Training
    elif f_type == "adv":
        def func(model_dict, use_cuda=False, temperature = 1.0):
            """
            Get G and D
            """
            generator = model_dict["generator"]
            discriminator = model_dict["discriminator"]
            h_w_t, c_w_t, h_m_t, c_m_t, last_goal, real_goal, x_t = \
                init_vars(generator, discriminator, use_cuda)
            t = 0
            feature_list = []
            delta_feature_list = [] # f_(t+c) - f_t
            delta_feature_for_worker_list = [] # f_t - f_(t-i)
            prediction_list = []
            real_goal_list = []
            all_goal_list = []
            gen_token_list = []
            batch_size = generator.worker.batch_size
            seq_len = discriminator.seq_len
            step_size = generator.step_size
            goal_out_size = generator.worker.goal_out_size
            vocab_size = discriminator.vocab_size
            """
            Perform forward step for adversarial training for discriminator and generator
            """
            while t < seq_len + 1:
                #Extract Feature from D
                if t == 0:
                    cur_sen = Variable(nn.init.constant(
                        torch.zeros(batch_size, seq_len), vocab_size
                    )).long()
                else:
                    cur_sen = torch.stack(gen_token_list).permute(1,0)
                    cur_sen = F.pad(cur_sen, (0, seq_len - t), value=vocab_size)
                #Why no cuda here: CHECK
                f_t = discriminator(cur_sen)["feature"]
                #Generator forward step
                x_t, h_m_t, c_m_t, h_w_t, c_w_t, last_goal, real_goal, sub_goal, probs, t_ = generator(x_t, f_t, h_m_t, c_m_t, h_w_t, c_w_t, last_goal, real_goal, t, temperature)
                if t % step_size == 0:
                    if t > 0:
                        real_goal = last_goal
                    last_goal = Variable(torch.zeros(batch_size, goal_out_size))
                    if use_cuda:
                        last_goal = last_goal.cuda(async=True)
                    real_goal_list.append(real_goal)
                #Store info for calculating loss function
                feature_list.append(f_t)
                prediction_list.append(probs)
                if t > 0:
                    if t % step_size == 0:
                        delta_feature_list.append(f_t-feature_list[t-step_size])
                        delta_feature_for_worker_list.append(f_t - feature_list[t - step_size])
                    else:
                        delta_feature_for_worker_list.append(f_t - feature_list[t - t%step_size])
                    all_goal_list.append(real_goal)
                gen_token_list.append(x_t) #next token generated by G
                t = t_
            #Post Process and return variables
            if len(real_goal_list) == len(delta_feature_list) + 1:
                real_goal_list = real_goal_list[:-1]
            prediction_list = prediction_list[:-1]
            gen_token_list = gen_token_list[:-1]
            real_goal_var = torch.stack(real_goal_list).permute(1,0,2)
            all_goal_var = torch.stack(all_goal_list).permute(1,0,2)
            prediction_var = torch.stack(prediction_list).permute(1,0,2)
            delta_feature_var = torch.stack(delta_feature_list).permute(1,0,2)
            gen_token_var = torch.stack(gen_token_list).permute(1,0,2)
            delta_feature_for_worker_var = torch.stack(delta_feature_for_worker_list).permute(1,0,2)
            results = {"real_goal": real_goal_var,
                        "all_goal": all_goal_var,
                        "prediction": prediction_var,
                        "delta_feature": delta_feature_var,
                        "delta_feature_for_worker": delta_feature_for_worker_var,
                        "gen_token": gen_token_var}
            for result in results.values():
                if result.is_contiguous():
                    result = result.contiguous()
            return results
        return func
        
    elif f_type == "rollout":
        def func(model_dict, input_x, given_num, use_cuda=False, temperature=1.0):
            #Get G and D
            generator = model_dict["generator"]
            discriminator = model_dict["discriminator"]
            #Init vairables and lists for forward step
            h_w_t, c_w_t, h_m_t, c_m_t, last_goal, real_goal, x_t = \
                init_vars(generator, discriminator, use_cuda)
            t = 0
            gen_token_list = []
            batch_size = generator.worker.batch_size
            seq_len = discriminator.seq_len
            step_size = generator.step_size
            goal_out_size = generator.worker.goal_out_size
            vocab_size = discriminator.vocab_size
            #Use input_x to perform G forward step
            while t < given_num +1:
                #Extract f_t
                if t == 0: 
                    cur_sen = Variable(nn.init.constant(torch.zeros(batch_size, seq_len), vocab_size)).long()
                    if use_cuda:
                        cur_sen = cur_sen.cuda(async=True)
                else:
                    cur_sen = torch.stack(gen_token_list).permute(1,0)
                    cur_sen = F.pad(cur_sen, (0, seq_len - t), value=vocab_size)
                f_t = discriminator(cur_sen)["feature"]
                #G forward step now that you have f
                _, h_m_t, c_m_t, h_w_t, c_w_t, last_goal, real_goal,\
                sub_goal, probs, t_ = generator( x_t, f_t, h_m_t, c_m_t, h_w_t, c_w_t, last_goal, real_goal, t, temperature)
                if t % step_size == 0:
                    if t > 0:
                        real_goal = last_goal
                    last_goal = Variable(torch.zeros(batch_size, goal_out_size))
                    if use_cuda:
                        last_goal = last_goal.cuda(async=True)
                if t < given_num:
                    x_t = input_x[:, t].contiguous()
                    gen_token_list.append(x_t)
                t = t_
                #Perform Rollout
                while t < seq_len + 1:
                    #Extract feature f_t
                    if len(gen_token_list) == 0:
                        cur_sen = Variable(nn.init.constant(torch.zeros(batch_size, seq_len), vocab_size)).long()
                        if use_cuda:
                            cur_sen = cur_sen.cuda(async=True)
                    else:
                        cur_sen = torch.stack(gen_token_list).permute(1,0)
                        cur_sen = F.pad(cur_sen, (0, seq_len - t + 1), value=vocab_size)
                    f_t = discriminator(cur_sen)["feature"]
                    #Generator forward step
                    x_t, h_m_t, c_m_t, h_w_t, c_w_t, last_goal, real_goal,sub_goal, probs, t_ = generator(x_t, f_t, h_m_t, c_m_t, h_w_t, c_w_t, last_goal,
                        real_goal, t, temperature)
                    if t % step_size == 0:
                        real_goal = last_goal
                    last_goal = Variable(torch.zeros(
                        batch_size, goal_out_size
                    ))
                    if use_cuda:
                        last_goal = last_goal.cuda(async=True)
                gen_token_list.append(x_t)
                t = t_
            gen_token = torch.stack(gen_token_list).permute(1, 0)
            return gen_token
        return func
    elif f_type == "gen":
        def func(model_dict, use_cuda=False, temperature=1.0):
            generator = model_dict["generator"]
            discriminator = model_dict["discriminator"]
            h_w_t, c_w_t, h_m_t, c_m_t, last_goal, real_goal, x_t = \
                init_vars(generator, discriminator, use_cuda)
            t = 0
            gen_token_list = []
            batch_size = generator.worker.batch_size
            seq_len = discriminator.seq_len
            step_size = generator.step_size
            goal_out_size = generator.worker.goal_out_size
            vocab_size = discriminator.vocab_size
            #G forward
            while t < seq_len:
                #Extract f_t
                if t == 0:
                    cur_sen = Variable(nn.init.constant(
                        torch.zeros(batch_size, seq_len), vocab_size)
                    ).long()
                    if use_cuda:
                        cur_sen = cur_sen.cuda(async=True)
                else:
                    cur_sen = torch.stack(gen_token_list).permute(1, 0)
                    cur_sen = F.pad(
                        cur_sen, (0, seq_len - t), value=vocab_size
                    )
                f_t = discriminator(cur_sen)["feature"]
                #G forward step
                x_t, h_m_t, c_m_t, h_w_t, c_w_t, last_goal, real_goal, sub_goal, probs, t_ = generator(x_t, f_t, h_m_t, c_m_t, 
                        h_w_t, c_w_t, last_goal,real_goal, t, temperature)
                if t % step_size == 0:
                    if t > 0:
                        real_goal = last_goal
                        last_goal = Variable(torch.zeros(batch_size, goal_out_size))
                    if use_cuda:
                        last_goal = last_goal.cuda(async=True)
                gen_token_list.append(x_t)
                t = t_
            gen_token = torch.stack(gen_token_list).permute(1,0)
            return gen_token
        return func
    else:
        raise("Invalid funnction type")
def get_sample(model_dict, use_cuda=False, temperature=1.0):
    return recurrent_func("gen")(model_dict, use_cuda, temperature)
def get_rewards(model_dict, input_x, rollout_num, use_cuda=False, temperature=1.0, delta=16.0):
    #Get G and D
    generator = model_dict["generator"]
    discriminator = model_dict["discriminator"]
    discriminator = discriminator.eval()
    #Prepare constants
    seq_len = discriminator.seq_len
    step_size = generator.step_size
    #Perform rollout and calculate reward
    rewards = []
    rollout_func = recurrent_func("rollout")
    for i in range(rollout_num):
        given_num = 0
        while given_num < seq_len:
            sample_for_reward = rollout_func(model_dict, input_x, given_num, use_cuda, temperature)
            pred = discriminator(sample_for_reward)["pred"]
            pred = pred[:, 1].data
            if use_cuda:
                pred = pred.cpu()
            pred = pred.numpy()
            pred = pred.reshape(-1)
            if i == 0:
                rewards.append(pred)
            else:
                rewards[int(given_num/step_size -1)] += pred
            given_num += step_size
    rewards = rescale(rewards, delta) / rollout_num
    if use_cuda:
        rewards = rewards.cuda(async=True)
    discriminator = discriminator.train()
    return rewards
def rescale(rewards, delta=16.0):
    ""