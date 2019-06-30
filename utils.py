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
    x_t = Variable(nn.init.constant_(torch.Tensor(
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
            #print("After sample size: {}".format(real_data.size()))
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
                    cur_sen = Variable(nn.init.constant_(
                        torch.zeros(batch_size, seq_len), vocab_size
                    )).long()
                    #print("Batch Size: {}".format(batch_size))
                    #print("Real Data: {}".format(cur_sen.size()))
                else:
                    cur_sen = real_data[:,:t]
                    #print("Real Data: {}".format(real_data.size()))
                    #print("t: {}".format(t))
                    cur_sen = cur_sen.contiguous()
                    cur_sen = F.pad(cur_sen.view(-1, t), (0, seq_len - t), value=vocab_size)
                if use_cuda:
                    cur_sen = cur_sen.cuda(async=True)
                #print("Current sentence:{}".format(cur_sen))
                #print("Current sentence size:{}".format(cur_sen.size()))
                f_t= discriminator(cur_sen)["feature"]
                #print("F_t from discr: {}".format(f_t))
                #print("F_t from discr: {}".format(f_t.size()))
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
            #print("Prediction stack before stacking: {}".format(torch.stack(prediction_list).size()))
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
                    cur_sen = Variable(nn.init.constant_(
                        torch.zeros(batch_size, seq_len), vocab_size
                    )).long()
                else:
                    #print("Cur sen size before permute: {}".format(cur_sen.size()))
                    cur_sen = torch.stack(gen_token_list).permute(1,0)
                    #print("Cur sen size: {}".format(cur_sen.size()))
                    cur_sen = F.pad(cur_sen, (0, seq_len - t), value=vocab_size)
                #Why no cuda here: CHECK: ADD CUDA!!!!
                if use_cuda:
                    cur_sen = cur_sen.cuda(async=True)
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
                #print("X size: {}".format(x_t.size()))
            #Post Process and return variables
            if len(real_goal_list) == len(delta_feature_list) + 1:
                real_goal_list = real_goal_list[:-1]
            prediction_list = prediction_list[:-1]
            gen_token_list = gen_token_list[:-1]
            real_goal_var = torch.stack(real_goal_list).permute(1,0,2)
            all_goal_var = torch.stack(all_goal_list).permute(1,0,2)
            prediction_var = torch.stack(prediction_list).permute(1,0,2)
            delta_feature_var = torch.stack(delta_feature_list).permute(1,0,2)
            #print(delta_feature_var)
            #print("Delta feature list size: {}".format(len(delta_feature_list)))
            #print("Gen token list size: {}".format(len(gen_token_list)))
            gen_token_var = torch.stack(gen_token_list).permute(1,0)
            #print("Gen token var after correct permute: {}".format(gen_token_var.size()))
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
                    cur_sen = Variable(nn.init.constant_(torch.zeros(batch_size, seq_len), vocab_size)).long()
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
                    cur_sen = Variable(nn.init.constant_(torch.zeros(batch_size, seq_len), vocab_size)).long()
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
                    cur_sen = Variable(nn.init.constant_(
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
        #print("Sequence length: {}".format(seq_len))
        #print("i stage in rollout: {}".format(i))
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
    """
    Why Rescaled activation: during adversarial training of SeqGAN severe gradient vanishing occurs when D is much stronger than G, i.e. the reward is too small value to update the parameters
    and thus need to be rescaled before being fed into G.
        parameters for rewards:
            type: list
            length: seq_len / c, where c is c recent goals(steps into future)
            elements: np.array(size=batch_size)
            R(reward matrix) = expit(delta * (0.5 - rank(i)/B)), where expit, is an activation function that re-projects the equidifferent scoring based on ranking to a more effective distribution. 
            In this model authors of the paper decided expit to be sigmoid function: expit = 1/(1+exp(-x))
    """
    r = np.array(rewards)
    _, batch_size = r.shape
    order = np.argsort(r)
    rank = np.argsort(order)
    rank = batch_size - rank
    rescaled_rewards = expit(delta*(0.5 - rank/batch_size))
    rescaled_rewards = np.transpose(rescaled_rewards)
    return Variable(torch.from_numpy(rescaled_rewards)).float()

def one_hot(x, vocab_size, use_cuda=False):
    batch_size, seq_len = x.size()
    #print(x.device)
    out = torch.zeros(batch_size* seq_len, vocab_size, device=x.device)
    #
    # print(out.size())
    x = x.contiguous()
    x = x.view(-1, 1)
    #print("X size: {}".format(x.size()))
    #print("Out size: {}".format(out.size()))
    #print("Out size at dim 1: {}".format(out.size(1)))
    if (x.data < vocab_size).all() == 0:
        for i, d in enumerate(x.data):
            if x[i].item() > vocab_size - 1 :
                x[i] = 0
                #print(x[i])
                #print (i)
    out = out.scatter_(1, x.data, 1.0) #setting particular values of a tensor at the provided indices, one hot vector at positions where there is word
    """
        check places with 1.0 in out
        a = (out == 1.0).nonzero()
        print(a)
    """

    out = out.view(batch_size, seq_len, vocab_size)
    out = Variable(out)
    
    if use_cuda:
        out = out.cuda(async=True)
    return out

def loss_func(f_type="pre_worker"):
    """
    5 kind of loss function: pre_worker, pre_manager, adv_worker, adv_manager, dis
    """
    if f_type == "pre_worker":
        def func(real_data, prediction, vocab_size, use_cuda=False):
            #print("Prediction shape before: {}".format(prediction.size()))
            prediction = torch.clamp(prediction, 1e-20, 1.0) # put min and max boundaries
            #print("One Hot: {}".format(one_hot(real_data, vocab_size, use_cuda).size()))
            #print("Real data size: {}".format(real_data.size()))
            #print("Log Prediction: {}".format(torch.log(prediction).size()))
            hot_one = one_hot(real_data, vocab_size, use_cuda)
            #print("Pred after reshape: {}".format(prediction.size()))
            #print("One Hot after reshape: {}".format(hot_one.size()))
            loss = -torch.mean(one_hot(real_data, vocab_size, use_cuda) * torch.log(prediction))
            return loss
        return func
    elif f_type == "pre_manager":
        def func(real_goal, delta_feature):
            loss = -torch.mean(1.0 - F.cosine_similarity(real_goal, delta_feature))
            return loss
        return func
    elif f_type == "adv_worker":
        def func(all_goal, delta_feature_for_worker, gen_token, prediction, vocab_size, use_cuda=False):
            intrinsic_rewards = 1.0 - F.cosine_similarity(all_goal, delta_feature_for_worker, dim=2)
            prediction = torch.clamp(prediction, 1e-20, 1.0)
            loss = -torch.mean(intrinsic_rewards * torch.sum(one_hot(gen_token, vocab_size, use_cuda)* torch.log(prediction), dim=2))
            return loss
        return func
    elif f_type == "adv_manager":
        def func(rewards, real_goal, delta_feature):
            loss = -torch.mean(rewards*(1.0 - F.cosine_similarity(delta_feature, real_goal, dim=2)))
            return loss
        return func
    elif f_type == "dis":
        def func(discriminator, input_x, score, use_cuda=False):
            """
            input_x:
                size(batch_size*seq_len)
                type(torch.LongTensor)
            score:
                size(batch_size * seq_len * vocab_size)
                type(torch.FloatTensor)
            """
            loss_func = nn.CrossEntropyLoss() 
            if use_cuda:
                loss_func = loss_func.cuda()
            input_x = input_x.view(-1) #last dim
            batch_size, seq_len, vocab_size = score.size()
            score = score.view(batch_size * seq_len, -1) #reshape
            loss = loss_func(score, input_x) + discriminator.l2_loss()
            return loss
        return func
    else:
        raise("Invalid loss function type")