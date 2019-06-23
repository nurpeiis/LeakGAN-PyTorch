import argparse
import pickle as pkl
import numpy as np
import json 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm

from data_iter import DisDataIter, GenDataIter
from utils import recurrent_func, loss_func, get_sample, get_rewards
from Discriminator import Discriminator
from Generator import Generator


#Arguments
parser = argparse.ArgumentParser(description="LeakGAN")
parser.add_argument("--hpc", action="store_true", default=False)
parser.add_argument("--data_path", type=str, default="/save_files/", metavar="PATH", 
                    help="Data path to save files (default: /save_files/)")
parser.add_argument("--rounds", type=int, default=150, metavar="N",
                    help="Rounds of adversarial training (default:150)")
parser.add_argument("--g_pretrain_steps", type=int, default=120, metavar="N",
                    help="Steps of pre-training generator (defaul: 120)")                    
parser.add_argument("--d_pretrain_steps", type=int, default=50, metavar="N",
                    help="Steps of pre-training discriminator (defaul: 50)")    
parser.add_argument("--g_steps", type=int, default=1, metavar="N", 
                    help="Steps of generator updates in one round of adversarial training (defaul: 1)") #gen_train_num
parser.add_argument("--d_steps", type=int, default=3, metavar="N",
                    help="Steps of discriminator updates in one round of adversarial training (defaul: 3)")      
parser.add_argument("--gk_epochs", type=int, default=1, metavar="N",
                    help="Epochs of generator updates in one step of generate update (defaul: 1)")        
parser.add_argument("--dk_epochs", type=int, default=3, metavar="N",
                    help="Epochs of discriminator updates in one step of generate update (defaul: 3)")  
parser.add_argument("--update_rate", type=float, default=0.8, metavar="UR",
                    help="Update rate of rollout model (defaul: 0.8)")
parser.add_argument("--n_rollout", type=int, default=16, metavar="N",
                    help="Number of rollouts (defaul: 16)") #rollout_num
parser.add_argument("--vocab_size", type=int, default=10, metavar="N",
                    help="Vocabulary size (defaul: 10)")
parser.add_argument("--batch_size", type=int, default=64, metavar="N",
                    help="Batch size(defaul: 64)")
parser.add_argument("--n_samples", type=int, default=6400, metavar="N",
                    help="Number of samples generated per time(defaul: 6400)")
parser.add_argument("--gen_lr", type=float, default=1e-3, metavar="LR",
                    help="Learning Rate of generator optimizer (defaul: 1e-3)")
parser.add_argument("--dis_lr", type=float, default=1e-3, metavar="LR",
                    help="Learning Rate of discriminator optimizer (defaul: 1e-3)")
parser.add_argument("--no_cuda", action="store_true", default=False,
                    help="Disable CUDA training (defaul: False)")                                                                        
parser.add_argument("--seed", type=int, default=1, metavar="S",
                    help="Random seed (defaul: 1)")

#Files
POSITIVE_FILE = "real.data"
NEGATIVE_FILE = "gene.data"

# Genrator Parameters
g_embed_dim = 32
g_hidden_dim = 32
g_seq_len = 20
#   MANAGER:
g_m_batch_size = 64
g_m_hidden_dim = 32
g_m_goal_out_size = 0
#   WORKER:
g_w_batch_size = 64
g_w_vocab_size = 5258
g_w_embed_dim = 32
g_w_hidden_dim = 32
g_w_goal_out_size = 0
g_w_goal_size = 16

g_step_size = 5
# Discriminator Parameters
d_seq_len = 20
d_num_classes = 2
d_vocab_size = 5258
d_dis_emb_dim = 64
d_filter_sizes = [1,2,3,4,5,6,7,8,9,10,15,20],
d_num_filters = [100,200,200,200,200,100,100,100,100,100,160,160],
d_start_token = 0
d_goal_out_size = 0
d_step_size = 5
d_dropout_prob = 0.2
d_l2_reg_lambda = 0.2

#List of models
def prepare_model_dict(use_cuda=False):
    f = open("model_params.json")
    params = json.load(f)
    f.close()
    discriminator_params = params["discriminator_params"]
    generator_params = params["generator_params"]
    worker_params = generator_params["worker_params"]
    manager_params = generator_params["manager_params"]
    discriminator_params["goal_out_size"] = sum(
        discriminator_params["num_filters"]
    )
    worker_params["goal_out_size"] = discriminator_params["goal_out_size"]
    manager_params["goal_out_size"] = discriminator_params["goal_out_size"]
    discriminator = Discriminator(**discriminator_params)
    generator = Generator(worker_params, manager_params,
                          generator_params["step_size"])
    if use_cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
    model_dict = {"generator": generator, "discriminator": discriminator}
    return model_dict

#List of optimizers
def prepare_optimizer_dict(model_dict, lr_dict): #lr_dict = learning rate 
    generator = model_dict["generator"]
    discriminator = model_dict["discriminator"]
    worker = generator.worker
    manager = generator.manager

    m_lr = lr_dict["manager"]
    w_lr = lr_dict["worker"]
    d_lr = lr_dict["discriminator"]

    w_optimizer = optim.Adam(worker.parameters(), lr=w_lr)
    m_optimizer = optim.Adam(manager.parameters(), lr=m_lr)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=d_lr)

    return {"worker": w_optimizer, "manager": m_optimizer,
            "discriminator": d_optimizer}

#List of Learning rate Schedulers
def prepare_scheduler_dict(optmizer_dict, step_size=200, gamma=0.99):
    w_optimizer = optmizer_dict["worker"]
    m_optimizer = optmizer_dict["manager"]
    d_optimizer = optmizer_dict["discriminator"]

    w_scheduler = optim.lr_scheduler.StepLR(w_optimizer, step_size=step_size,
                                            gamma=gamma)
    m_scheduler = optim.lr_scheduler.StepLR(m_optimizer, step_size=step_size,
                                            gamma=gamma)
    d_scheduler = optim.lr_scheduler.StepLR(d_optimizer, step_size=step_size,
                                            gamma=gamma)
    return {"worker": w_scheduler, "manager": m_scheduler,
            "discriminator": d_scheduler}

#Pretraining the Generator
def pretrain_generator(model_dict, optimizer_dict, scheduler_dict, dataloader, vocab_size, max_norm=5.0, use_cuda=False):
    #get the models of generator
    generator = model_dict["generator"]
    worker = generator.worker
    manager = generator.worker
    #get the optimizers
    m_optimizer = optimizer_dict["manager"]
    w_optimizer = optimizer_dict["worker"]
    
    m_optimizer.zero_grad()
    w_optimizer.zero_grad()

    m_lr_scheduler = scheduler_dict["manager"]
    w_lr_scheduler = scheduler_dict["worker"]
    """
     Perform pretrain step for real data
    """
    for i, sample in enumerate(dataloader):
        m_lr_scheduler.step()
        w_lr_scheduler.step()

        sample = Variable(sample)
        if use_cuda:
            sample = sample.cuda(asyn=True)
        
        # Calculate pretrain loss
        pre_rets = recurrent_func("pre")(model_dict, sample, use_cuda)
        real_goal = pre_rets["real_goal"]
        prediction = pre_rets["prediction"]
        delta_feature = pre_rets["delta_feature"]

        m_loss = loss_func("pre_manager")(real_goal, delta_feature)
        torch.autograd.grad(m_loss, manager.parameters())
        clip_grad_norm(manager.parameters(), max_norm=max_norm)
        m_optimizer.step()
        m_optimizer.zero_grad()

        w_loss = loss_func("pre_worker")(sample, prediction, vocab_size, use_cuda)
        torch.autograd.grad(w_loss, worker.parameters())
        clip_grad_norm(worker.parameters(), max_norm=max_norm)
        w_optimizer.step()
        w_optimizer.zero_grad()
    """
    Update model_dict, optimizer_dict, and scheduler_dict
    """

    generator.woroker = worker
    generator.manager = manager
    model_dict["generator"] = generator

    optimizer_dict["manager"] = m_optimizer
    optimizer_dict["worker"] = w_optimizer

    scheduler_dict["manager"] = m_lr_scheduler
    scheduler_dict["worker"] = w_lr_scheduler

    return model_dict, optimizer_dict, scheduler_dict

def generate_samples(model_dict, negative_file, batch_size,
                     use_cuda=False, temperature=1.0):
    neg_data = []
    for _ in range(batch_size):
        sample = get_sample(model_dict, use_cuda, temperature)
        sample = sample.cpu()
        neg_data.append(sample.data.numpy())
    neg_data = np.concatenate(neg_data, axis=0)
    np.save(negative_file, neg_data)

def pretrain_discriminator(model_dict, optimizer_dict, scheduler_dict,
                           dis_dataloader_params, vocab_size, positive_file,
                           negative_file, batch_size, num_epochs, use_cuda=False, temperature=1.0):
    discriminator = model_dict["discriminator"]

    d_optimizer = optimizer_dict["discriminator"]
    d_lr_scheduler = scheduler_dict["discriminator"]

    generate_samples(model_dict, negative_file, batch_size, use_cuda, temperature)
    dis_dataloader_params["positive_filepath"] = positive_file
    dis_dataloader_params["negative_filepath"] = negative_file
    dataloader = dis_dataloade_loader(**dis_dataloader_params) #this is where data iterator is used

    cross_entropy = nn.CrossEntropyLoss() 
    if use_cuda:
        cross_entropy = cross_entropy.cuda()
    
    for epoch in range(num_epochs):
        for _, sample in enumerate(dataloader):
            d_optimizer.zero_grad()
            data, label = sample["data"], sample["label"] #initialize sample variables
            data = Variable(data)
            label = Variable(label)
            if use_cuda:
                data = data.cuda()
                label = label.cuda()
            outs = discriminator(data)
            loss = cross_entropy(outs["score"], label.view(-1)) + discriminator.l2_loss()
            d_lr_scheduler.step()
            loss.backward()
            d_optimizer.step()
    
    model_dict["discriminator"] = discriminator
    optimizer_dict["discriminator"] = d_optimizer
    scheduler_dict["discriminator"] = d_lr_scheduler
    return model_dict, optimizer_dict, scheduler_dict


        