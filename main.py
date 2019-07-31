import argparse
import pickle as pkl
import numpy as np
import json 
import glob
import os #for checkpoint management
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_

from data_iter import real_data_loader, dis_data_loader
from utils import recurrent_func, loss_func, get_sample, get_rewards
from Discriminator import Discriminator
from Generator import Generator
from target_lstm import TargetLSTM

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

def get_params(filePath):
    with open(filePath, 'r') as f:
        params = json.load(f)
    f.close()
    return params

def get_arguments():
    train_params = get_params("./params/train_params.json")
    leak_gan_params = get_params("./params/leak_gan_params.json")
    target_params = get_params("./params/target_params.json")
    dis_data_params = get_params("./params/dis_data_params.json")
    real_data_params = get_params("./params/real_data_params.json")
    return {
        "train_params": train_params,
        "leak_gan_params": leak_gan_params,
        "target_params": target_params,
        "dis_data_params": dis_data_params,
        "real_data_params" : real_data_params
    }
#List of models
def prepare_model_dict(use_cuda=False):
    f = open("./params/leak_gan_params.json")
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
def pretrain_generator(model_dict, optimizer_dict, scheduler_dict, dataloader, vocab_size, max_norm=5.0, use_cuda=False, epoch=1, tot_epochs=100):
    #get the models of generator
    generator = model_dict["generator"]
    worker = generator.worker
    manager = generator.manager
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
        #print("DataLoader: {}".format(dataloader))
        m_lr_scheduler.step()
        w_lr_scheduler.step()

        sample = Variable(sample)
        if use_cuda:
            sample = sample.cuda(async=True)
        
        # Calculate pretrain loss
        if (sample.size() == torch.zeros([64, 20]).size()): #sometimes smaller than 64 (16) is passed, so this if statement disables it
            #print("Sample size: {}".format(sample.size()))
            pre_rets = recurrent_func("pre")(model_dict, sample, use_cuda)
            real_goal = pre_rets["real_goal"]
            prediction = pre_rets["prediction"]
            delta_feature = pre_rets["delta_feature"]

            m_loss = loss_func("pre_manager")(real_goal, delta_feature)
            torch.autograd.grad(m_loss, manager.parameters())
            clip_grad_norm_(manager.parameters(), max_norm=max_norm)
            m_optimizer.step()
            m_optimizer.zero_grad()
            
            w_loss = loss_func("pre_worker")(sample, prediction, vocab_size, use_cuda)
            torch.autograd.grad(w_loss, worker.parameters())
            clip_grad_norm_(worker.parameters(), max_norm=max_norm)
            w_optimizer.step()
            w_optimizer.zero_grad()
            if i == 63:
                print("Pre-Manager Loss: {:.5f}, Pre-Worker Loss: {:.5f}\n".format(m_loss, w_loss))
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
                           negative_file, batch_size, epochs, use_cuda=False, temperature=1.0):
    discriminator = model_dict["discriminator"]

    d_optimizer = optimizer_dict["discriminator"]
    d_lr_scheduler = scheduler_dict["discriminator"]

    generate_samples(model_dict, negative_file, batch_size, use_cuda, temperature)
    dis_dataloader_params["positive_filepath"] = positive_file
    dis_dataloader_params["negative_filepath"] = negative_file
    #print(dis_dataloader_params)
    dataloader = dis_data_loader(**dis_dataloader_params) #this is where data iterator is used

    cross_entropy = nn.CrossEntropyLoss() #this one is similar to NLL (negative log likelihood)
    if use_cuda:
        cross_entropy = cross_entropy.cuda()
    
    for epoch in range(epochs):
        for i, sample in enumerate(dataloader):
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
            if i == 63:
                print("Pre-Discriminator loss: {:.5f}".format(loss))
    
    model_dict["discriminator"] = discriminator
    optimizer_dict["discriminator"] = d_optimizer
    scheduler_dict["discriminator"] = d_lr_scheduler
    return model_dict, optimizer_dict, scheduler_dict

#Adversarial training 
def adversarial_train(model_dict, optimizer_dict, scheduler_dict, dis_dataloader_params,
                      vocab_size, pos_file, neg_file, batch_size, gen_train_num=1,
                      dis_train_epoch=5, dis_train_num=3, max_norm=5.0,
                      rollout_num=4, use_cuda=False, temperature=1.0, epoch=1, tot_epoch=100):
    """
        Get all the models, optimizer and schedulers
    """                     
    generator = model_dict["generator"]
    discriminator = model_dict ["discriminator"]
    worker = generator.worker
    manager = generator.manager

    m_optimizer = optimizer_dict["manager"]
    w_optimizer = optimizer_dict["worker"]
    d_optimizer = optimizer_dict["discriminator"]

    #Why zero grad only m and w?
    m_optimizer.zero_grad()
    w_optimizer.zero_grad()

    m_lr_scheduler = scheduler_dict["manager"]
    w_lr_scheduler = scheduler_dict["worker"]
    d_lr_scheduler = scheduler_dict["discriminator"]

    #Adversarial training for generator
    for _ in range(gen_train_num):
        m_lr_scheduler.step()
        w_lr_scheduler.step()

        m_optimizer.zero_grad()
        w_optimizer.zero_grad()

        #get all the return values
        adv_rets = recurrent_func("adv")(model_dict, use_cuda)
        real_goal = adv_rets["real_goal"]
        all_goal = adv_rets["all_goal"]
        prediction = adv_rets["prediction"]
        delta_feature = adv_rets["delta_feature"]
        delta_feature_for_worker = adv_rets["delta_feature_for_worker"]
        gen_token = adv_rets["gen_token"]

        rewards = get_rewards(model_dict, gen_token, rollout_num, use_cuda)
        m_loss = loss_func("adv_manager")(rewards, real_goal, delta_feature)
        w_loss = loss_func("adv_worker")(all_goal, delta_feature_for_worker, gen_token, prediction, vocab_size, use_cuda)

        torch.autograd.grad(m_loss, manager.parameters()) #based on loss improve the parameters
        torch.autograd.grad(w_loss, worker.parameters())
        clip_grad_norm_(manager.parameters(), max_norm)
        clip_grad_norm_(worker.parameters(), max_norm)
        m_optimizer.step()
        w_optimizer.step()
        print("Adv-Manager loss: {:.5f} Adv-Worker loss: {:.5f}".format(m_loss, w_loss))
    
    del adv_rets
    del real_goal
    del all_goal
    del prediction
    del delta_feature
    del delta_feature_for_worker
    del gen_token
    del rewards

    #Adversarial training for discriminator
    for n in range(dis_train_epoch):
        generate_samples(model_dict, neg_file, batch_size, use_cuda, temperature)
        dis_dataloader_params["positive_filepath"] = pos_file
        dis_dataloader_params["negative_filepath"] = neg_file
        dataloader = dis_data_loader(**dis_dataloader_params)

        cross_entropy = nn.CrossEntropyLoss()
        if use_cuda:
            cross_entropy = cross_entropy.cuda()
        """
        for d-steps do
            Use current G, θm,θw to generate negative examples and combine with given positive examples S 
            Train discriminator Dφ for k epochs by Eq. (2)
        end for
        """
        for _ in range(dis_train_num): 
            for i, sample in enumerate(dataloader):
                data, label = sample["data"], sample["label"]
                data = Variable(data)
                label = Variable(label)
                if use_cuda:
                    data = data.cuda(async=True)
                    label = label.cuda(async=True)
                outs = discriminator(data)
                loss = cross_entropy(outs["score"], label.view(-1)) + discriminator.l2_loss()
                d_optimizer.zero_grad()
                d_lr_scheduler.step()
                loss.backward()
                d_optimizer.step()
        print("{}/{} Adv-Discriminator Loss: {:.5f}".format(n, range(dis_train_epoch),loss))
    #Save all changes
    model_dict["discriminator"] = discriminator
    generator.worker = worker
    generator.manager = manager
    model_dict["generator"] = generator

    optimizer_dict["manager"] = m_optimizer
    optimizer_dict["worker"] = w_optimizer
    optimizer_dict["discriminator"] = d_optimizer

    scheduler_dict["manager"] = m_lr_scheduler
    scheduler_dict["worker"] = w_lr_scheduler
    scheduler_dict["disciminator"] = d_lr_scheduler

    return model_dict, optimizer_dict, scheduler_dict


def save_checkpoint(model_dict, optimizer_dict, scheduler_dict, ckpt_num, replace=False):
    file_name = "checkpoint" + str(ckpt_num) + ".pth.tar"
    torch.save({"model_dict": model_dict, "optimizer_dict": optimizer_dict, "scheduler_dict": scheduler_dict, "ckpt_num": ckpt_num}, file_name)
    if replace:
        ckpts = glob.glob("checkpoint*")
        ckpt_nums = [int(x.split('.')[0][10:]) for x in ckpts]
        oldest_ckpt = "checkpoint" + str(min(ckpt_nums)) + ".pth.tar"
        os.remove(oldest_ckpt)

def restore_checkpoint(ckpt_path):
    checkpoint = torch.load(ckpt_path)
    return checkpoint

def main():
    """
    Get all parameters
    """
    param_dict = get_arguments()
    use_cuda = torch.cuda.is_available()
    #Random seed
    torch.manual_seed(param_dict["train_params"]["seed"])
    #Pretrain step
    checkpoint_path = param_dict["train_params"]["checkpoint_path"]
    if checkpoint_path is not None:
        checkpoint = restore_checkpoint(checkpoint_path)
        model_dict = checkpoint["model_dict"]
        optimizer_dict = checkpoint["optimizer_dict"]
        scheduler_dict = checkpoint["scheduler_dict"]
        ckpt_num = checkpoint["ckpt_num"]
    else:
        model_dict = prepare_model_dict(use_cuda)
        lr_dict = param_dict["train_params"]["lr_dict"]
        optimizer_dict = prepare_optimizer_dict(model_dict, lr_dict)
        gamma = param_dict["train_params"]["decay_rate"]
        step_size = param_dict["train_params"]["decay_step_size"]
        scheduler_dict = prepare_scheduler_dict(optimizer_dict, gamma=gamma, step_size=step_size)
    #Pretrain discriminator
    print ("#########################################################################")
    print ("Start Pretraining Discriminator...")
    with open("./params/dis_data_params.json", 'r') as f:
        dis_data_params = json.load(f)
    if use_cuda:
        dis_data_params["pin_memory"] = True
    f.close()
    pos_file = dis_data_params["positive_filepath"]
    neg_file = dis_data_params["negative_filepath"]
    batch_size = param_dict["train_params"]["generated_num"]
    vocab_size = param_dict["leak_gan_params"]["discriminator_params"]["vocab_size"]
    for i in range(param_dict["train_params"]["pre_dis_epoch_num"]):
        print("Epoch: {}/{}  Pre-Discriminator".format(i, param_dict["train_params"]["pre_dis_epoch_num"]))
        model_dict, optimizer_dict, scheduler_dict = pretrain_discriminator(model_dict, optimizer_dict, scheduler_dict, dis_data_params, vocab_size=vocab_size, positive_file=pos_file, negative_file=neg_file, batch_size=batch_size, epochs=1, use_cuda=use_cuda)
    ckpt_num = 0
    save_checkpoint(model_dict, optimizer_dict, scheduler_dict, ckpt_num)

    #Pretrain generator 
    print ("#########################################################################")
    print ("Start Pretraining Generator...")
    real_data_params = param_dict["real_data_params"]
    if use_cuda:
        real_data_params["pin_memory"] = True
    r_dataloader = real_data_loader(**real_data_params)
    for epoch in range(param_dict["train_params"]["pre_gen_epoch_num"]):
        print("Epoch: {}/{}  Pre-Generator".format(epoch, param_dict["train_params"]["pre_gen_epoch_num"]))
        model_dict, optimizer_dict, scheduler_dict = pretrain_generator(model_dict, optimizer_dict, scheduler_dict, r_dataloader, vocab_size=vocab_size, use_cuda=use_cuda, epoch=epoch, tot_epochs=range(param_dict["train_params"]["pre_gen_epoch_num"]))
    #Finish pretrain and save the checkpoint
    save_checkpoint(model_dict, optimizer_dict, scheduler_dict, ckpt_num)
    
    
    ckpt_num = 1
    #Adversarial train of D and G
    print ("#########################################################################")
    print ("Start Adversarial Training...")
    vocab_size = param_dict["leak_gan_params"]["discriminator_params"]["vocab_size"]
    save_num = param_dict["train_params"]["save_num"] #save checkpoint after this number of repetitions
    replace_num = param_dict["train_params"]["replace_num"]

    for epoch in range(param_dict["train_params"]["total_epoch"]):
        print("Epoch: {}/{}  Adv".format(epoch, param_dict["train_params"]["total_epoch"]))
        model_dict, optimizer_dict, scheduler_dict = adversarial_train(model_dict, optimizer_dict, scheduler_dict, dis_data_params, vocab_size=vocab_size, pos_file=pos_file, neg_file=neg_file, batch_size=batch_size, use_cuda=use_cuda, epoch=epoch, tot_epoch=param_dict["train_params"]["total_epoch"])
        if (epoch + 1) % save_num == 0:
            ckpt_num += 1
            if ckpt_num % replace_num == 0:
                save_checkpoint(model_dict, optimizer_dict, scheduler_dict, ckpt_num, replace=True)
            else:
               save_checkpoint(model_dict, optimizer_dict, scheduler_dict, ckpt_num)

if __name__ == "__main__":
    main()