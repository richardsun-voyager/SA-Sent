#!/usr/bin/python
from __future__ import division
import torch
from data_reader_general import data_reader, data_generator
import pickle
import numpy as np
import codecs
import copy
import os
import models
from util import create_logger, AverageMeter
from util import save_checkpoint as save_best_checkpoint
import json
import yaml
from tqdm import tqdm
import os.path as osp
from tensorboardX import SummaryWriter
import logging
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import argparse
from torch import optim
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='TSA')

parser.add_argument('--config', default='cfgs/tweets/config_crf_cnn_elmo_tweets.yaml')#cfgs/indo/config_crf_elmo_indo.yaml')
parser.add_argument('--load_path', default='', type=str)
parser.add_argument('--e', '--evaluate', action='store_true')

args = parser.parse_args()
#tool functions
def adjust_learning_rate(optimizer, epoch, args):
    '''
    Descend learning rate
    '''
    lr = args.lr / (2 ** (epoch // args.adjust_every))
    print("Adjust lr to ", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def create_opt(parameters, config):
    '''
    Create optimizer
    '''
    if config.opt == "SGD":
        optimizer = optim.SGD(parameters, lr=config.lr, weight_decay=config.l2)
    elif config.opt == "Adam":
        optimizer = optim.Adam(parameters, lr=config.lr, weight_decay=config.l2)
    elif config.opt == "Adadelta":
        optimizer = optim.Adadelta(parameters, lr=config.lr)
    elif config.opt == "Adagrad":
        optimizer = optim.Adagrad(parameters, lr=config.lr)
    return optimizer

def mkdirs(dir):
    '''
    Create folder
    '''
    if not os.path.exists(dir):
        os.mkdir(dir)

def save_checkpoint(save_model, i_iter, args, is_best=True):
    '''
    Save the model to local disk
    '''
#     suffix = '{}_iter'.format(0)
    dict_model = save_model.state_dict()
#     print(args.snapshot_dir + suffix)
    filename = args.snapshot_dir
    save_best_checkpoint(dict_model, is_best, i_iter, filename)


def train(model, dg_train, dg_valid, dg_test, optimizer, args, tb_logger):
    cls_loss_value = AverageMeter(10)
    best_acc = 0
    best_f1 = 0
    model.train()
    is_best = False
    logger.info("Start Experiment")
    loops = int(dg_train.data_len / args.batch_size)
    for e_ in range(args.epoch):
        if e_ % args.adjust_every == 0:
            adjust_learning_rate(optimizer, e_, args)
        for idx in range(loops):
            sent_vecs, mask_vecs, label_list, sent_lens = next(dg_train.get_elmo_samples())
            cls_loss, norm_pen = model(sent_vecs.cuda(), mask_vecs.cuda(), label_list.cuda(), sent_lens.cuda())
            cls_loss_value.update(cls_loss.item())

            total_loss = cls_loss + norm_pen
            model.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm, norm_type=2)
            optimizer.step()

            if idx % args.print_freq == 0:
                print("cls loss {0} with penalty {1}".format(cls_loss.item(), norm_pen.item()))
                logger.info("i_iter {}/{} cls_loss: {:3f}".format(idx, loops, cls_loss_value.avg))
                tb_logger.add_scalar("train_loss", idx+e_*loops, cls_loss_value.avg)
                
        valid_acc, valid_f1 = evaluate_test(dg_valid, model, args)
        logger.info("epoch {}, Validation acc: {}".format(e_, valid_acc))
        if valid_acc > best_acc:
            is_best = True
            best_acc = valid_acc
            save_checkpoint(model, e_, args, is_best)
            output_samples = False
            if e_ % 10 == 0:
                output_samples = True
            test_acc, test_f1 = evaluate_test(dg_test, model, args, output_samples)
            logger.info("epoch {}, Test acc: {}".format(e_, test_acc))
        model.train()
        is_best = False


def evaluate_test(dr_test, model, args, sample_out=False):
    
    logger.info("Evaluting")
    dr_test.reset_samples()
    model.eval()
    all_counter = 0
    correct_count = 0
    true_labels = []
    pred_labels = []
    print("transitions matrix ", model.inter_crf.transitions.data)
    while dr_test.index < dr_test.data_len:
        sent, mask, label, sent_len = next(dr_test.get_elmo_samples())
        pred_label, scores, best_seq = model.predict(sent.cuda(), mask.cuda(), sent_len.cuda())

        #Compute correct predictions
        correct_count += sum(pred_label==label.cuda()).item()
        
        true_labels.extend(label.cpu().numpy())
        pred_labels.extend(pred_label.cpu().numpy())
        

            

    acc = correct_count * 1.0 / dr_test.data_len
    f1 = f1_score(true_labels, pred_labels, average='macro')
    logger.info('Confusion Matrix:')
    logger.info(confusion_matrix(true_labels, pred_labels))
    logger.info('Accuracy:{}'.format(acc))
    logger.info('f1_score:{}'.format(f1))
    logger.info('precision:{}'.format(precision_score(true_labels, pred_labels, average='macro')))
    logger.info('recall:{}'.format(recall_score(true_labels, pred_labels, average='macro')))
    return acc, f1


def main():
    """ Create the model and start the training."""
    with open(args.config) as f:
        config = yaml.load(f)


    for k, v in config['common'].items():
        setattr(args, k, v)
    mkdirs(osp.join("logs/"+args.exp_name))
    mkdirs(osp.join("checkpoints/"+args.exp_name))
    global logger
    logger = create_logger('global_logger', 'logs/' + args.exp_name + '/log.txt')

    logger.info('{}'.format(args))


    for key, val in vars(args).items():
        logger.info("{:16} {}".format(key, val))



    cudnn.enabled = True
    args.snapshot_dir = osp.join(args.snapshot_dir, args.exp_name)

    global tb_logger
    tb_logger =SummaryWriter("logs/" + args.exp_name)
    global best_acc
    best_acc = 0
    
    ##Load datasets
    dr = data_reader(args)
    train_data = dr.load_data(args.train_path)
    valid_data = dr.load_data(args.valid_path)
    test_data = dr.load_data(args.test_path)
    logger.info("Training Samples: {}".format(len(train_data)))
    logger.info("Validating Samples: {}".format(len(valid_data)))
    logger.info("Testing Samples: {}".format(len(test_data)))


    dg_train = data_generator(args, train_data)
    dg_valid = data_generator(args, valid_data, False)
    dg_test = data_generator(args, test_data, False)

    model = models.__dict__[args.arch](args)
    if args.use_gpu:
        model.cuda()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = create_opt(parameters, args)
    


    if args.training:
        train(model, dg_train, dg_valid, dg_test, optimizer, args, tb_logger)
    else:
        pass




if __name__ == "__main__":
    main()