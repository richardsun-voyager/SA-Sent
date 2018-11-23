#!/usr/bin/python
from __future__ import division
from data_reader_general import data_reader
import pickle
import numpy as np
import codecs
import copy
import os
from Layer import SimpleCat
from util import create_logger, AverageMeter
from util import save_checkpoint as save_best_checkpoint
import json
import yaml
from tqdm import tqdm
import os.path as osp
from tensorboardX import SummryWritter
import logging
import torch.nn.functional as F
import torch.backends.cudnn as cudnn



model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='TSA')

parser.add_argument('--config', default='cfgs/config_crf_elmo.yaml')
parser.add_argument('--load_path', default='', type=str)
parser.add_argument('--e', '--evaluate', action='store_true')


def adjust_learning_rate(optimizer, epoch):
    lr = config.lr / (2 ** (epoch // config.adjust_every))
    print("Adjust lr to ", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def create_opt(parameters, config):
    if config.opt == "SGD":
        optimizer = optim.SGD(parameters, lr=config.lr, weight_decay=config.l2)
    elif config.opt == "Adam":
        optimizer = optim.Adam(parameters, lr=config.lr, weight_decay=config.l2)
    elif config.opt == "Adadelta":
        optimizer = optim.Adadelta(parameters, lr=config.lr)
    elif config.opt == "Adagrad":
        optimizer = optim.Adagrad(parameters, lr=config.lr)
    return optimizer

def load_data(data_path, if_utf=False):
    f = open(data_path, 'rb')
    obj = pickle.load(f)
    f.close()
    return obj
def mkdirs(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def save_checkpoint(save_model, i_iter, args, is_best=True):
    suffix = '{}_i_iter'.format(which_model)
    dict_model = save_model.state_dict()
    print(args.snapshot_dir + suffix)

    save_best_checkpoint(dict_model, is_best, osp.join(args.snapshot_dir, suffix))


def train(model, dg_train, dg_valid, dg_test, optimizer, args):
    cls_loss_value = AverageMeter(10)
    model.train()
    is_best = False
    logger.info("Start Experiment")
    loops = int(dg_train.data_len / args.batch_size)
    for e_ in range(config.epoch):
        if e_ % args.ajust_every == 0:
            adjust_learning_rate(optimizer, e_)
        for idx in range(loops):
            sent_vecs, mask_vecs, label_list, sent_lens = next(dg_train.get_elmo_samples())
            if config.use_gpu:
                sent_vecs, mask_vecs = sent_vecs.cuda(), mask_vecs.cuda()
                label_list, sent_lens = label_list.cuda(), sent_lens.cuda()
            cls_loss = model(sent_vecs, mask_vecs, label_list, sent_lens)
            cls_loss_value.update(cls_loss.item())
            model.zero_grad()
            cls_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm, norm_type=2)
            optimizer.step()

            if e_ % args.print_freq:
                logger.info("i_iter {}/{} cls_loss: {:3f}".format(idx, loops, cls_loss_value.avg))


        valid_acc = evaluate_test(dg_valid, model)
        logger.info("epoch {}, Validation acc: {}".format(e_, valid_acc))
        if valid_acc > best_acc:
            is_best = True
            best_acc = valid_acc
            save_checkpoint(model, e_, args, is_best)
        test_acc = evaluate_test(dg_test, model)
        logger.info("epoch {}, Test acc: {}".format(e_, test_acc))
        model.train()


def evaluate_test(dr_test, model):
    logger.info("Evaluting")
    dr_test.reset_samples()
    model.eval()
    all_counter = 0
    correct_count = 0
    print("transitions matrix ", model.inter_crf.transitions.data)
    while dr_test.index < dr_test.data_len:
        sent, mask, label, sent_len = next(dr_test.get_elmo_samples())
        sent = cat_layer(sent, mask)
        if config.if_gpu:
            sent, mask = sent.cuda(), mask.cuda()
            label, sent_len = label.cuda(), sent_len.cuda()
        pred_label, best_seq = model.predict(sent, mask, sent_len)
        #visualize(sent, mask, best_seq, pred_label, label)

        correct_count += sum(pred_label==label).item()

    acc = correct_count * 1.0 / dr_test.data_len
    print("Test Sentiment Accuray {0}, {1}:{2}".format(acc, correct_count, all_counter))
    return acc

# def evaluate_dev(dev_batch, model):
#     print("Evaluting")
#     model.eval()
#     all_counter = 0
#     correct_count = 0
#     for triple_list in dev_batch:
#         for sent, mask, label in triple_list:
#             pred_label, best_seq = model.predict(sent, mask)

#             all_counter += 1
#             if pred_label == label:  correct_count += 1
#     acc = correct_count * 1.0 / all_counter
#     print("Sentiment Accuray {0}, {1}:{2}".format(acc, correct_count, all_counter))
#     return acc


def main():
    """ Create the model and start the training."""
    with open(args.config) as f:
        config = yaml.load(f)


    for k, v in config['common'].items():
        setattr(args, k, v)
    mkdirs(osp.join("logs/"+args.exp_name))

    logger = create_logger('global_logger', 'logs/' + args.exp_name + '/log.txt')

    logger.info('{}'.format(args))


    for key, val in vars(args).items():
        logger.info("{:16} {}".format(key, val))



    cudnn.enabled = True
    args.snapshot_dir = osp.join(args.snapshot_dir, args.exp_name)

    tb_logger =SummryWritter("logs/" + args.exp_name)

    id2label = ["positive", "neutral", "negative"]
    global best_acc
    best_acc = 0
    cat_layer = SimpleCat(config)
    #cat_layer.load_vector()
    if not args.finetune_embed:
        cat_layer.word_embed.weight.requires_grad = False
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


    if args.training:
        train(model, dg_train, dg_valid, dg_test, args)
    else:
        evaluate_test




if __name__ == "__main__":
    main()