#!/usr/bin/python
from __future__ import division
from model_crf import *
from data_reader import data_reader
from config import config
import pickle
import numpy as np
import codecs
import copy
import os
from Layer import SimpleCat

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

id2label = ["positive", "neutral", "negative"]

cat_layer = SimpleCat(config)
cat_layer.load_vector()
cat_layer.word_embed.weight.requires_grad = False

def train():
    print(config)
    best_acc = 0
    best_model = None

    # dr = data_reader(config)
    # dr.load_data('data/2014/training.pickle')
    # dr_valid = data_reader(config, False)
    # dr_valid.load_data('data/2014/valid.pickle')
    # dr_test = data_reader(config, False)
    # dr_test.load_data('data/2014/testing.pickle')


    ###Indoneisan
    dr = data_reader(config)
    dr.load_data(config.train_path)
    dr_valid = data_reader(config, False)
    dr_valid.load_data(config.valid_path)
    dr_test = dr_valid

    #sent_vecs, mask_vecs, label_list, sent_lens = dr.get_samples()

    model = AspectSent(config)


        # ###Bailin
    if config.if_gpu: model = model.cuda()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    # pdb.set_trace()
    optimizer = create_opt(parameters, config)

    with open(config.log_path+'log.txt', 'w') as f:
        f.write('Start Experiment\n')


    loops = int(dr.data_len/config.batch_size)
    for e_ in range(config.epoch):
        print("Epoch ", e_ + 1)
        model.train()
        if e_ % config.adjust_every == 0:  
            adjust_learning_rate(optimizer, e_)

        for _ in np.arange(loops):
            model.zero_grad() 
            sent_vecs, mask_vecs, label_list, sent_lens = next(dr.get_ids_samples())
            sent_vecs = cat_layer(sent_vecs, mask_vecs, False)
            if config.if_gpu: 
                sent_vecs, mask_vecs = sent_vecs.cuda(), mask_vecs.cuda()
                label_list, sent_lens = label_list.cuda(), sent_lens.cuda()
            cls_loss = model(sent_vecs, mask_vecs, label_list, sent_lens)
            cls_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_norm, norm_type=2)
            optimizer.step()

        #Save the model
        acc = evaluate_test(dr_test, model)
        print("Test acc: ", acc)
        #Record the result
        with open(config.log_path+'log.txt', 'a') as f:
            f.write('Epoch '+str(e_)+'\n')
            f.write('Test accuracy:'+str(acc)+'\n')

        if acc > best_acc: 
            best_acc = acc
            best_model = copy.deepcopy(model)
            torch.save(best_model, config.model_path+'model.pt')





# def visualize(sent, mask, best_seq, pred_label, gold):
#     try:
#         print(u" ".join([id2word[x] for x in sent]))
#     except:
#         print("unknow char..")
#         return
#     print("Mask", mask)
#     print("Seq", best_seq)
#     print("Predict: {0}, Gold: {1}".format(id2label[pred_label], id2label[gold]))
#     print("")

def evaluate_test(dr_test, model):
    print("Evaluting")
    dr_test.reset_samples()
    model.eval()
    all_counter = 0
    correct_count = 0
    print("transitions matrix ", model.inter_crf.transitions.data)
    while dr_test.index < dr_test.data_len:
        sent, mask, label, sent_len = next(dr_test.get_ids_samples())
        sent = cat_layer(sent, mask, False)
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

if __name__ == "__main__":
    train()