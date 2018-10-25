#!/usr/bin/python
from __future__ import division
from model_att_concat import *
from data_reader_general import data_reader, data_generator
from configs.config_att import config
import pickle
from Layer import GloveMaskCat
import numpy as np
import codecs
import copy
import os

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
cat_layer = GloveMaskCat(config)
#Run an attention model, concatenate context hidden states and aspect embedding
#This is according to Yequan Wang's model, attention-based ASC
def train():
    print(config)
    best_acc = 0
    best_model = None

    ###Indoneisan
    dr = data_reader(config)
    train_data = dr.load_data('data/Indonesian/richard_train_shuffle.pkl')
    valid_data = dr.load_data('data/Indonesian/richard_valid_shuffle.pkl')
    test_data = dr.load_data('data/Indonesian/richard_test_shuffle.pkl')
    dg_train = data_generator(config, train_data)
    dg_valid =data_generator(config, valid_data, False)
    dg_test =data_generator(config, test_data, False)

    model = attTSA(config)


        # ###Bailin
    if config.if_gpu: model = model.cuda()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    # pdb.set_trace()
    optimizer = create_opt(parameters, config)

    with open(config.log_path+'att_concat_log.txt', 'w') as f:
        f.write('Start Experiment\n')


    loops = int(dg_train.data_len/config.batch_size)
    for e_ in range(config.epoch):
        print("Epoch ", e_ + 1)
        model.train()
        if e_ % config.adjust_every == 0:  
            adjust_learning_rate(optimizer, e_)

        for _ in np.arange(loops):
            model.zero_grad() 
            sent_vecs, mask_vecs, label_list, sent_lens = next(dg_train.get_ids_samples())
            sent_vecs, target_avg = cat_layer(sent_vecs, mask_vecs)#Batch_size*max_len*(2*emb_size)
            if config.if_gpu: 
                sent_vecs, target_avg = sent_vecs.cuda(), target_avg.cuda()
                label_list, sent_lens = label_list.cuda(), sent_lens.cuda()

            cls_loss = model(sent_vecs, target_avg, label_list, sent_lens)
            l2_loss = 0
            for w in model.parameters():
                if w is not None:
                    l2_loss += w.norm(2)
            print("cls loss {0} regularizrion loss {1}".format(cls_loss.item(), l2_loss.item()))
            #cls_loss += l2_loss * 0.005
            cls_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_norm, norm_type=2)
            optimizer.step()

        valid_acc = evaluate_test(dg_valid, model)
        with open(config.log_path+'att_concat_log.txt', 'a') as f:
            f.write('Epoch '+str(e_)+'\n')
            f.write('Validation accuracy:'+str(valid_acc)+'\n')
            if e_ % 1 == 0:
                print('Testing....')
                test_acc = evaluate_test(dg_test, model)
                f.write('Testing accuracy:'+str(test_acc)+'\n')


        if valid_acc > best_acc: 
            best_acc = valid_acc
            best_model = copy.deepcopy(model)
            torch.save(best_model, config.model_path+'att_concat_model.pt')






def visualize(sent, mask, best_seq, pred_label, gold):
    try:
        print(u" ".join([id2word[x] for x in sent]))
    except:
        print("unknow char..")
        return
    print("Mask", mask)
    print("Seq", best_seq)
    print("Predict: {0}, Gold: {1}".format(id2label[pred_label], id2label[gold]))
    print("")

def evaluate_test(dr_test, model):
    print("Evaluting")
    dr_test.reset_samples()
    model.eval()
    all_counter = 0
    correct_count = 0
    while dr_test.index < dr_test.data_len:
        sent, mask, label, sent_len = next(dr_test.get_ids_samples())
        sent, target = cat_layer(sent, mask)
        if config.if_gpu: 
            sent, target = sent.cuda(), target.cuda()
            label, sent_len = label.cuda(), sent_len.cuda()
        pred_label, _ = model.predict(sent, target, sent_len) 

        correct_count += sum(pred_label==label).item()
    if dr_test.data_len < 1:
        print('Testing Data Error')
    acc = correct_count * 1.0 / dr_test.data_len
    print("Sentiment Accuray {0}, {1}:{2}".format(acc, correct_count, all_counter))
    return acc


if __name__ == "__main__":
    train()