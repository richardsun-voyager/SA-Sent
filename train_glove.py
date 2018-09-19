#!/usr/bin/python
from __future__ import division
from model_glove import *
from data_reader import data_reader
from config import config
import pickle
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

def pad_data(sents, masks, labels):
    sent_lens = [len(tokens) for tokens in sents]
    sent_lens = torch.LongTensor(sent_lens)
    label_list = torch.LongTensor(labels)
    max_len = max(sent_lens)
    batch_size = len(sent_lens)
    #Padding mask
    mask_vecs = np.zeros([batch_size, max_len])
    mask_vecs = torch.LongTensor(mask_vecs)
    for i, mask in enumerate(masks):
        mask_vecs[i, :len(mask)] = torch.LongTensor(mask)
    #padding sent
    sent_vecs = np.zeros([batch_size, max_len])
    sent_vecs = torch.LongTensor(sent_vecs)
    for i, s in enumerate(sents):
        sent_vecs[i, :len(s)] = torch.LongTensor(s)
    sent_lens, perm_idx = sent_lens.sort(0, descending=True)
    sent_vecs = sent_vecs[perm_idx]
    mask_vecs = mask_vecs[perm_idx]
    label_list = label_list[perm_idx]
    return sent_vecs, mask_vecs, label_list, sent_lens



id2word = load_data('data/bailin_data/dic.pkl')
id2label = ["positive", "neutral", "negative"]

def train():
    print(config)
    best_acc = 0
    best_model = None

    train_batch, test_batch = load_data('data/bailin_data/data.pkl')

    #sent_vecs, mask_vecs, label_list, sent_lens = dr.get_samples()

    model = attTSA(config)


        # ###Bailin
    if config.if_gpu: model = model.cuda()
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    # pdb.set_trace()
    optimizer = create_opt(parameters, config)

    with open(config.log_path+'log.txt', 'w') as f:
        f.write('Start Experiment\n')


    for e_ in range(config.epoch):
        print("Epoch ", e_ + 1)
        model.train()
        if e_ % config.adjust_every == 0:  
            adjust_learning_rate(optimizer, e_)

        for i, triple_list in enumerate(train_batch):
            model.zero_grad() 
            if len(triple_list) == 0: 
                continue
            ##Modified by Richard Sun
            sent, mask, label = zip(*triple_list)

            #print('Preprocessing...')
            sent_vecs, mask_vecs, label_list, sent_lens = pad_data(sent, mask, label)

            cls_loss = model(sent_vecs, mask_vecs, label_list, sent_lens)
            l2_loss = 0
            for w in parameters:
                l2_loss += torch.norm(w)
            if i %30 == 0:
                print("cls loss {0} regularizrion loss {1}".format(cls_loss.item(), l2_loss.item()))
            cls_loss += l2_loss * 0.001
            cls_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_norm, norm_type=2)
            optimizer.step()

        acc = evaluate_test(test_batch, model)
        print("Test acc: ", acc)

        # for _ in np.arange(loops):
        #     model.zero_grad() 
        #     sent_vecs, mask_vecs, label_list, sent_lens = next(dr.get_samples())
        #     if config.if_gpu: 
        #         sent_vecs, mask_vecs = sent_vecs.cuda(), mask_vecs.cuda()
        #         label_list, sent_lens = label_list.cuda(), sent_lens.cuda()
        #     cls_loss = model(sent_vecs, mask_vecs, label_list, sent_lens)
        #     cls_loss.backward()
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_norm, norm_type=2)
        #     optimizer.step()

        # #Save the model
        # acc = evaluate_test(dr_test, model)
        # print("Test acc: ", acc)
        # #Record the result
        # with open(config.log_path+'log.txt', 'a') as f:
        #     f.write('Epoch '+str(e_)+'\n')
        #     f.write('Test accuracy:'+str(acc)+'\n')

        # if acc > best_acc: 
        #     best_acc = acc
        #     best_model = copy.deepcopy(model)
        #     torch.save(best_model, config.model_path+'model.pt')





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

def evaluate_test(test_batch, model):
    print("Evaluting")
    model.eval()
    all_counter = 0
    correct_count = 0
    for triple_list in test_batch:
        model.zero_grad() 
        if len(triple_list) == 0: 
            continue
        ##Modified by Richard Sun
        sent, mask, label = triple_list
        all_counter += len(sent)

        #print('Preprocessing...')
        sent_vecs, mask_vecs, label_list, sent_lens = pad_data([sent], [mask], [label])
        #print(sent_vecs.size())
        pred_label = model.predict(sent_vecs, mask_vecs, sent_lens)

        correct_count += sum(pred_label==label_list).item()
            
    acc = correct_count * 1.0 / len(test_batch)
    print("Test Sentiment Accuray {0}, {1}:{2}".format(acc, correct_count, all_counter))
    return acc

def evaluate_dev(dev_batch, model):
    print("Evaluting")
    model.eval()
    all_counter = 0
    correct_count = 0
    for triple_list in dev_batch:
        for sent, mask, label in triple_list:
            pred_label, best_seq = model.predict(sent, mask) 

            all_counter += 1
            if pred_label == label:  correct_count += 1
    acc = correct_count * 1.0 / all_counter
    print("Sentiment Accuray {0}, {1}:{2}".format(acc, correct_count, all_counter))
    return acc

if __name__ == "__main__":
    train()