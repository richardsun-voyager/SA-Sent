#!/usr/bin/python
from __future__ import division
from model_gcnn2_glove import *
from data_reader_general import *
from config import config
from torch import optim
import pickle
from Layer import GloveMaskCat
import numpy as np
import codecs
import copy
import os

def adjust_learning_rate(optimizer, epoch):
    lr = config.lr / (1.5 ** (epoch // config.adjust_every))
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



id2word = load_data('data/bailin_data/dic.pkl')
id2label = ["positive", "neutral", "negative"]
#Load concatenation layer and attention model layer
cat_layer = GloveMaskCat(config)


def train():
    print(config)
    best_acc = 0
    best_model = None

    TRAIN_DATA_PATH = "data/2014/Restaurants_Train_v2.xml"
    TEST_DATA_PATH = "data/2014/Restaurants_Test_Gold.xml"
    path_list = [TRAIN_DATA_PATH, TEST_DATA_PATH]
    #First time, need to preprocess and save the data
    #Read XML file
    # dr = data_reader(config)
    # dr.read_train_test_data(path_list)
    # print('Data Preprocessed!')



    #Load preprocessed data directly
    dr = data_reader(config)
    train_data = dr.load_data(config.data_path+'Restaurants_Train_v2.xml.pkl')

    # train_path = config.data_path + 'train_restaurant.pkl'
    valid_path = config.data_path + 'valid_restaurant.pkl'
    # train_data = dr.load_data(train_path)
    valid_data = dr.load_data(valid_path)
    #Split training data into training and validating parts
    #dr.split_save_data(train_path, valid_path)
    #print('Split successfullly')
    test_data = dr.load_data(config.data_path+'Restaurants_Test_Gold.xml.pkl')
    dg_train = data_generator(config, train_data)
    dg_valid = data_generator(config, valid_data, False)
    dg_test = data_generator(config, test_data, False)

    # dr_valid.load_data(config.valid_path)
    # dr_test = dr_valid
    # dr_test = data_reader(config, False)
    # dr_test.load_data(config.test_path)

    

    cat_layer.load_vector()

    #train_batch, test_batch = load_data('data/bailin_data/data.pkl')
    #sent_vecs, mask_vecs, label_list, sent_lens = dr.get_samples()

    model = CNN_Gate_Aspect_Text(config)


        # ###Bailin
    if config.if_gpu: model = model.cuda()
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    # pdb.set_trace()
    optimizer = create_opt(parameters, config)

    with open(config.log_path+'log.txt', 'w') as f:
        f.write('Start Experiment\n')

    loops = int(dg_train.data_len/config.batch_size)
    for e_ in range(config.epoch):
        print("Epoch ", e_ + 1)
        model.train()
        if e_ % config.adjust_every == 0:  
            adjust_learning_rate(optimizer, e_)

        for _ in np.arange(loops):
            optimizer.zero_grad() 
            sent_vecs, mask_vecs, label_list, sent_lens = next(dg_train.get_ids_samples())
            #Note here we do not use average target embeddings
            sent_vecs, target_avg = cat_layer(sent_vecs, mask_vecs, False)#Batch_size*max_len*(2*emb_size)
            if config.if_gpu: 
                sent_vecs, target_avg = sent_vecs.cuda(), target_avg.cuda()
                label_list, sent_lens = label_list.cuda(), sent_lens.cuda()

            cls_loss = model(sent_vecs, target_avg, label_list, sent_lens)
            l2_loss = 0
            for w in model.parameters():
                if w is not None:
                    l2_loss += w.norm(2)
            print("cls loss {0} regularizrion loss {1}".format(cls_loss.item(), l2_loss.item()))
            #cls_loss += l2_loss * 0.008#previous we tried 0.005
            cls_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_norm, norm_type=2)
            optimizer.step()

        acc = 0
        acc = evaluate_test(dg_valid, model)
        with open(config.log_path+'log.txt', 'a') as f:
            f.write('Epoch '+str(e_)+'\n')
            f.write('Validation accuracy:'+str(acc)+'\n')
            if e_ % 1 == 0:
                print('Testing....')
                acc = evaluate_test(dg_test, model)
                f.write('Testing accuracy:'+str(acc)+'\n')








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

# def evaluate_test(test_batch, model):
#     print("Evaluting")
#     model.eval()
#     all_counter = 0
#     correct_count = 0
#     for triple_list in test_batch:
#         model.zero_grad() 
#         if len(triple_list) == 0: 
#             continue
#         ##Modified by Richard Sun
#         sent, mask, label = triple_list
#         all_counter += len(sent)

#         #print('Preprocessing...')
#         sent_vecs, mask_vecs, label_list, sent_lens = pad_data([sent], [mask], [label])
#         #print(sent_vecs.size())
#         pred_label = model.predict(sent_vecs, mask_vecs, sent_lens)

#         correct_count += sum(pred_label==label_list).item()
            
#     acc = correct_count * 1.0 / len(test_batch)
#     print("Test Sentiment Accuray {0}, {1}:{2}".format(acc, correct_count, all_counter))
#     return acc

def evaluate_test(dr_test, model):
    print("Evaluting")
    dr_test.reset_samples()
    model.eval()
    all_counter = 0
    correct_count = 0
    while dr_test.index < dr_test.data_len:
        sent, mask, label, sent_len = next(dr_test.get_ids_samples())
        sent, target = cat_layer(sent, mask, False)#not using average target embeddings
        if config.if_gpu: 
            sent, target = sent.cuda(), target.cuda()
            label, sent_len = label.cuda(), sent_len.cuda()
        pred_label  = model.predict(sent, target, sent_len) 

        correct_count += sum(pred_label==label).item()
    if dr_test.data_len < 1:
        print('Testing Data Error')
    acc = correct_count * 1.0 / dr_test.data_len
    print("Sentiment Accuray {0}, {1}:{2}".format(acc, correct_count, all_counter))
    return acc

if __name__ == "__main__":
    train()