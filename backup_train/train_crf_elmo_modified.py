#!/usr/bin/python
from __future__ import division
from model_crf_elmo_modified import *
from data_reader_general import data_reader, data_generator
from configs.config_crf_elmo import config
import pickle
import numpy as np
import codecs
import copy
import os
from sklearn.metrics import confusion_matrix
torch.manual_seed(111)

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



def train():
    print(config)
    best_acc = 0
    best_model = None

    #Load preprocessed data directly
    dr = data_reader(config)
    train_data = dr.load_data(config.train_path)
    valid_data = dr.load_data(config.valid_path)
    test_data = dr.load_data(config.test_path)
    print('Training Samples:', len(train_data))
    print('Validating Samples:', len(valid_data))
    print('Testing Samples:', len(test_data))

    dg_train = data_generator(config, train_data)
    dg_valid =data_generator(config, valid_data, False)
    dg_test =data_generator(config, test_data, False)

    model = AspectSent(config)



    ###Bailin
    if config.if_gpu: model = model.cuda()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    # pdb.set_trace()
    optimizer = create_opt(parameters, config)

    with open(config.log_path+'crf_elmo_log.txt', 'w') as f:
        f.write('Start Experiment\n')


    loops = int(dg_train.data_len/config.batch_size)
    for e_ in range(config.epoch):
        print("Epoch ", e_ + 1)
        model.train()
        if e_ % config.adjust_every == 0:  
            adjust_learning_rate(optimizer, e_)

        for _ in np.arange(loops):
            model.zero_grad() 
            sent_vecs, mask_vecs, label_list, sent_lens = next(dg_train.get_elmo_samples())
            if config.if_gpu: 
                sent_vecs, mask_vecs = sent_vecs.cuda(), mask_vecs.cuda()
                label_list, sent_lens = label_list.cuda(), sent_lens.cuda()
            cls_loss = model(sent_vecs, mask_vecs, label_list, sent_lens)
            cls_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_norm, norm_type=2)
            optimizer.step()

        #Save the model
        valid_acc = evaluate_test(dg_valid, model)
        print("Validation acc: ", valid_acc)
        #Record the result
        with open(config.log_path+'crf_elmo_log.txt', 'a') as f:
            f.write('Epoch '+str(e_)+'\n')
            f.write('Validation accuracy:'+str(valid_acc)+'\n')
            if e_%1 == 0:
                test_acc = evaluate_test(dg_test, model)
                f.write('Test accuracy:'+str(test_acc)+'\n')

        if valid_acc > best_acc: 
            best_acc = valid_acc
            best_model = copy.deepcopy(model)
            torch.save(best_model, config.model_path+'crf_elmo_model.pt')
    

def evaluate_test(dr_test, model):
    print("Evaluting")
    dr_test.reset_samples()
    model.eval()
    all_counter = dr_test.data_len
    correct_count = 0
    true_labels = []
    pred_labels = []
    while dr_test.index < dr_test.data_len:
        #The data may be ordered
        sent, mask, label, sent_len = next(dr_test.get_elmo_samples())
        if config.if_gpu: 
            sent, mask = sent.cuda(), mask.cuda()
            label, sent_len = label.cuda(), sent_len.cuda()
        pred_label, _, _  = model.predict(sent, mask, sent_len) 
        #Record the testing label
        pred_labels.extend(pred_label.cpu().numpy())
        true_labels.extend(label.cpu().numpy())

        correct_count += sum(pred_label==label).cpu().item()
        #print(correct_count)
    if dr_test.data_len < 1:
        print('Testing Data Error')
    acc = correct_count * 1.0 / dr_test.data_len
    print(confusion_matrix(true_labels, pred_labels))
    print("Sentiment Accuray {0}, {1}:{2}".format(acc, correct_count, all_counter))
    return acc


if __name__ == "__main__":
    train()