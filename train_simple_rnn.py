#!/usr/bin/python
################Train a simple attention model, context word embedding and target word embedding not concatenated########
from __future__ import division
from model_simple_rnn import *
from data_reader_general import *
from configs.config import config
import pickle
from Layer import GloveMaskCat
import numpy as np
import codecs
import copy
import os
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
torch.manual_seed(222)
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




id2label = ["positive", "neutral", "negative"]
#Load concatenation layer and attention model layer
cat_layer = GloveMaskCat(config)

###Run a simple attention model
def train():
    print(config)
    best_acc = 0
    best_model = None

        ###Indoneisan
    dr = data_reader(config)
    #dr.read_raw_data('data/2014/')

    #train_data = dr.load_data(config.data_path+'Restaurants_Train_v2.xml.pkl')
    # valid_data = dr.load_data(config.data_path+'valid_restaurant.pkl')
    #dr.split_save_data(config.train_path, config.valid_path)
    train_data = dr.load_data(config.train_path)
    valid_data = dr.load_data(config.valid_path)
    test_data = dr.load_data(config.data_path+'Restaurants_Test_Gold.xml.pkl')
    print('Training Samples:', len(train_data))
    print('Validating Samples:', len(valid_data))
    print('Testing Samples:', len(test_data))
    

    dg_train = data_generator(config, train_data)
    dg_valid =data_generator(config, valid_data, False)
    dg_test =data_generator(config, test_data, False)

    

    cat_layer.load_vector()
    


    model = attTSA(config)


        # ###Bailin
    if config.if_gpu: model = model.cuda()
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    # pdb.set_trace()
    optimizer = create_opt(parameters, config)

    with open(config.log_path+'simple_glove_log.txt', 'w') as f:
        f.write('Start Experiment\n')

    loops = int(dg_train.data_len/config.batch_size)
    for e_ in range(config.epoch):
        print("Epoch ", e_ + 1)
        model.train()
        if e_ % config.adjust_every == 0:  
            adjust_learning_rate(optimizer, e_)

        for _ in np.arange(loops):
            model.zero_grad() 
            sent_vecs, mask_vecs, label_list, sent_lens, _ = next(dg_train.get_ids_samples())
            sent_vecs, target_avg = cat_layer(sent_vecs, mask_vecs)#Batch_size*max_len*(2*emb_size)
            if config.if_gpu: 
                sent_vecs, mask_vecs = sent_vecs.cuda(), mask_vecs.cuda()
                label_list, sent_lens = label_list.cuda(), sent_lens.cuda()

            cls_loss = model(sent_vecs, mask_vecs, label_list, sent_lens)
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
        with open(config.log_path+'simple_glove_log.txt', 'a') as f:
            f.write('Epoch '+str(e_)+'\n')
            f.write('Validation accuracy:'+str(valid_acc)+'\n')
            if e_ % 1 == 0:
                print('Testing....')
                test_acc = evaluate_test(dg_test, model)
                f.write('Testing accuracy:'+str(test_acc)+'\n')


        if valid_acc > best_acc: 
            best_acc = valid_acc
            best_model = copy.deepcopy(model)
            torch.save(best_model, config.model_path+'simple_glove_model.pt')





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
    true_labels = [item[2] for item in dr_test.data_batch]
    pred_labels = []
    while dr_test.index < dr_test.data_len:
        sent, mask, label, sent_len, _ = next(dr_test.get_ids_samples())
        sent, target = cat_layer(sent, mask)
        if config.if_gpu: 
            sent, mask = sent.cuda(), mask.cuda()
            label, sent_len = label.cuda(), sent_len.cuda()
        pred_label, _  = model.predict(sent, mask, sent_len) 
        #Record the testing label
        pred_labels.extend(pred_label.numpy())

        correct_count += sum(pred_label==label).item()
    if dr_test.data_len < 1:
        print('Testing Data Error')
    acc = correct_count * 1.0 / dr_test.data_len
    print(confusion_matrix(true_labels, pred_labels))
    print("Sentiment Accuray {0}, {1}:{2}".format(acc, correct_count, all_counter))
    return acc

def generate_simple_logits(dr_test, model):
    print("Evaluting")
    dr_test.reset_samples()
    model.eval()
    logits_record = []
    while dr_test.index < dr_test.data_len:
        sent, mask, label, sent_len, _ = next(dr_test.get_ids_samples())
        sent, target = cat_layer(sent, mask)
        if config.if_gpu: 
            sent, mask = sent.cuda(), mask.cuda()
            label, sent_len = label.cuda(), sent_len.cuda()
        _, logits  = model.predict(sent, mask, sent_len) 
        logits_record.append(logits)
    logits_record = torch.cat(logits_record, 0)
    return logits_record

if __name__ == "__main__":
    train()