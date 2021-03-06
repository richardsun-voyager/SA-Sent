#!/usr/bin/python
from __future__ import division
from model_rnn_gcnn_glove_parsing import *
from data_reader_general import *
from configs.config_rnn_gcnn import config
from torch import optim
from parse_path import constituency_path
import pickle
from Layer import GloveMaskCat
import numpy as np
import codecs
import copy
import os
from sklearn.metrics import confusion_matrix
torch.manual_seed(222)
cp = constituency_path()
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

def convert_mask_index(masks):
    '''
    Find the indice of none zeros values in masks, namely the target indice
    '''
    target_indice = []
    for mask in masks:
        indice = torch.nonzero(mask == 1).squeeze(1).numpy()
        target_indice.append(indice)
    return target_indice

def get_context_weight(tokens, targets, max_len):
    weights = np.zeros([len(tokens), max_len])
    for i, token in enumerate(tokens):
        #print('Original word num')
        #print(len(token))
        #text = ' '.join(token)#Connect them into a string
        #stanford nlp cannot identify the abbreviations ending with '.' in the sentences

        try:
            max_w, min_w, a_v = cp.proceed(token, targets[i])
            weights[i, :len(max_w)] = max_w
        except Exception as e:
            print(e)
            print(token, targets[i])
    return torch.FloatTensor(weights)


id2label = ["positive", "neutral", "negative"]
#Load concatenation layer and attention model layer
cat_layer = GloveMaskCat(config)
if config.if_gpu: cat_layer = cat_layer.cuda()


def train():
    print(config)
    best_acc = 0
    best_model = None

#     TRAIN_DATA_PATH = "data/2014/Restaurants_Train_v2.xml"
#     TEST_DATA_PATH = "data/2014/Restaurants_Test_Gold.xml"
#     path_list = [TRAIN_DATA_PATH, TEST_DATA_PATH]
#     #First time, need to preprocess and save the data
#     #Read XML file
#     dr = data_reader(config)
#     dr.read_train_test_data(path_list)
#     print('Data Preprocessed!')
    
#     dr = data_reader(config)
#     dr.load_data('data/restaurant/Restaurants_Train_v2.xml.pkl')
#     dr.split_save_data(config.train_path, config.valid_path)
#     print('Splitting finished')


    #Load preprocessed data directly
    dr = data_reader(config)
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

    model = CNN_Gate_Aspect_Text(config)


    #Create trainable parameters
    if config.if_gpu: model = model.cuda()
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = create_opt(parameters, config)

    with open(config.log_path+'rnn_gcnn_log.txt', 'w') as f:
        f.write('Start Experiment\n')

    loops = int(dg_train.data_len/config.batch_size)
    for e_ in range(config.epoch):
        print("Epoch ", e_ + 1)
        model.train()
        if e_ % config.adjust_every == 0:  
            adjust_learning_rate(optimizer, e_)

        for _ in np.arange(loops):
            optimizer.zero_grad() 
            #Generate samples
            sent_vecs, mask_vecs, label_list, sent_lens, texts = next(dg_train.get_ids_samples(True))
            sent_vecs, target_avg = cat_layer(sent_vecs, mask_vecs)#Batch_size*max_len*(2*emb_size)
            
            #Get parsing weights
            target_indice = convert_mask_index(mask_vecs)#Get target indice
            max_len = max(sent_lens).item()
            weights = get_context_weight(texts, target_indice, max_len)

            if config.if_gpu: 
                sent_vecs, mask_vecs = sent_vecs.cuda(), mask_vecs.cuda()
                label_list, sent_lens = label_list.cuda(), sent_lens.cuda()
            #Compute loss
            cls_loss = model(sent_vecs, mask_vecs, label_list, sent_lens, weights)
            l2_loss = 0
            for w in model.parameters():
                if (w is not None) and w.requires_grad:
                    l2_loss += w.norm(2)
                    
            print("cls loss {0} regularizrion loss {1}".format(cls_loss.item(), l2_loss.item()))
            cls_loss += l2_loss * 0.001#previous we tried 0.005
            cls_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_norm, norm_type=2)
            optimizer.step()

        valid_acc = evaluate_test(dg_valid, model)
        with open(config.log_path+'rnn_gcnn_log.txt', 'a') as f:
            f.write('Epoch '+str(e_)+'\n')
            f.write('Validation accuracy:'+str(valid_acc)+'\n')
            if e_ % 1 == 0:
                print('Testing....')
                test_acc = evaluate_test(dg_test, model)
                f.write('Testing accuracy:'+str(test_acc)+'\n')

        if valid_acc > best_acc: 
            best_acc = valid_acc
            best_model = copy.deepcopy(model)
            torch.save(best_model, config.model_path+'gcnn_model.pt')







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
    true_labels = []
    pred_labels = []
    while dr_test.index < dr_test.data_len:
        sent, mask, label, sent_len, tokens = next(dr_test.get_ids_samples())
        sent, target = cat_layer(sent, mask)
        
        #Get parsing weights
        target_indice = convert_mask_index(mask)#Get target indice
        max_len = max(sent_len).item()
        weights = get_context_weight(tokens, target_indice, max_len)
        
        if config.if_gpu: 
            sent, mask = sent.cuda(), mask.cuda()
            label, sent_len = label.cuda(), sent_len.cuda()
        pred_label  = model.predict(sent, mask, sent_len, weights) 
        true_labels.extend(label.cpu().numpy())
        pred_labels.extend(pred_label.cpu().numpy())
        #Count number of correct predictions
        correct_count += sum(pred_label==label).cpu().item()
        record_mislabeled_samples(pred_label, label, tokens, mask)
    if dr_test.data_len < 1:
        print('Testing Data Error')
    acc = correct_count * 1.0 / dr_test.data_len
    print(confusion_matrix(true_labels, pred_labels))
    print("Sentiment Accuray {0}, {1}:{2}".format(acc, correct_count, all_counter))
    return acc

def record_mislabeled_samples(pred, label, tokens, mask):
    file_path = 'data/mislabeled_samples/records.txt'
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            f.write('GCNN Model:\n')
#     print(pred)
#     print(label)
    indice = torch.nonzero(pred - label)
    if indice.nelement() > 0:
        indice = indice.squeeze(1)
        record_tokens = [tokens[i] for i in indice.cpu().numpy()]
        record_mask = mask[indice]
        record_pred = pred[indice]
        record_label = label[indice]
        target_indice = [torch.nonzero(item).squeeze(1).cpu().numpy() for item in record_mask]

        for i, tokens in enumerate(record_tokens):
            targets = [record_tokens[i][t] for t in target_indice[i]]
            targets = ' '.join(targets)
            with open(file_path, 'a') as f:
                f.write(targets + ' True label:' + str(record_label[i]) + ' Pred Label:'+str(record_pred[i]))
                f.write('\n')
                f.write(' '.join(tokens))
                f.write('\n')
                f.write('*********************\n')
    

if __name__ == "__main__":
    train()