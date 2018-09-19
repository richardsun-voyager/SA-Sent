from collections import namedtuple, defaultdict
import codecs
from config import config
from bs4 import BeautifulSoup
import pdb
import torch
#import tokenizer
import numpy as np
import re
import pickle
import random
import os
##Added by Richard Sun
from allennlp.modules.elmo import Elmo, batch_to_ids
import en_core_web_sm
nlp = en_core_web_sm.load()

options_file = "../data/Elmo/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "../data/Elmo/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
elmo = Elmo(options_file, weight_file, 2, dropout=0)

SentInst = namedtuple("SentenceInstance", "id text text_ids text_inds opinions")
OpinionInst = namedtuple("OpinionInstance", "target_text polarity class_ind target_mask")

class dataHelper():
    def __init__(self, config):
        '''
        This class is able to:
        1. Load datasets
        2. Split sentences into words
        3. Map words into Idx
        '''
        self.config = config

        # id map to instance
        self.id2label = ["positive", "neutral", "negative"]
        self.label2id = {v:k for k,v in enumerate(self.id2label)}

        self.UNK = "UNK"
        self.EOS = "EOS"

        # data
        self.train_data = None
        self.test_data = None
    
    def read_xml_data(self, file_name):
        '''
        Read XML data
        '''
        f = codecs.open(file_name, "r", encoding="utf-8")
        soup = BeautifulSoup(f.read(), "lxml")
        sentence_tags = soup.find_all("sentence")
        sentence_list = []
        for sent_tag in sentence_tags:
            sent_id = sent_tag.attrs["id"]
            sent_text = sent_tag.find("text").contents[0]
            opinion_list = []
            try:
                asp_tag = sent_tag.find_all("aspectterms")[0]
            except:
                # print "{0} {1} has no opinions".format(sent_id, sent_text)
                #print(sent_tag)
                continue
            opinion_tags = asp_tag.find_all("aspectterm")
            for opinion_tag in opinion_tags:
                term = opinion_tag.attrs["term"]
                if term not in sent_text: pdb.set_trace()
                polarity = opinion_tag.attrs["polarity"]
                opinion_inst = OpinionInst(term, polarity, None, None)
                opinion_list.append(opinion_inst)
            sent_Inst = SentInst(sent_id, sent_text, None, opinion_list)
            sentence_list.append(sent_Inst)

        return sentence_list


    def tokenize(self, sent_str):
        '''
        Split a sentence into tokens
        '''
        # return word_tokenize(sent_str)
        sent_str = " ".join(sent_str.split("-"))
        sent_str = " ".join(sent_str.split("/"))
        sent_str = " ".join(sent_str.split("!"))
        sent = nlp(sent_str)
        return [item.text for item in sent]
        
    # namedtuple is protected!
    def process_raw_data(self, data):
        '''
        Tokenize each sentence, compute aspect mask for each sentence
        '''
        sent_len = len(data)
        #print('Sentences Num:', sent_len)
        for sent_i in np.arange(sent_len):
            sent_inst = data[sent_i]
            sent_tokens = self.tokenize(sent_inst.text)
            sent_inst = sent_inst._replace(text_inds = sent_tokens)
            opinion_list = []
            opi_len = len(sent_inst.opinions)
            for opi_i in np.arange(opi_len):
                opi_inst = sent_inst.opinions[opi_i]

                target = opi_inst.target_text
                target_tokens = self.tokenize(target)
                try:
                    target_start = sent_tokens.index(target_tokens[0])
                    target_end = sent_tokens[max(0, target_start - 1):].index(target_tokens[-1])  + max(0, target_start - 1)
                except:
                    #pdb.set_trace()
                    print('Target error '+target_tokens[0])
                    continue
                    
                if target_start < 0 or target_end < 0:
                    #pdb.set_trace()
                    print('Traget not in the vocabulary')
                    continue
                    
                mask = [0] * len(sent_tokens)
                for m_i in range(target_start, target_end + 1):
                    mask[m_i] = 1

                label = opi_inst.polarity
                if label == "conflict":  continue  # ignore conflict ones
                opi_inst = opi_inst._replace(class_ind = self.label2id[label])
                opi_inst = opi_inst._replace(target_mask = mask)
                opinion_list.append(opi_inst)
            
            sent_inst = sent_inst._replace(opinions = opinion_list)
            
            data[sent_i] = sent_inst
        return data

    
    def read(self, train_data):
        '''
        Preprocess the data
        '''
        self.train_data = self.read_xml_data(train_data)
        #self.test_data = self.read_xml_data(test_data)
        print('Dataset number:', len(self.train_data))
        #print('Testing dataset number:', len(self.test_data))
        train_data = self.process_raw_data(self.train_data)
        #test_data = self.process_raw_data(self.test_data)
        return train_data
    
        

    # shuffle and to batch size
    def to_batches(self, data, if_batch = False):
        all_triples = []
        # list of list
        pair_couter = defaultdict(int)
        for sent_inst in data:
            tokens = sent_inst.text_inds
            #print(tokens)
            for opi_inst in sent_inst.opinions:
                if opi_inst.polarity is None:  continue # conflict one
                mask = opi_inst.target_mask
                polarity = opi_inst.class_ind
                if tokens is None or mask is None or polarity is None: 
                    continue
                all_triples.append([tokens, mask, polarity])
                pair_couter[polarity] += 1
                
        print(pair_couter)

        if if_batch:
            print('Shuffle')
            random.shuffle(all_triples)
            batch_n = int(len(all_triples) / self.config.batch_size + 1)
            print("{0} instances with {1} batches".format(len(all_triples), batch_n))
            ret_triples = []
            
            offset = 0
            for i in range(batch_n):
                start = self.config.batch_size * i
                end = min(self.config.batch_size * (i+1), len(all_triples) )
                ret_triples.append(all_triples[start : end])
            return ret_triples
        else:
            return all_triples


class data_reader:
    def __init__(self, config, is_training=True):
        '''
        Load dataset and create batches for training and testing
        '''
        self.is_training = is_training
        self.config = config
        self.index = 0
        self.dh = dataHelper(config)

    def read_raw_data(self, data_path):
        '''
        Reading Raw Dataset
        '''
        print('Reading Dataset....')
        data = self.dh.read(data_path)
        print('Preprocessing Dataset....')
        self.data_batch = self.dh.to_batches(data)
        self.data_len = len(self.data_batch)
        print('Preprocessing Over!')

    def save_data(self, save_path):
        '''
        Save the data in specified folder
        '''
        try:
            with open(save_path, "wb") as f:
                pickle.dump(self.data_batch,f)
            print('Saving successfully!')
        except:
            print('Saving failure!')   

    def load_data(self, load_path):
        '''
        Load the dataset
        '''
        if os.path.exists(load_path):
            with open(load_path, 'rb') as f:
                self.data_batch = pickle.load(f)
                self.data_len = len(self.data_batch)
        else:
            print('Data not exist!')    

    def split_save_data(self, train_path, valid_path):
        '''
        split dataset into training and validation parts
        '''
        np.random.shuffle(self.data_batch)
        train_num = int(self.data_len*5.0/6)
        training_batch = self.data_batch[:train_num]
        valid_batch = self.data_batch[train_num:]
        try:
            with open(train_path, "wb") as f:
                pickle.dump(training_batch,f)
            with open(valid_path, "wb") as f:
                pickle.dump(valid_batch,f)  
            print('Saving successfully!')
        except:
            print('Saving failure!')   
    
    def generate_sample(self, all_triples):
        '''
        Generate a batch of training dataset
        '''
        batch_size = self.config.batch_size
        select_index = np.random.choice(len(all_triples), batch_size)
        select_trip = [all_triples[i] for i in select_index]
        return select_trip

    

    def elmo_transform(self, triples):
        '''
        Transform sentences into elmo
        '''
        token_list, mask_list, label_list = zip(*triples)
        sent_lens = [len(tokens) for tokens in token_list]
        sent_lens = torch.LongTensor(sent_lens)
        label_list = torch.LongTensor(label_list)
        max_len = max(sent_lens)
        batch_size = len(sent_lens)
        character_ids = batch_to_ids(token_list)
        embeddings = elmo(character_ids)
        #batch_size*word_num * 1024
        sent_vecs = embeddings['elmo_representations'][0]
        #Padding the mask to same lengths
        mask_vecs = np.zeros([batch_size, max_len])
        mask_vecs = torch.LongTensor(mask_vecs)
        for i, mask in enumerate(mask_list):
            mask_vecs[i, :len(mask)] = torch.LongTensor(mask)
        return sent_vecs, mask_vecs, label_list, sent_lens

    def reset_samples(self):
        self.index = 0

    def get_samples(self):
        '''
        Generate random samples for training process
        Generate samples for testing process
        '''
        if self.is_training:
            samples = self.generate_sample(self.data_batch)
            sent_vecs, mask_vecs, label_list, sent_lens = self.elmo_transform(samples)
            #Sort the lengths, and change orders accordingly
            sent_lens, perm_idx = sent_lens.sort(0, descending=True)
            sent_vecs = sent_vecs[perm_idx]
            mask_vecs = mask_vecs[perm_idx]
            label_list = label_list[perm_idx]
        else:
            if self.index == self.data_len:
                print('Testing Over!')
            #First get batches of testing data
            if self.data_len - self.index >= config.batch_size:
                #print('Testing Sample Index:', self.index)
                start = self.index
                end = start + config.batch_size
                samples = self.data_batch[start: end]
                self.index += config.batch_size
                sent_vecs, mask_vecs, label_list, sent_lens = self.elmo_transform(samples)
                #Sort the lengths, and change orders accordingly
                sent_lens, perm_idx = sent_lens.sort(0, descending=True)
                sent_vecs = sent_vecs[perm_idx]
                mask_vecs = mask_vecs[perm_idx]
                label_list = label_list[perm_idx]
            else:#Then generate testing data one by one
                samples =  self.data_batch[self.index] 
                sent_vecs, mask_vecs, label_list, sent_lens = self.elmo_transform([samples])
                self.index += 1
        yield sent_vecs, mask_vecs, label_list, sent_lens

        
    

def read_data():
    TRAIN_DATA_PATH = "data/2014/Restaurants_Train_v2.xml"
    TEST_DATA_PATH = "data/2014/Restaurants_Test_Gold.xml"
    dr = data_reader(config)
    dr.read_raw_data(TRAIN_DATA_PATH)
    dr.split_save_data('data/2014/training.pickle', 'data/2014/valid.pickle')
    dr.read_raw_data(TEST_DATA_PATH)
    dr.save_data('data/2014/testing.pickle')


if __name__ == "__main__":
    read_data()
