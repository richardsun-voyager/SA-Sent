from __future__ import division

class Config():
    def __init__(self):
        # for reader
        self.batch_size = 32

        #Map words to lower case
        self.embed_num = 5120
        #For elmo the emb_size is 1024
        self.embed_dim = 100#1024#300
        self.mask_dim = 50
        self.if_update_embed = True

        # lstm
        #self.l_hidden_size = 256
        self.l_hidden_size = 512#elmo
        self.l_num_layers = 2 # forward and backward
        self.l_dropout = 0.1

        # penlaty
        self.C1 = 0.1
        self.C2 = 0.001
        self.if_reset = True

        self.opt = "Adam"
        self.dropout = 0.5
        self.epoch = 30
        self.lr = 0.01/ self.batch_size
        self.l2 = 0.0
        self.adjust_every = 8
        self.clip_norm = 3
        self.k_fold = 6

        # data processing
        self.if_replace = False

        # traning
        self.if_gpu = False
        # self.data_path = "data/2014/data.pkl"
        # self.dic_path = "data/2014/dic.pkl"

        #################Restaurant
        self.pretrained_embed_path = "../data/glove_indo.100d.txt"
        self.embed_path = "data/indo/vocab/local_emb.pkl"
        self.data_path = "data/indo/"
        self.train_path = "data/indo/train.pkl"
        self.valid_path = "data/indo/valid.pkl"
        self.test_path = "data/indo/test.pkl"
        self.dic_path = "data/indo/vocab/dict.pkl"


        self.model_path = "data/models/"
        self.log_path = "data/logs/"
        
        self.is_stanford_nlp = False
        self.elmo_config_file = "../data/Indo_Elmo/options.json"
        self.elmo_weight_file = "../data/Indo_Elmo/weights.hdf5"

    
    def __repr__(self):
        return str(vars(self))

config = Config()