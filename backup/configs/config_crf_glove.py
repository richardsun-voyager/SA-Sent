from __future__ import division

class Config():
    def __init__(self):
        # for reader
        self.batch_size = 32

        #Map words to lower case
        self.embed_num = 5091#5120
        #For elmo the emb_size is 1024
        self.embed_dim = 300#1024#300
        self.mask_dim = 30
        self.if_update_embed = True

        # lstm
        #self.l_hidden_size = 256
        self.l_hidden_size = 256#elmo
        self.l_num_layers = 2 # forward and backward
        self.l_dropout = 0.1

        # penlaty
        self.C1 = 0.1
        self.C2 = 0.1
        self.if_reset = True

        self.opt = "Adam"
        self.dropout = 0.5
        self.epoch = 30
        self.lr = 0.01
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
#         self.pretrained_embed_path = "../data/word_embeddings/glove.840B.300d.txt"
#         self.embed_path = "data/restaurant_parse/vocab/local_emb.pkl"
#         self.data_path = "data/restaurant_parse/"
#         self.train_path = "data/restaurant_parse/train.pkl"
#         self.valid_path = "data/restaurant_parse/valid.pkl"
#         self.test_path = "data/restaurant_parse/Restaurants_Test_Gold.xml.pkl"
#         self.dic_path = "data/restaurant_parse/vocab/dict.pkl"
        
        self.pretrained_embed_path = "../data/word_embeddings/glove.840B.300d.txt"
        self.embed_path = "data/laptop/vocab/local_emb.pkl"
        self.data_path = "data/laptop/"
        self.train_path = "data/laptop/train.pkl"
        self.valid_path = "data/laptop/valid.pkl"
        self.test_path = "data/laptop/Laptops_Test_Gold.xml.pkl"
        self.dic_path = "data/laptop/vocab/dict.pkl"


        self.model_path = "data/models/"
        self.log_path = "data/logs/"
        
        self.is_stanford_nlp = False
        self.elmo_config_file = '../data/Elmo/elmo_2x4096_512_2048cnn_2xhighway_options.json'
        self.elmo_weight_file = '../data/Elmo/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'

    
    def __repr__(self):
        return str(vars(self))

config = Config()