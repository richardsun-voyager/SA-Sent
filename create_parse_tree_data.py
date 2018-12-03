from data_reader_general import data_reader, data_generator
from backup.configs.config_crf_glove import config
import re
from stanfordcorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP(r'../data/stanford-corenlp-full-2018-02-27', memory='8g',timeout=3000)

def create_tree(dataset, file):
    print('Creating file ', file)
    with open(file, 'w') as f:
        for item in dataset:
            tree = nlp.parse(item[4])
            tree = re.sub('\n', '', tree)
            line = tree + '|||' + ' '.join(item[5]) + '|||' + str(item[2]) + '|||' + item[4] + '\n'
            f.write(line)
            
    

def main():
    dr = data_reader(config)
    train_data = dr.load_data(config.train_path)
    valid_data = dr.load_data(config.valid_path)
    test_data = dr.load_data(config.test_path)
    print("Training Samples: {}".format(len(train_data)))
    print("Validating Samples: {}".format(len(valid_data)))
    print("Testing Samples: {}".format(len(test_data)))

    train_file = 'data/parse_trees/res_train_trees.txt'
    dev_file = 'data/parse_trees/res_dev_trees.txt'
    test_file = 'data/parse_trees/res_test_trees.txt'
    
    create_tree(train_data, train_file)
    create_tree(valid_data, dev_file)
    create_tree(test_data, test_file)
    print('Processing over')
    
if __name__ == "__main__":
    main()
    