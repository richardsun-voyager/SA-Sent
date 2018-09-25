from sent_normalizer import text_norm
import pandas as pd
path = '../data/Indonesian/indo_tweets.csv'

def normalize(text):
    text = text_norm(text)
    return ' '.join(text)

#Multi-processing
if __name__ == '__main__':
    data = pd.read_csv(path)
    print('Normalization Starts...')
    data.Tweet = data.Tweet.map(normalize)
    data.Keyword = data.Keyword.map(normalize)
    data.to_csv('normalized_indo_tweets.csv', index=False)
    print('Normalization ends....')
    
    