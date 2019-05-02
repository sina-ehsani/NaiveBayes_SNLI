from data_utils import savedata

import json
import re
import numpy as np 
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
# from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import WordNetLemmatizer

# from nltk.corpus import stopwords
# from keras.utils import np_utils

lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()
# stops = set(stopwords.words("english"))

def extract_tokens_from_binary_parse(parse):
    return parse.replace('(', ' ').replace(')', ' ').replace('-LRB-', '(').replace('-RRB-', ')').split()

def yield_examples(fn, skip_no_majority=True, limit=None):
    for i, line in enumerate(open(fn)):
        if limit and i > limit:
            break
        data = json.loads(line)
        label = data['gold_label']
        s1 = ' '.join(extract_tokens_from_binary_parse(data['sentence1_binary_parse']))
        s2 = ' '.join(extract_tokens_from_binary_parse(data['sentence2_binary_parse']))
        if skip_no_majority and label == '-':
            continue
        yield (label, s1, s2)
        
def clean_data(sentence_list):
    '''This Fucntion will, remove ponctuations and stop-words, lower-case and stem for each sentence'''
    
    clean_sentence=[]
    for i in sentence_list:
        i = re.sub(r"[^a-zA-Z]"," ", i)
        words = i.lower().split() #Lower-case
        # words = [w for w in words if not w in stops] # Remove stop worlds
        # words = [lemmatizer.lemmatize(w) for w in words ]
        words = [ps.stem(w) for w in words ] #Stemming
        clean_sentence.append(" ".join(words))
    return(clean_sentence)
    
    
def prepro(fn, limit=None):
    '''This is the preprocessing function, will have the filename as an input, and returns both sentences and their label'''
    raw_data = list(yield_examples(fn=fn, limit=limit))
    
    left = [s1 for _, s1, s2 in raw_data]
    right = [s2 for _, s1, s2 in raw_data]
    
    left = clean_data(left)
    right = clean_data(right)

    LABELS = {'contradiction': 0, 'neutral': 1, 'entailment': 2}
    Y = np.array([LABELS[l] for l, s1, s2 in raw_data])
#     Y = np_utils.to_categorical(Y, len(LABELS))

    return left, right, Y


def main():
    train = prepro('snli_1.0/snli_1.0_train.jsonl')
    test = prepro('snli_1.0/snli_1.0_test.jsonl')
    dev = prepro('snli_1.0/snli_1.0_dev.jsonl')
    
    savedata(test, "test_stem")
    savedata(train,"train_stem")
    savedata(dev,"dev_stem")

    

if __name__ == '__main__':
    main()
        


