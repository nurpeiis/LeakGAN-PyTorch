from functools import reduce
import numpy as np
import pickle

def text_to_tensor(filePath):
    """
        Read text from file
    """
    with open(filePath, 'r') as f:
        lines = f.readlines()
    f.close()
    corpus = []
    for l in lines:
        l = l.strip().split(' ') #strip removes blank spaces from both sides
        if len(l) < 28:
            corpus.append(l)
    """
    Get all words used in text
    """
    vocab = []
    for p in corpus:
        vocab.extend(p) #save all into a single list
    vocab = list(set(vocab)) #save only unique characters
    for i in range(len(vocab)):
        if vocab[i] == "<R>":
            break
    del vocab[i]
    vocab.append("<R>") #we only need one <R> not several

    """
    Encode text into a LongTensor
    """
    corpus_num = []
    for p in corpus:
        corpus_num.append(list(map(lambda x: vocab.index(x) + 1, p)))
    corpus_data = np.array(corpus_data)

    """
    Save preprocessed file
    """
    np.save("corpus", corpus_data) #save the training corpus data, where words are represented as numbers(their index in vocab array)
    f = open("chars.pkl", "wb") #this is in a sense table of keys
    pickle.dump(vocab, f)


def tensor_to_text(input_x, vocab):
    #vocab will convert some integer into a word
    poem = []
    sen_len = 5
    for index, x in enumerate(input_x):
        if index != 0:
            if index % sen_len == 0:
                poem.append("\n")
        poem.append(vocab[x-1]) #x-1 because when encoding we added one
    poem_ = ""
    poem_ = reduce((lambda x, y:x + y), poem) # just add strings
    print(poem_) #to see
    return poem_