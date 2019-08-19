# -*- coding: utf-8 -*-
import os
import re
import pickle
from tqdm import tqdm
PAD_token = 0  # Used for padding short sentences
UNK = 1 # unkown word or char



class Voc:
    def __init__(self):
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {UNK: "UNKOWN", PAD_token: "PAD"}
        self.num_words = 2
        self.char2index = {}
        self.char2count = {}
        self.index2char = {UNK: "UNKOWN", PAD_token: "PAD"}
        self.num_chars = 2

    def addSentence(self, pair):
        for sentence in pair:
            for word in sentence.split(' '):
                self.addWord(word)
                for char in word:
                    self.addChar(char)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def addChar(self, char):
        if char not in self.char2index:
            self.char2index[char] = self.num_words
            self.char2count[char] = 1
            self.index2char[self.num_words] = char
            self.num_chars += 1
        else:
            self.char2count[char] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD"}
        self.num_words = 1  # Count default tokens

        for word in keep_words:
            self.addWord(word)


def normalizeString(s):
    s = re.sub('\s+','',s).strip()
    return s

def readVocs(datafile):
    print("Reading lines...")
    lines = open(datafile, encoding='utf-8').read().strip().split('\n')
    # Split every line into pairs and normalize
    pairs = [[s for s in l.split('\t')] for l in lines]
    voc = Voc()
    return voc, pairs

def loadPrepareData(datafile, save_dir):
    print("Start preparing training data ...")
    voc, pairs = readVocs(datafile)
    print("Read {!s} sentence pairs".format(len(pairs)))
    print("Counting words and chars ...")
    for pair in tqdm(pairs):
        voc.addSentence(pair)
        # voc.addSentence(pair[2])
    print("Counted words:", voc.num_words)
    print("Counted chars:", voc.num_chars)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir,'word2index.pkl'),'wb') as f:
        pickle.dump(voc.word2index,f)
    with open(os.path.join(save_dir,'char2index.pkl'),'wb') as f:
        pickle.dump(voc.char2index,f)
    with open(os.path.join(save_dir, 'index2word.pkl'), 'wb') as f:
        pickle.dump(voc.index2word, f)
    with open(os.path.join(save_dir, 'index2char.pkl'), 'wb') as f:
        pickle.dump(voc.index2char, f)
    return voc, pairs


if __name__ == '__main__':
    save_dir = os.path.join('data','dictionary')
    voc, pairs = loadPrepareData('data/cut_sent.txt', save_dir)

    # for pair in pairs[:10]:
    #     print(pair)
