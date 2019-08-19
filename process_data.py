# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.utils.data import Dataset
from keras.preprocessing.sequence import pad_sequences
from torch.nn.utils.rnn import pad_sequence
import numpy as np

import pickle
import codecs
from random import shuffle

from config import Config

FLAGS = Config("data/")

data_dir = "data/"

device = "cuda"

class ChatbotDataset(Dataset):
    # Initialize data.
    def __init__(self, input_file, word2idx, char2idx, isshuffle=True):
        self.word2idx = word2idx
        self.char2idx = char2idx
        self.data = load_file(input_file, isshuffle)

    def __getitem__(self, index):
        return get_data(self.data, index, self.word2idx, self.char2idx)

    def __len__(self):
        return len(self.data)



def load_file(input_file, isshuffle=True):
    # word2idx['UNK'] = len(word2idx) + 1
    # char2idx['UNK'] = len(char2idx) + 1

    revs = []
    with codecs.open(input_file, 'r', 'utf-8') as f:
        for k, line in enumerate(f):
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            label = int(parts[0])
            context = parts[1:-1]  # multi-turn
            # context = " ".join(parts[1:-1])  # single-turn
            response = parts[-1]

            data = {"y": label, "c": context, "r": response}
            revs.append(data)
    print("processed dataset with %d context-response pairs " % (len(revs)))
    if isshuffle == True:
        shuffle(revs)
    return revs

def get_char_word_idx_from_sent(sent, word_idx_map, char_idx_map, max_word_len, max_char_len):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    token_ids = [word_idx_map.get(word, 1) for word in sent.split()]
    x = pad_sequences([token_ids], padding='post', maxlen=max_word_len)[0]
    # x_mask = pad_sequences([len(token_ids) * [1]], padding='post', maxlen=max_word_len)[0]
    # x_len = min(len(token_ids), max_word_len)

    x_char = np.zeros([max_word_len, max_char_len], dtype=np.int32)
    # x_char_mask = np.zeros([max_word_len, max_char_len], dtype=np.int32)
    # x_char_len = np.zeros([max_word_len], dtype=np.int32)

    # get char index
    for i, word in enumerate(sent.split()):
        if i >= max_word_len: continue
        char_ids = [char_idx_map.get(c, 1) for c in word]
        x_char[i] = pad_sequences([char_ids], padding='post', maxlen=max_char_len)[0]
        # x_char_mask[i] = pad_sequences([len(char_ids) * [1]], padding='post', maxlen=max_char_len)[0]
        # x_char_len[i] = len(char_ids)

    return x, x_char


def get_char_word_idx_from_sent_msg(sents, word_idx_map, char_idx_map, max_turn, max_word_len, max_char_len):
    word_turns = []
    word_masks = []
    # word_lens = []

    char_turns = []
    # char_masks = []
    # char_lens = []

    for sent in sents:
        words = sent.split()
        token_ids = np.array([word_idx_map.get(word, 0) for word in words], dtype=np.int32)
        x = pad_sequences([token_ids], padding='post', maxlen=max_word_len)[0]
        x_mask = pad_sequences([len(token_ids) * [1]], padding='post', maxlen=max_word_len)

        word_turns.append(x)
        word_masks.append(x_mask)
        # word_lens.append(min(len(words), max_word_len))

        x_char = np.zeros([max_word_len, max_char_len], dtype=np.int32)
        # x_char_mask = np.zeros([max_word_len, max_char_len], dtype=np.int32)
        # x_char_len = np.zeros([max_word_len], dtype=np.int32)

        for i, word in enumerate(words):
            if i >= max_word_len: continue
            char_ids = [char_idx_map.get(c, 0) for c in word]
            x_char[i] = pad_sequences([char_ids], padding='post', maxlen=max_char_len)[0]
            # x_char_mask[i] = pad_sequences([len(char_ids) * [1]], padding='post', maxlen=max_char_len)[0]
            # x_char_len[i] = len(char_ids)

        char_turns.append(x_char)
        # char_masks.append(x_char_mask)
        # char_lens.append(x_char_len)

    word_turns_new = np.zeros([max_turn, max_word_len], dtype=np.int32)
    word_masks_new = np.zeros([max_turn, max_word_len], dtype=np.int32)
    # word_lens_new = np.zeros([max_turn], dtype=np.int32)

    char_turns_new = np.zeros([max_turn, max_word_len, max_char_len], dtype=np.int32)
    # char_masks_new = np.zeros([max_turn, max_word_len, max_char_len], dtype=np.int32)
    # char_lens_new = np.zeros([max_turn, max_word_len], dtype=np.int32)

    if len(word_turns) <= max_turn:
        word_turns_new[-len(word_turns):] = word_turns
        word_masks_new[-len(word_turns):] = word_masks
        # word_lens_new[-len(word_turns):] = word_lens

        char_turns_new[-len(word_turns):] = char_turns
        # char_masks_new[-len(word_turns):] = char_masks
        # char_lens_new[-len(word_turns):] = char_lens

    if len(word_turns) > max_turn:
        word_turns_new[:] = word_turns[len(word_turns) - max_turn:len(word_turns)]
        word_masks_new[:] = word_masks[len(word_turns) - max_turn:len(word_turns)]
        # word_lens_new[:] = word_lens[len(word_turns) - max_turn:len(word_turns)]

        char_turns_new[:] = char_turns[len(word_turns) - max_turn:len(word_turns)]
        # char_masks_new[:] = char_masks[len(word_turns) - max_turn:len(word_turns)]
        # char_lens_new[:] = char_lens[len(word_turns) - max_turn:len(word_turns)]

    return word_turns_new, word_masks_new, char_turns_new


def get_data(revs, index, word2idx, char2idx):
    # with open('data/train.pkl', 'wb') as f:
        # contexts.
        # for i,rev in revs:
        rev = revs[index]
        context, content_mask, char_context = \
            get_char_word_idx_from_sent_msg(rev["c"], word2idx, char2idx, FLAGS.max_turn, FLAGS.max_word_len,
                                            FLAGS.max_char_len)
        response, char_response = \
            get_char_word_idx_from_sent(rev['r'], word2idx, char2idx, FLAGS.max_word_len, FLAGS.max_char_len)

        y_label = rev["y"]

        context = np.reshape(context, [FLAGS.max_turn, FLAGS.max_word_len])
        content_mask = np.reshape(content_mask, [FLAGS.max_turn, FLAGS.max_word_len])
        # context_len = np.reshape(context_len, [FLAGS.max_turn])
        response = np.reshape(response, [FLAGS.max_word_len])
        # response_mask = np.reshape(response_mask, [FLAGS.max_word_len])

        char_context = np.reshape(char_context, [FLAGS.max_turn, FLAGS.max_word_len, FLAGS.max_char_len])
        # char_content_mask = np.reshape(char_content_mask, [FLAGS.max_turn, FLAGS.max_word_len, FLAGS.max_char_len])
        # char_context_len = np.reshape(char_context_len, [FLAGS.max_turn, FLAGS.max_word_len])
        char_response = np.reshape(char_response, [FLAGS.max_word_len, FLAGS.max_char_len])
        # char_response_mask = np.reshape(char_response_mask, [FLAGS.max_word_len, FLAGS.max_char_len])
        # char_response_len = np.reshape(char_response_len, [FLAGS.max_word_len])

            # contexts

        # pickle.dump((context, content_mask, context_len,
        #           char_context, char_content_mask, char_context_len,
        #           response, response_mask, response_len,
        #           char_response, char_response_mask, char_response_len, y_label),f)

        return (context, content_mask, char_context, response, char_response, y_label)



if __name__ == '__main__':
    word2idx = pickle.load(open(FLAGS.word_dict_path, 'rb'))
    char2idx = pickle.load(open(FLAGS.char_dict_path, 'rb'))
    data = load_file(FLAGS.test_file, isshuffle=True)
    with open('data/train.pkl', 'wb') as f:
        for i in range(10):
            result = get_data(data, i, word2idx, char2idx)
            pickle.dump(result,f)