# -*- coding: utf-8 -*-
import os

class Config():
    def __init__(self, data_path='data'):
        self.train_file = os.path.join(data_path, 'train.txt')
        self.valid_file = os.path.join(data_path, 'valid.txt')
        self.test_file = os.path.join(data_path, 'test.txt')
        self.checkpoint_single = 'checkpoint_single'
        self.checkpoint_multi = 'checkpoint_multi'
        if not os.path.exists(self.checkpoint_single):
            os.makedirs(self.checkpoint_single)
        if not os.path.exists(self.checkpoint_multi):
            os.makedirs(self.checkpoint_multi)

        self.vocab_size = 888144  # 888143
        self.embed_dim = 300
        self.rnn_dim = 300
        self.init_dict = True
        self.init_embeddings_path = os.path.join(data_path, 'embedding/word_emb_matrix.pkl')
        self.word_dict_path = os.path.join(data_path, 'embedding/word_dict.pkl')

        self.vocab_char_size = 10981 # 10981
        self.char_embed_dim = 64
        self.char_hid = 64
        self.init_char_dict = True
        self.init_char_embeddings_path = os.path.join(data_path, 'embedding/char_emb_matrix.pkl')
        self.char_dict_path = os.path.join(data_path, 'embedding/char_dict.pkl')

        self.max_turn = 8
        self.max_word_len = 20
        self.max_char_len = 6

        self.use_char = True
        self.use_word = True
        self.use_seq = True
        self.use_conv = True
        self.use_self = True
        self.use_cross = True

        self.dropout_prob = 0.2

        self.lr = 1e-3
        self.lr_decay = False
        self.decay_rate = 0.9
        self.decay_steps = 2000
        self.lr_minimal = 0.00005

        self.batch_size = 128
        self.num_epochs = 100
        self.print_every = 1000
        self.valid_every = 10000
        self.checkpoint_every = 20000
        self.test_every = 10000
        self.write_every = 500

        self.reload_model = False
        self.log_root = 'logs_tmp/'

        self.gpu = 'gpu:0'
        self.seed = 2019

