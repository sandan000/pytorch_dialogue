# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from modules import MultiHeadAttention, conv, Batch_Coattention_NNsubmulti
import numpy as np

from config import Config
FLAGS = Config()

class Interaction_matching_batch(nn.Module):
    def __init__(self, emb_size, rnn_dim):
        super(Interaction_matching_batch, self).__init__()
        self.batch_coattention_nnsubmulti = Batch_Coattention_NNsubmulti(emb_size, activation=True)
        self.dropout = nn.Dropout(p = FLAGS.dropout_prob)
        self.coatt_gru = nn.GRU(emb_size, rnn_dim, batch_first=True)
        # nn.init.kaiming_normal(self.coatt_gru.all_weights)
        self.final_gru = nn.GRU(rnn_dim, rnn_dim, batch_first=True)
        # nn.init.kaiming_normal(self.final_gru.all_weights)
        self.linear = nn.Linear(rnn_dim, 2)
        # nn.init.kaiming_normal_(self.linear.weight)

    def forward(self, context_embeddings, response_embeddings, context_mask):
        res_coatt = self.batch_coattention_nnsubmulti(context_embeddings, response_embeddings, context_mask)
        res_coatt = self.dropout(res_coatt)

        self.coatt_gru.flatten_parameters()
        res_hiddens, res_final = self.coatt_gru(res_coatt)
        res_fea = torch.reshape(res_final, (-1, FLAGS.max_turn, res_final.size()[-1]))
        self.final_gru.flatten_parameters()
        _, last_hidden = self.final_gru(res_fea)
        last_hidden = torch.unbind(last_hidden, dim = 0)

        logits = self.linear(torch.cat(last_hidden, dim = -1))
        return logits





class model(nn.Module):
    def __init__(self, pretrained_word_embeddings=None, pretrained_char_embeddings=None):
        super(model, self).__init__()

        self.char_kernels = [2]
        self.char_emb = torch.nn.Embedding(FLAGS.vocab_char_size, FLAGS.char_embed_dim)
        if pretrained_char_embeddings is not None:
            self.char_emb.weight.data.copy_(torch.from_numpy(np.array(pretrained_char_embeddings)))
        self.char_conv = conv(FLAGS.char_embed_dim, FLAGS.char_hid, self.char_kernels, bias=True, activation=F.relu, isNormalize=False)#.cuda()
        self.char_dropout = nn.Dropout(FLAGS.dropout_prob).cuda()
        self.utt_char_conv = conv(FLAGS.char_embed_dim, FLAGS.char_hid, self.char_kernels, bias=True, activation=F.relu, isNormalize=False)#.cuda()
        self.char_interaction_matching_batch = Interaction_matching_batch(FLAGS.char_hid * len(self.char_kernels), FLAGS.char_hid)#.cuda()

        # word
        self.word_embeddings = nn.Embedding(FLAGS.vocab_size, FLAGS.embed_dim)#.cuda()
        if pretrained_word_embeddings is not None:
            self.word_embeddings.weight.data.copy_(torch.from_numpy(np.array(pretrained_word_embeddings)))
        self.word_context_dropout = nn.Dropout(FLAGS.dropout_prob)#.cuda()
        self.word_response_dropout = nn.Dropout(FLAGS.dropout_prob)#.cuda()
        self.word_interaction_matching_batch = Interaction_matching_batch(FLAGS.embed_dim, FLAGS.embed_dim)#.cuda()

        # seq
        self.sentence_gru_cell = nn.GRU(FLAGS.embed_dim, FLAGS.rnn_dim).cuda()
        self.seq_interaction_matching_batch = Interaction_matching_batch(FLAGS.rnn_dim, FLAGS.rnn_dim)#.cuda()

        # conv
        conv_dim = 50
        self.conv_kernels = [1, 2, 3]
        self.conv = conv(FLAGS.embed_dim, conv_dim, kernel_size=self.conv_kernels, bias=True, activation=F.relu, isNormalize=True)#.cuda()
        self.conv_interaction_matching_batch = Interaction_matching_batch(conv_dim * len(self.conv_kernels), conv_dim * len(self.conv_kernels))#.cuda()

        # self attention
        self.self_multihead_attention = MultiHeadAttention(FLAGS.embed_dim, 12)#.cuda()
        self.self_interaction_matching_batch = Interaction_matching_batch(FLAGS.embed_dim, FLAGS.embed_dim)#.cuda()

        # cross attention
        self.cross_multihead_attention = MultiHeadAttention(FLAGS.embed_dim, 12)#.cuda()
        self.cross_interaction_matching_batch = Interaction_matching_batch(FLAGS.embed_dim, FLAGS.embed_dim)#.cuda()



    def forward(self, context, context_mask,
              char_context,
              response,
              char_response):

        self.y_pred = []
        parall_context_mask = context_mask.view(-1, FLAGS.max_word_len)
        # expand_response_len = torch.unsqueeze(response_len, 1).repeat(1, FLAGS.max_turn).view([-1])
        # expand_response_mask = torch.unsqueeze(response_mask, 1).repeat(1, FLAGS.max_turn, 1).view([-1, FLAGS.max_word_len])

        # parall_context_len = context_len.view([-1])

        # all_utterance_mask = torch.unbind(content_mask, dim=1)
        # concat_context_mask = torch.cat(all_utterance_mask, dim = 1)

        if FLAGS.use_char:
            conv_dim = FLAGS.char_hid
            # kernels = [2]
            response_char_embeddings = self.char_emb(char_response.long()).view(-1, FLAGS.max_char_len, FLAGS.char_embed_dim)
            response_char_embeddings = response_char_embeddings.permute(0,2,1)
            response_char_embeddings = self.char_conv(response_char_embeddings)
            # response_char_embeddings = response_char_embeddings.permute(0,2,1)
            response_char_embeddings,_ = torch.max(response_char_embeddings, dim=1)
            response_char_embeddings = response_char_embeddings.view(-1, FLAGS.max_word_len, list(response_char_embeddings.size())[-1])
            response_char_embeddings = self.char_dropout(response_char_embeddings)

            expand_response_char_embeddings = torch.unsqueeze(response_char_embeddings,1).repeat(1,FLAGS.max_turn, 1, 1).view(-1, FLAGS.max_word_len, conv_dim)

            context_char_embeddings = self.char_emb(char_context.long())
            cont_char_embeddings = []
            for k, utt_char_emb in enumerate(torch.unbind(context_char_embeddings, dim=1)):
                utt_char_embeddings = torch.reshape(utt_char_emb, (-1, FLAGS.max_char_len, FLAGS.char_embed_dim))
                utt_char_embeddings = utt_char_embeddings.permute(0, 2, 1)
                utt_char_embeddings = self.utt_char_conv(utt_char_embeddings)
                # utt_char_embeddings = utt_char_embeddings.permute(0, 2, 1)
                utt_char_embeddings, _ = torch.max(utt_char_embeddings, dim=1)
                utt_char_embeddings = torch.reshape(utt_char_embeddings, (-1, FLAGS.max_word_len, list(utt_char_embeddings.size())[-1]))
                cont_char_embeddings.append(utt_char_embeddings)
            context_char_embeddings = torch.stack(cont_char_embeddings, dim=1)
            parall_context_char_embeddings = torch.reshape(context_char_embeddings, (-1, FLAGS.max_word_len, conv_dim * len(self.char_kernels)))

            char_interaction = self.char_interaction_matching_batch(parall_context_char_embeddings, expand_response_char_embeddings, parall_context_mask)
            self.y_pred.append(char_interaction)

        context_embeddings = self.word_embeddings(context.long())
        response_embeddings = self.word_embeddings(response.long())
        context_embeddings = self.word_context_dropout(context_embeddings)
        response_embeddings = self.word_response_dropout(response_embeddings)
        parall_context_embeddings = context_embeddings.view(-1, FLAGS.max_word_len, FLAGS.embed_dim)
        all_utterance_embeddings = torch.unbind(context_embeddings, dim=1)
        expand_response_embeddings = torch.unsqueeze(response_embeddings, 1).repeat(1, FLAGS.max_turn, 1, 1).view(-1, FLAGS.max_word_len, FLAGS.embed_dim)

        if FLAGS.use_word:
            word_interaction = self.word_interaction_matching_batch(parall_context_embeddings, expand_response_embeddings, parall_context_mask)
            self.y_pred.append(word_interaction)

        if FLAGS.use_seq:
            # response_embeddings = pack_padded_sequence(response_embeddings, lengths=response_len, batch_first=True)
            # parall_context_embeddings = pack_padded_sequence(parall_context_embeddings, lengths=parall_context_len, batch_first=True)
            self.sentence_gru_cell.flatten_parameters()
            response_GRU_embeddings, _ = self.sentence_gru_cell(response_embeddings)
            self.sentence_gru_cell.flatten_parameters()
            context_GRU_embeddings, _ = self.sentence_gru_cell(parall_context_embeddings)
            expand_response_GRU_embeddings = torch.unsqueeze(response_GRU_embeddings, 1).repeat(1, FLAGS.max_turn, 1, 1).view(-1, FLAGS.max_word_len, FLAGS.rnn_dim)
            seq_interaction = self.seq_interaction_matching_batch(context_GRU_embeddings, expand_response_GRU_embeddings, parall_context_mask)
            self.y_pred.append(seq_interaction)

        if FLAGS.use_conv:
            conv_dim = 50
            # kernels = [1, 2, 3]
            response_embeddings_permute = response_embeddings.permute(0, 2, 1)
            response_conv_embeddings = self.conv(response_embeddings_permute)
            # response_conv_embeddings = response_conv_embeddings.permute(0, 2, 1)
            parall_context_embeddings_permute = parall_context_embeddings.permute(0, 2, 1)
            context_conv_embeddings = self.conv(parall_context_embeddings_permute)
            # context_conv_embeddings = context_conv_embeddings.permute(0, 2, 1)
            expand_response_conv_embeddings = torch.unsqueeze(response_conv_embeddings, 1).repeat(1, FLAGS.max_turn, 1, 1).view(-1, FLAGS.max_word_len, conv_dim*len(self.conv_kernels))
            conv_interaction = self.conv_interaction_matching_batch(context_conv_embeddings, expand_response_conv_embeddings, parall_context_mask)
            self.y_pred.append(conv_interaction)

        if FLAGS.use_self:
            response_self_att_embeddings = self.self_multihead_attention(response_embeddings, response_embeddings)
            context_self_att_embeddings = self.self_multihead_attention(parall_context_embeddings, parall_context_embeddings)
            expand_response_self_att_embeddings = torch.unsqueeze(response_self_att_embeddings, 1).repeat(1, FLAGS.max_turn, 1, 1)
            expand_response_self_att_embeddings = expand_response_self_att_embeddings.view(-1, FLAGS.max_word_len, FLAGS.embed_dim)
            self_att_interaction = self.self_interaction_matching_batch(context_self_att_embeddings, expand_response_self_att_embeddings, parall_context_mask)
            self.y_pred.append(self_att_interaction)

        if FLAGS.use_cross:
            expand_response_embeddings = torch.unsqueeze(response_embeddings, 1).repeat(1, FLAGS.max_turn, 1, 1).view(-1, FLAGS.max_word_len, FLAGS.embed_dim)
            context_cross_att_embeddings = self.cross_multihead_attention(parall_context_embeddings, expand_response_embeddings)
            response_cross_att_embeddings = []
            for k, utterance_embeddings in enumerate(all_utterance_embeddings):
                response_cross_att_embedding = self.cross_multihead_attention(response_embeddings, utterance_embeddings)
                response_cross_att_embeddings.append(response_cross_att_embedding)
            response_cross_att_embeddings = torch.stack(response_cross_att_embeddings, dim=1)
            response_cross_att_embeddings = torch.reshape(response_cross_att_embeddings,(-1, FLAGS.max_word_len, FLAGS.embed_dim))
            cross_att_interaction = self.cross_interaction_matching_batch(context_cross_att_embeddings, response_cross_att_embeddings, parall_context_mask)
            self.y_pred.append(cross_att_interaction)

        return self.y_pred
























