# -*- coding: utf-8 -*-
import torch
from torch import nn
import math
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
# initializer = lambda: tf.contrib.layers.variance_scaling_initializer(factor=1.0,
#                                                              mode='FAN_AVG',
#                                                              uniform=True,
#                                                              dtype=tf.float32)
# initializer_relu = lambda: tf.contrib.layers.variance_scaling_initializer(factor=2.0,
#                                                              mode='FAN_IN',
#                                                              uniform=False,
#                                                              dtype=tf.float32)
# regularizer = tf.contrib.layers.l2_regularizer(scale = 3e-7)


def masked_softmax(scores, mask):
    numerator = torch.exp(torch.add(scores, torch.neg(torch.max(scores, dim=2, keepdim=True)[0])))
    numerator = torch.clamp(numerator, 1e-5, 1e+5)
    numerator = numerator * torch.unsqueeze(mask, dim=1)
    denominator = torch.sum(numerator, dim=2, keepdim=True)
    weights = torch.div(numerator + 1e-5 / list(mask.size())[-1], denominator + 1e-5)
    return weights


class Normalize(nn.Module):
    def __init__(self, params_shape):
        super(Normalize, self).__init__()
        self.beta = nn.Parameter(torch.zeros(params_shape))
        self.gamma = nn.Parameter(torch.zeros(params_shape))
    def forward(self, x, epsilon = 1e-5):
        mean = torch.mean(x, dim=-1, keepdim=True)
        variance = torch.var(x, dim=-1, keepdim=True)
        #variance =  torch.where(torch.isnan(variance), torch.zeros_like(variance), variance)

        normalized = (x - mean) / ((variance + epsilon) ** (.5))
        output = self.gamma * normalized + self.beta
        return output


class conv(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size, bias=False, activation=None, isNormalize=False):
        super(conv, self).__init__()
        # self.convs = []
        self.activation = activation
        self.isNormalize = isNormalize
        self.kernel_size = kernel_size
        if self.isNormalize:
            self.normalize = Normalize([channel_out * len(kernel_size)])
        if len(self.kernel_size) == 1:
            self.conv1 = nn.Conv1d(channel_in, channel_out, self.kernel_size[0], stride=1, bias=bias)
            nn.init.xavier_uniform_(self.conv1.weight)
        else:
            self.conv1 = nn.Conv1d(channel_in, channel_out, self.kernel_size[0], stride=1, bias=bias)
            nn.init.xavier_uniform_(self.conv1.weight)
            self.conv2 = nn.Conv1d(channel_in, channel_out, self.kernel_size[1], stride=1, bias=bias)
            nn.init.xavier_uniform_(self.conv2.weight)
            self.conv3 = nn.Conv1d(channel_in, channel_out, self.kernel_size[2], stride=1, bias=bias)
            nn.init.xavier_uniform_(self.conv3.weight)
        # for k in kernel_size:
            # pad =
            # self.convs.append(nn.Conv1d(channel_in, channel_out, k, stride=1, bias=bias).cuda())
            # nn.init.xavier_uniform_(self.convs[-1].weight)

    def forward(self, x):
        conv_features = []
        convs = [self.conv1]
        if len(self.kernel_size) > 1:
            convs.append(self.conv2)
            convs.append(self.conv3)
        for i,conv in enumerate(convs):
            if self.kernel_size[i]-1 != 0:
                input = F.pad(x, [0, self.kernel_size[i]-1], mode='reflect')
            else:
                input = x
            feature = conv(input)
            if self.activation != None:
                feature = self.activation(feature)
            conv_features.append(feature.permute(0, 2, 1))
        output = torch.cat(conv_features, dim = -1)
        if self.isNormalize:
            output = self.normalize(output, 1e-5)
        return output



class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        epsilon = 1e-5
        attn = torch.matmul(q, k.permute(0, 2, 1))
        attn = attn / (self.temperature + epsilon)

        if mask is not None:
            exps = torch.exp(attn)
            exps = torch.clamp(exps, 1e-5, 1e+5)
            masked_exps = exps * mask.float()
            masked_sums = masked_exps.sum(dim = -1, keepdim=True) + epsilon
            attn = masked_exps/masked_sums
        else:
            exps = torch.exp(attn)
            exps_sum = exps.sum(dim=-1, keepdim=True) + epsilon
            attn = exps/exps_sum
            # paddings = torch.ones_like(attn) * float("-inf")
            # attn = torch.where(torch.eq(mask, 0), paddings, attn)

        # attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)

        return output


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, d_model, n_head = 12, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head

        self.w_qs = nn.Linear(d_model, d_model, bias=True)
        self.w_ks = nn.Linear(d_model, d_model, bias=True)
        self.w_vs = nn.Linear(d_model, d_model, bias=True)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_model)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_model)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_model)))

        self.q_activate = F.relu
        self.k_activate = F.relu
        self.v_activate = F.relu

        self.softmax = nn.Softmax(dim=-1)
        # self.attention = ScaledDotProductAttention(temperature=np.power(d_model, 0.5))
        # self.layer_norm = nn.LayerNorm(d_model)

        # self.fc = nn.Linear(d_model, d_model)
        # nn.init.xavier_normal_(self.fc.weight)

        # self.dropout = nn.Dropout(dropout)
        self.normalize = Normalize([d_model])


    def forward(self, query, key):

        n_head = self.n_head

        # value = key
        sz_b, len_q, q_dim = query.size()
        sz_b, len_k, k_dim = key.size()
        sz_b, len_v, v_dim = key.size()

        d_k = k_dim // n_head
        d_v = v_dim // n_head

        Q = self.q_activate(self.w_qs(query))
        K = self.k_activate(self.w_ks(key))
        V = self.v_activate(self.w_vs(key))

        Q_ = torch.cat(torch.chunk(Q, n_head, dim=2), dim=0) # (h*N, T_q, C/h)
        K_ = torch.cat(torch.chunk(K, n_head, dim=2), dim=0) # (h*N, T_k, C/h)
        V_ = torch.cat(torch.chunk(V, n_head, dim=2), dim=0) # (h*N, T_k, C/h)

        outputs = torch.matmul(Q_, K_.permute(0, 2, 1))  # (h*N, T_q, T_k)

        outputs = outputs / np.power(list(K_.size())[-1], 0.5)

        key_masks = torch.sign(torch.abs(torch.sum(key, dim=-1)))#, requires_grad=False)  # (N, T_k)
        key_masks = key_masks.repeat(n_head, 1)  # (h*N, T_k)
        key_masks = torch.unsqueeze(key_masks, 1).repeat(1, len_q, 1)  # (h*N, T_q, T_k)

        paddings = torch.ones_like(outputs) * (-2 ** 32 + 1)
        output = torch.where(torch.eq(key_masks, 0), paddings, outputs.clone()) # (h*N, T_q, T_k)

        output = self.softmax(output)  # (h*N, T_q, T_k)

        query_masks = torch.sign(torch.abs(torch.sum(query, dim=-1)))#, requires_grad=False)
        query_masks = query_masks.repeat(n_head, 1)
        query_masks = torch.unsqueeze(query_masks, -1).repeat(1, 1, list(key.size())[1])
        output = output * query_masks

        output = torch.matmul(output, V_)

        output = torch.cat(torch.chunk(output, n_head, dim=0), dim=2)
        output = output + query

        output = self.normalize(output)

        return output


        # residual = q

        # mask = Variable(torch.sign(torch.abs(torch.sum(key, dim = -1))), requires_grad=False) # (N, T_k)
        # mask = mask.repeat(n_head, 1) # (h*N, T_k)
        # mask = torch.unsqueeze(mask, 1).repeat(1, len_q, 1) # (h*N, T_q, T_k)
        #
        # q = F.relu(self.w_qs(query).view(sz_b, len_q, n_head, d_k))
        # k = F.relu(self.w_ks(key).view(sz_b, len_k, n_head, d_k))
        # v = F.relu(self.w_vs(key).view(sz_b, len_v, n_head, d_v))
        #
        # q = torch.reshape(q.permute(2, 0, 1, 3), (-1, len_q, d_k)) # (n*b) x lq x dk
        # k = torch.reshape(k.permute(2, 0, 1, 3), (-1, len_k, d_k)) # (n*b) x lk x dk
        # v = torch.reshape(v.permute(2, 0, 1, 3),(-1, len_v, d_v))# (n*b) x lv x dv
        #
        # output = self.attention(q, k, v, mask=mask)
        #
        # output = output.view(n_head, sz_b, len_q, d_v)
        # output = torch.reshape(output.permute(1, 2, 0, 3), (sz_b, len_q, -1)) # b x lq x (n*dv)
        #
        # # output = self.dropout(self.fc(output))
        # output = self.normalize(output + query)



class Batch_Coattention_NNsubmulti(nn.Module):
    def __init__(self, emb_size, bias=True, activation=None):
        super(Batch_Coattention_NNsubmulti, self).__init__()
        dim = emb_size
        self.weight = nn.Parameter(torch.Tensor(dim,dim))
        nn.init.xavier_uniform_(self.weight)
        self.linear = nn.Linear(2 * emb_size, dim, bias)
        nn.init.kaiming_normal_(self.linear.weight, mode='fan_out', nonlinearity='relu')
        if activation is not None:
            self.activation = F.relu
    def forward(self, utterance, response, utterance_mask):
        e_utterance = torch.einsum('aij,jk->aik', utterance, self.weight)
        a_matrix = torch.matmul(response, e_utterance.permute(0,2,1))
        reponse_atten = torch.matmul(masked_softmax(a_matrix, utterance_mask), utterance)
        feature_mul = reponse_atten * response
        feature_sub = reponse_atten - response
        input = torch.cat([feature_mul, feature_sub], dim=-1)
        # weight = torch.Tensor(list(input.size())[-1], )
        feature_last = F.relu(self.linear(input))
        return feature_last



