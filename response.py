import torch
import torch.functional as F
from torch.utils.data import DataLoader,Dataset
from time import time
import numpy as np
import os
import re
import pickle
import jieba
from elasticsearch import Elasticsearch

from process_data import get_char_word_idx_from_sent,get_char_word_idx_from_sent_msg
from config import Config
FLAGS = Config()
from model import model
torch.backends.cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = '6'

es = Elasticsearch(['172.18.0.23'])

mrfn = model()
mrfn.cuda()
mrfn.load_state_dict(torch.load('checkpoint_single/model_212427_single_gpu.ckpt'))
print('load model done!')
#print(mrfn.state_dict().keys())

word2idx = pickle.load(open(FLAGS.word_dict_path, 'rb'))
char2idx = pickle.load(open(FLAGS.char_dict_path, 'rb'))

class ResponseDataset(Dataset):
    # Initialize data.
    def __init__(self, context, responses):
        self.context = context
        self.responses = responses

    def __getitem__(self, index):
        return get_data(self.context, self.responses[index])

    def __len__(self):
        return len(self.responses)

def get_data(context, response):
        context, content_mask, char_context = \
            get_char_word_idx_from_sent_msg(context, word2idx, char2idx, FLAGS.max_turn, FLAGS.max_word_len,
                                            FLAGS.max_char_len)
        response, char_response = \
            get_char_word_idx_from_sent(response, word2idx, char2idx, FLAGS.max_word_len, FLAGS.max_char_len)

        context = np.reshape(context, [FLAGS.max_turn, FLAGS.max_word_len])
        content_mask = np.reshape(content_mask, [FLAGS.max_turn, FLAGS.max_word_len])
        response = np.reshape(response, [FLAGS.max_word_len])
        char_context = np.reshape(char_context, [FLAGS.max_turn, FLAGS.max_word_len, FLAGS.max_char_len])
        char_response = np.reshape(char_response, [FLAGS.max_word_len, FLAGS.max_char_len])

        return (context, content_mask, char_context, response, char_response)


def make_variable(tuple):
    return (t.float().cuda() for t in tuple)

def test(data_loader):
    for i, input in enumerate(data_loader):
        context, content_mask, \
        char_context, \
        response, \
        char_response = make_variable(input)

        mrfn.eval()
        pred = mrfn(context, content_mask,
                    char_context,
                    response,
                    char_response)
        return pred

def predict(context):
    if "你好你好" in context[:-1]:
        context.remove("你好你好")
        if "你好" in context[:-1]:
            context.remove("你好")

    condidate_size = 128

    dsl = {
        'query': {
            "bool":{
                "must": [
                    {'match': {"context": context[-1]}}
                ],
                "should":[
                    {"match": {"context": ' '.join(context[:-1])}}
                ]
            }
        }
    }

    result = es.search(index='dialogue', body=dsl, size=condidate_size)
    responses = list(map(lambda x:' '.join(jieba.cut(x['_source']['response'])), result['hits']['hits']))
    context = [' '.join(jieba.cut(c)) for c in context]
    # cont = ''
    # for c in context:
    #     cont += ' '.join(jieba.cut(c))

    dataset = ResponseDataset(context, responses)
    data_loader = DataLoader(dataset, batch_size=condidate_size, pin_memory=True)
    pred = test(data_loader)
    y_pred = np.sum(pred, axis=0).cpu().detach().numpy()
    res_idx = np.argmax(y_pred, axis = 0)[-1]

    return re.sub('\s+', '', responses[res_idx])

def response():
    context = []
    while True:
        inp = input("context：")
        if inp == 'END': break

        context.append(inp)
        start_time = time()
        response = predict(context)
        end_time = time()
        print("response：" + response)

        print(end_time-start_time)
        context.append(response)

if __name__ == "__main__":
    # print(predict(["你好","你好你好","你在干嘛？"]))
    response()