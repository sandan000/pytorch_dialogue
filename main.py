import torch
from torch import nn
from torch import optim
import torch.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from datetime import datetime
import numpy as np
import os
import pickle

from process_data import ChatbotDataset
from config import Config
from model import model

device_ids = [5]
os.environ["CUDA_VISIBLE_DEVICES"] = '5'

FLAGS = Config()
torch.backends.cudnn.benchmark = True

word2idx = pickle.load(open(FLAGS.word_dict_path, 'rb'))
char2idx = pickle.load(open(FLAGS.char_dict_path, 'rb'))

train_dataset = ChatbotDataset(FLAGS.train_file, word2idx, char2idx)
train_loader = DataLoader(dataset=train_dataset, batch_size=FLAGS.batch_size, num_workers=8, pin_memory= True, shuffle=True, drop_last=True)

valid_dataset = ChatbotDataset(FLAGS.valid_file, word2idx, char2idx)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=FLAGS.batch_size, num_workers=8, pin_memory=True, shuffle=True)

test_dataset = ChatbotDataset(FLAGS.test_file, word2idx, char2idx)
test_loader = DataLoader(dataset=test_dataset, batch_size=FLAGS.batch_size,pin_memory=True, shuffle=False)

print('dataset load finished!')

class My_loss(nn.Module):
    def __init__(self):
        super(My_loss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
    def forward(self, pred, target):
        loss = 0.0
        for p in pred:
            single_loss = self.criterion(p, target)
            loss = loss + single_loss
        return loss

    # def backward(self, retain_graph=True):
    #     self.loss.backward(retain_graph=retain_graph)
    #     return self.loss

def acc(pred, target):
    y_pred = np.sum(pred, axis=0)
    correct = torch.eq(torch.argmax(y_pred, dim=1), target).cpu().numpy()
    accuracy = np.mean(correct)
    return accuracy

def make_variable(tuple):
    return (t.float().cuda() for t in tuple)
    # return torch.LongTensor(tuple)

# def train():
#     total_loss = 0.0
#     for i, input in enumerate(train_loader):
#
#         context, content_mask, context_len,\
#         char_context, char_content_mask, char_context_len,\
#         response, response_mask, response_len,\
#         char_response, char_response_mask, char_response_len, y_label = make_variable(input)
#
#         pred = mrfn(context, content_mask, context_len,
#                     char_context, char_content_mask, char_context_len,
#                     response, response_mask, response_len,
#                     char_response, char_response_mask, char_response_len, y_label)
#
#         target = y_label.long()
#         loss = criterion(pred, target)
#         total_loss += loss.data
#
#         mrfn.zero_grad()
#         # loss.backward(retain_graph=True)
#         loss.backward()
#         optimizer.step()
#         return pred, target


def test(data_loader, name):
    correct = 0.0
    for i, input in enumerate(data_loader):
        context, content_mask, \
        char_context, \
        response, \
        char_response, y_label = make_variable(input)

        mrfn.eval()
        pred = mrfn(context, content_mask,
                    char_context,
                    response,
                    char_response, y_label)

        target = y_label.long()
        accuracy = acc(pred, target)
        # print("Step: %d\t| acc: %.3f\t" % (i, float(accuracy)))
        correct += accuracy
    print('\n' + name +' set: Accuracy: {}\n'.format(100. * correct / len(data_loader)))
    if name == 'Valid':
        result_valid.write(name +' set: Accuracy: {}\n'.format(100. * correct / len(data_loader)))
    else:
        result_test.write(name + ' set: Accuracy: {}\n'.format(100. * correct / len(data_loader)))



if __name__ == '__main__':
    torch.cuda.manual_seed(FLAGS.seed)
    pretrained_word_embeddings = None
    pretrained_char_embeddings = None
    if FLAGS.init_dict:
        pretrained_word_embeddings = pickle.load(open(FLAGS.init_embeddings_path, 'rb'))
        print(len(pretrained_word_embeddings))
    if FLAGS.init_char_dict:
        pretrained_char_embeddings = pickle.load(open(FLAGS.init_char_embeddings_path, 'rb'))
        print(len(pretrained_char_embeddings))

    mrfn = model(pretrained_word_embeddings, pretrained_char_embeddings)
    # mrfn.load_state_dict(torch.load('checkpoint_single/model_20000_single_gpu.ckpt'))

    if len(device_ids) > 1:
        # os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(','.join(device_ids))
        mrfn = nn.DataParallel(mrfn, device_ids=device_ids).cuda()
        # cudnn.benchmark = True
    else:
        mrfn.cuda()


    criterion = My_loss()
    optimizer = torch.optim.Adam(mrfn.parameters(), lr=FLAGS.lr)

    print('start train ...')
    if not os.path.exists('result'):
        os.makedirs('result')
    result_train = open('result/result_train.txt', 'w', encoding='utf-8')
    result_test = open('result/result_test.txt', 'w', encoding='utf-8')
    result_valid = open('result/result_valid.txt', 'w', encoding='utf-8')

    i = 0
    for epoch in range(1, FLAGS.num_epochs + 1):
        # total_loss = 0.0
        for input in train_loader:
            i += 1
            context, content_mask, \
            char_context, \
            response, \
            char_response, y_label = make_variable(input)

            torch.autograd.set_detect_anomaly(True)
            mrfn.train()
            pred = mrfn(context, content_mask,
                        char_context,
                        response,
                        char_response, y_label)

            target = y_label.long()
            loss = criterion(pred, target)
            # total_loss += loss.data

            mrfn.zero_grad()
            # loss.backward(retain_graph=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mrfn.parameters(), 10)
            optimizer.step()
            # print(mrfn.cross_interaction_matching_batch.linear.weight.grad)

            with torch.no_grad():
                if i % FLAGS.write_every == 0:
                    time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    accuracy = acc(pred, target)
                    result_train.write("Step: %d\t| loss: %.3f\t| acc: %.3f\t" % (i, loss.data, float(accuracy)) + '\n')
                if i % FLAGS.print_every == 0:
                    print("Step: %d\t| loss: %.3f\t| acc: %.3f\t| %s" % (i, loss.data, float(accuracy), time_str))



            with torch.no_grad():
                if i % FLAGS.checkpoint_every == 0:
                    if len(device_ids) > 1:
                        torch.save(mrfn.module.state_dict(), os.path.join(FLAGS.checkpoint_multi, 'model{}_multi_gpu.ckpt'.format(i)))
                        print("Write model{}_multi_gpu.ckpt done!".format(i))
                    else:
                        torch.save(mrfn.state_dict(), os.path.join(FLAGS.checkpoint_single, 'model_{}_single_gpu.ckpt'.format(i)))
                        print("Write model_{}_single_gpu.ckpt done!".format(i))

                if i % FLAGS.valid_every == 0:
                    test(valid_loader, 'Valid')

                if i % FLAGS.test_every == 0:
                    test(test_loader, 'Test')
            torch.cuda.empty_cache()

        # epoch end and save model
        if len(device_ids) > 1:
            torch.save(mrfn.module.state_dict(),
                       os.path.join(FLAGS.checkpoint_multi, 'model{}_multi_gpu.ckpt'.format(i)))
            print("Write model{}_multi_gpu.ckpt done!".format(i))
        else:
            torch.save(mrfn.state_dict(), os.path.join(FLAGS.checkpoint_single, 'model_{}_single_gpu.ckpt'.format(i)))
            print("Write model_{}_single_gpu.ckpt done!".format(i))






