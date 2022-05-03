from select import epoll
import torch
from torch.utils.data import Dataset, DataLoader
from data import build_corpus
from utils import extend_maps, prepocess_data_for_lstmcrf
from evaluate_m import pred_res, hmm_train_eval, crf_train_eval, \
    bilstm_train_and_eval, ensemble_evaluate, bilstm_eval

def load_dict(dict_path):
    """load_dict"""
    vocab = {}
    for line in open(dict_path, 'r', encoding='utf-8'):
        key, value = line.strip('\n').split('\t')
        vocab[int(key)] = value
    return vocab


"""训练模型，评估结果"""

# 读取数据
print("读取数据...")
train_word_lists, train_tag_lists, word2id = build_corpus("train", data_dir="/home/zcming/projs/CBLUE/CBLUEDatasets")

id2label = load_dict("/home/zcming/projs/CBLUE/CBLUEDatasets/CMeEE/CMeEE_label_map.dict")
tag2id = {val: key for key, val in id2label.items()}

dev_word_lists, dev_tag_lists = build_corpus("dev", make_vocab=False, data_dir="/home/zcming/projs/CBLUE/CBLUEDatasets")
test_word_lists = build_corpus("test", make_vocab=False, data_dir="/home/zcming/projs/CBLUE/CBLUEDatasets")
orig_text = test_word_lists
idx = 0
for word in test_word_lists:
    orig_text[idx] = "".join(word)
    idx += 1
epoch = 10
train_batch_size = 128

bilstm_word2id, bilstm_tag2id = extend_maps(word2id, tag2id, for_crf=False)

'''for e in range(epoch):
    for data_index, tag_index in train_dataloader:
        print("")'''




'''
# 训练评估ｈｍｍ模型
print("正在训练评估HMM模型...")
dev_hmm_pred, hmm_pred = hmm_train_eval(
    (train_word_lists, train_tag_lists),
    (dev_word_lists, dev_tag_lists),
    test_word_lists,
    word2id,
    tag2id
)

print("正在预测HMM模型...")
hmm_pred_res(orig_text= orig_text,test_res=hmm_pred, test_word_lists =test_word_lists, 
            result_output_dir="/home/zcming/projs/named_entity_recognition", tag2id=tag2id, id2label=id2label)
'''

# 训练评估CRF模型
'''    print("正在训练评估CRF模型...")
dev_hmm_pred, crf_pred = crf_train_eval(
    (train_word_lists, train_tag_lists),
    (dev_word_lists, dev_tag_lists),
    test_word_lists,
    word2id
)
pred_res(orig_text= orig_text,test_res=crf_pred, test_word_lists =test_word_lists, 
            result_output_dir="/home/zcming/projs/named_entity_recognition", tag2id=tag2id, id2label=id2label)'''

# 训练评估BI-LSTM模型
print("正在训练评估BiLSTM模型...")
# LSTM模型训练的时候需要在word2id和tag2id加入PAD和UNK
#bilstm_word2id, bilstm_tag2id = extend_maps(word2id, tag2id, for_crf=False)
'''bilstm_train_and_eval(
    (train_word_lists, train_tag_lists),
    (dev_word_lists, dev_tag_lists),
    test_word_lists,
    bilstm_word2id, bilstm_tag2id,
    crf=False
)'''
lstm_pred = bilstm_eval(test_word_lists, bilstm_word2id, bilstm_tag2id, model_name = 'bilstm')
pred_res(orig_text= orig_text,test_res=lstm_pred, test_word_lists =test_word_lists, 
            result_output_dir="/home/zcming/projs/named_entity_recognition", tag2id=tag2id, id2label=id2label)
'''
print("正在训练评估Bi-LSTM+CRF模型...")
# 如果是加了CRF的lstm还要加入<start>和<end> (解码的时候需要用到)
crf_word2id, crf_tag2id = extend_maps(word2id, tag2id, for_crf=True)
# 还需要额外的一些数据处理
train_word_lists, train_tag_lists = prepocess_data_for_lstmcrf(
    train_word_lists, train_tag_lists
)
dev_word_lists, dev_tag_lists = prepocess_data_for_lstmcrf(
    dev_word_lists, dev_tag_lists
)
test_word_lists, test_tag_lists = prepocess_data_for_lstmcrf(
    test_word_lists, test_tag_lists, test=True
)
lstmcrf_pred = bilstm_train_and_eval(
    (train_word_lists, train_tag_lists),
    (dev_word_lists, dev_tag_lists),
    (test_word_lists, test_tag_lists),
    crf_word2id, crf_tag2id
)

ensemble_evaluate(
    [dev_pred, crf_pred],# lstm_pred]# , lstmcrf_pred],
    test_tag_lists
)
'''

print("finished")
