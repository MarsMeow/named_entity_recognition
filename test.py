from select import epoll
import torch
from torch.utils.data import Dataset, DataLoader
from .data import build_corpus
from utils import extend_maps, prepocess_data_for_lstmcrf
from evaluate_m import pred_res, hmm_train_eval, crf_train_eval, \
    bilstm_train_and_eval, ensemble_evaluate

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

print("正在训练评估BiLSTM模型...")
# LSTM模型训练的时候需要在word2id和tag2id加入PAD和UNK
#bilstm_word2id, bilstm_tag2id = extend_maps(word2id, tag2id, for_crf=False)
lstm_pred = bilstm_train_and_eval(
    (train_word_lists, train_tag_lists),
    (dev_word_lists, dev_tag_lists),
    test_word_lists,
    bilstm_word2id, bilstm_tag2id,
    crf=False
)
pred_res(orig_text= orig_text,test_res=lstm_pred, test_word_lists =test_word_lists, 
            result_output_dir="/home/zcming/projs/named_entity_recognition", tag2id=tag2id, id2label=id2label)













def sort_by_lengths(word_lists, tag_lists):
    
    pairs = list(zip(word_lists, tag_lists))
    indices = sorted(range(len(pairs)),
                     key=lambda k: len(pairs[k][0]),
                     reverse=True)
    pairs = [pairs[i] for i in indices]
    # pairs.sort(key=lambda pair: len(pair[0]), reverse=True)

    word_lists, tag_lists = list(zip(*pairs))

    return word_lists, tag_lists, indices