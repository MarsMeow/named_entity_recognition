import time
import os
import json
import pdb
from collections import Counter

import torch

from models.hmm import HMM
from models.crf import CRFModel
from models.bilstm_crf import BILSTM_Model
from utils import save_model, flatten_lists, load_model
from evaluating import Metrics




def extract_result(results, test_input, id2label):
    predicts = []
    for j in range(len(results)):
        text = "".join(test_input[j])
        ret = []
        entity_name = ""
        flag = []
        visit = False
        start_idx, end_idx = 0, 0
        for i, (char, tag) in enumerate(zip(text, results[j])):
            tag = id2label[tag]
            if tag[0] == "B":
                if entity_name != "":
                    x = dict((a, flag.count(a)) for a in flag)
                    y = [k for k, v in x.items() if max(x.values()) == v]
                    ret.append({"start_idx": start_idx, "end_idx": end_idx, "type": y[0], "entity": entity_name})
                    flag.clear()
                    entity_name = ""
                visit = True
                start_idx = i
                entity_name += char
                flag.append(tag[2:])
                end_idx = i
            elif tag[0] == "I" and visit:
                entity_name += char
                flag.append(tag[2:])
                end_idx = i
            else:
                if entity_name != "":
                    x = dict((a, flag.count(a)) for a in flag)
                    y = [k for k, v in x.items() if max(x.values()) == v]
                    ret.append({"start_idx": start_idx, "end_idx": end_idx, "type": y[0], "entity": entity_name})
                    flag.clear()
                start_idx = i + 1
                visit = False
                flag.clear()
                entity_name = ""

        if entity_name != "":
            x = dict((a, flag.count(a)) for a in flag)
            y = [k for k, v in x.items() if max(x.values()) == v]
            ret.append({"start_idx": start_idx, "end_idx": end_idx, "type": y[0], "entity": entity_name})
        predicts.append(ret)
    return predicts

def ee_commit_prediction(orig_text, preds, output_dir):
    pred_result = []
    for item in zip(orig_text, preds):
        tmp_dict = {'text': item[0], 'entities': item[1]}
        pred_result.append(tmp_dict)
    with open(os.path.join(output_dir, 'CMeEE_test.json'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(pred_result, indent=2, ensure_ascii=False))

def word_trans_tag(words, tag2id):
    res = []
    for w in words:
        res.append(tag2id[w])
    return res    



def pred_res(orig_text, test_res, test_word_lists, result_output_dir, tag2id, id2label):
    predictions = []
    test_inputs = test_word_lists
    for tes in test_res:
        predictions.append(word_trans_tag(tes, tag2id))
    # predictions = [pred[1:-1] for pred in predictions]
    predicts = extract_result(predictions, test_inputs, id2label)
    ee_commit_prediction(orig_text=orig_text, preds=predicts, output_dir=result_output_dir)

def hmm_train_eval(train_data, dev_data, test_data, word2id, tag2id, remove_O=False):
    """训练并评估hmm模型"""
    # 训练HMM模型
    train_word_lists, train_tag_lists = train_data
    dev_word_lists, dev_tag_lists = dev_data
    test_word_lists = test_data
    hmm_model = HMM(len(tag2id), len(word2id))
    hmm_model.train(train_word_lists,
                    train_tag_lists,
                    word2id,
                    tag2id)
    save_model(hmm_model, "/home/zcming/projs/named_entity_recognition/ckpts/hmm.pkl")

    # 评估hmm模型
    dev_tag_lists_pred = hmm_model.test(dev_word_lists,
                                    word2id,
                                    tag2id)
    print("finish dev")
    pred_tag_lists = hmm_model.test(test_word_lists,
                                    word2id,
                                    tag2id)
    print("finish pre")
    metrics = Metrics(dev_tag_lists, dev_tag_lists_pred, remove_O=remove_O)
    metrics.report_scores()
    metrics.report_confusion_matrix()

    return dev_tag_lists_pred, pred_tag_lists

def crf_train_eval(train_data, dev_data, test_data, remove_O=False):

    # 训练CRF模型
    train_word_lists, train_tag_lists = train_data
    dev_word_lists, dev_tag_lists = dev_data
    test_word_lists = test_data
    crf_model = CRFModel()
    crf_model.train(train_word_lists, train_tag_lists, )
    save_model(crf_model, "/home/zcming/projs/named_entity_recognition/ckpts/crf.pkl")

    dev_tag_lists_pred = crf_model.test(dev_word_lists)
    pred_tag_lists = crf_model.test(test_word_lists)
    metrics = Metrics(dev_tag_lists, dev_tag_lists_pred, remove_O=remove_O)
    metrics.report_scores()
    metrics.report_confusion_matrix()

    return dev_tag_lists_pred, pred_tag_lists


def bilstm_train_and_eval(train_data, dev_data, test_data,
                          word2id, tag2id, crf=True, remove_O=False):
    train_word_lists, train_tag_lists = train_data
    dev_word_lists, dev_tag_lists = dev_data

    start = time.time()
    vocab_size = len(word2id)
    out_size = len(tag2id)
    bilstm_model = BILSTM_Model(vocab_size, out_size, crf=crf)
    bilstm_model.train(train_word_lists, train_tag_lists, dev_word_lists, dev_tag_lists, word2id, tag2id)

    model_name = "bilstm_crf" if crf else "bilstm"
    save_model(bilstm_model, "/home/zcming/projs/named_entity_recognition/ckpts/"+model_name+".pkl")

    print("训练完毕,共用时{}秒.".format(int(time.time()-start)))


def bilstm_eval(test_data, word2id, tag2id, model_name = 'bilstm' ):
    test_word_lists = test_data
    print("评估{}模型中...".format(model_name))
    bilstm_model = load_model("/home/zcming/projs/named_entity_recognition/ckpts/"+model_name+".pkl")
    with torch.no_grad():
        pred_tag_lists = bilstm_model.test(test_word_lists, word2id, tag2id)

    return pred_tag_lists


def ensemble_evaluate(results, targets, remove_O=False):
    """ensemble多个模型"""
    for i in range(len(results)):
        results[i] = flatten_lists(results[i])

    pred_tags = []
    for result in zip(*results):
        ensemble_tag = Counter(result).most_common(1)[0][0]
        pred_tags.append(ensemble_tag)

    targets = flatten_lists(targets)
    assert len(pred_tags) == len(targets)

    print("Ensemble 四个模型的结果如下：")
    metrics = Metrics(targets, pred_tags, remove_O=remove_O)
    metrics.report_scores()
    metrics.report_confusion_matrix()
