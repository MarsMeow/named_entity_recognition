import os
from torch.utils.data import Dataset
from cblue.utils import load_json, load_dict, write_dict

import json
import numpy as np
import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader

from cblue.utils import seed_everything, ProgressBar, TokenRematch
from cblue.metrics import sts_metric, qic_metric, qqr_metric, qtr_metric, \
    ctc_metric, ee_metric, er_metric, re_metric, cdn_cls_metric, cdn_num_metric
from cblue.metrics import sts_commit_prediction, qic_commit_prediction, qtr_commit_prediction, \
    qqr_commit_prediction, ctc_commit_prediction, ee_commit_prediction, cdn_commit_prediction
from cblue.models import convert_examples_to_features, save_zen_model




class EEDataProcessor(object):
    def __init__(self, root, is_lower=True, no_entity_label='O'):
        self.task_data_dir = os.path.join(root, 'CMeEE')
        self.train_path = os.path.join(self.task_data_dir, 'CMeEE_train.json')
        self.dev_path = os.path.join(self.task_data_dir, 'CMeEE_dev.json')
        self.test_path = os.path.join(self.task_data_dir, 'CMeEE_test.json')

        self.label_map_cache_path = os.path.join(self.task_data_dir, 'CMeEE_label_map.dict')
        self.label2id = None
        self.id2label = None
        self.no_entity_label = no_entity_label
        self._get_labels()
        self.num_labels = len(self.label2id.keys())

        self.is_lower = is_lower

    def get_train_sample(self):
        return self._pre_process(self.train_path, is_predict=False)

    def get_dev_sample(self):
        return self._pre_process(self.dev_path, is_predict=False)

    def get_test_sample(self):
        return self._pre_process(self.test_path, is_predict=True)

    def _get_labels(self):
        if os.path.exists(self.label_map_cache_path):
            label_map = load_dict(self.label_map_cache_path)
        else:
            label_set = set()
            samples = load_json(self.train_path)
            for sample in samples:
                for entity in sample["entities"]:
                    label_set.add(entity['type'])
            label_set = sorted(label_set)
            labels = [self.no_entity_label]
            for label in label_set:
                labels.extend(["B-{}".format(label), "I-{}".format(label)])
            label_map = {idx: label for idx, label in enumerate(labels)}
            write_dict(self.label_map_cache_path, label_map)
        self.id2label = label_map
        self.label2id = {val: key for key, val in self.id2label.items()}

    def _pre_process(self, path, is_predict):
        def label_data(data, start, end, _type):
            """label_data"""
            for i in range(start, end + 1):
                suffix = "B-" if i == start else "I-"
                data[i] = "{}{}".format(suffix, _type)
            return data

        outputs = {'text': [], 'label': [], 'orig_text': []}
        samples = load_json(path)
        for data in samples:
            if self.is_lower:
                text_a = ["，" if t == " " or t == "\n" or t == "\t" else t
                          for t in list(data["text"].lower())]
            else:
                text_a = ["，" if t == " " or t == "\n" or t == "\t" else t
                          for t in list(data["text"])]
            # text_a = "\002".join(text_a)
            outputs['text'].append(text_a)
            outputs['orig_text'].append(data['text'])
            if not is_predict:
                labels = [self.no_entity_label] * len(text_a)
                for entity in data['entities']:
                    start_idx, end_idx, type = entity['start_idx'], entity['end_idx'], entity['type']
                    labels = label_data(labels, start_idx, end_idx, type)
                outputs['label'].append('\002'.join(labels))
        return outputs

    def extract_result(self, results, test_input):
        predicts = []
        for j in range(len(results)):
            text = "".join(test_input[j])
            ret = []
            entity_name = ""
            flag = []
            visit = False
            start_idx, end_idx = 0, 0
            for i, (char, tag) in enumerate(zip(text, results[j])):
                tag = self.id2label[tag]
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

class EEDataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            mode='train',
            max_length=128,
            model_type='bert',
            ngram_dict=None
    ):
        super(EEDataset, self).__init__()

        self.orig_text = samples['orig_text']
        self.texts = samples['text']
        if mode != "test":
            self.labels = samples['label']
        else:
            self.labels = None

        self.data_processor = data_processor
        self.max_length = max_length
        self.mode = mode
        self.ngram_dict = ngram_dict
        self.model_type = model_type

    def __len__(self):
        return len(self.texts)




def build_corpus(split, make_vocab=True, data_dir="./data/"):
    """读取数据"""
    assert split in ['train', 'dev', 'test']

    word_lists = []
    tag_lists = []

    data_processor = EEDataProcessor(root=data_dir, is_lower=False)
    dataset_class = EEDataset
    if split == 'train':
        train_samples = data_processor.get_train_sample()
        train_dataset = dataset_class(train_samples, data_processor, mode='train',
                                      model_type='hmm', ngram_dict=None, max_length=128)
        for label_idx in range(len(train_dataset.labels)):  # 15000
            train_dataset.labels[label_idx] = train_dataset.labels[label_idx].split('\002')
        word_lists = train_dataset.texts
        tag_lists = train_dataset.labels
    elif split == 'dev':
        eval_samples = data_processor.get_dev_sample()
        eval_dataset = dataset_class(eval_samples, data_processor, mode='eval',
                                     model_type='hmm', ngram_dict=None, max_length=128)
        for label_idx in range(len(eval_dataset.labels)):
            eval_dataset.labels[label_idx] = eval_dataset.labels[label_idx].split('\002')
        word_lists = eval_dataset.texts
        tag_lists = eval_dataset.labels
    elif split == 'test':
        test_samples = data_processor.get_test_sample()
        test_dataset = dataset_class(test_samples, data_processor, mode='test', ngram_dict=None,
                                     max_length=128, model_type='hmm')
        word_lists = test_dataset.texts
        # tag_lists = test_dataset.labels

    # 如果make_vocab为True，还需要返回word2id和tag2id
    if make_vocab:
        word2id = build_map(word_lists)
        # tag2id = build_map(tag_lists)
        return word_lists, tag_lists, word2id
    elif split == 'test':
        return word_lists
    else:
        return word_lists, tag_lists


def build_map(lists):
    maps = {}
    for list_ in lists:
        for e in list_:
            if e not in maps:
                maps[e] = len(maps)

    return maps
