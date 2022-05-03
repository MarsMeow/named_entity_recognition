import torch
import torch.nn as nn
import gc

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, out_size):
        """初始化参数：
            vocab_size:字典的大小
            emb_size:词向量的维数
            hidden_size:隐向量的维数
            out_size:标注的种类
        """
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.bilstm = nn.LSTM(emb_size, hidden_size,
                              batch_first=True,
                              bidirectional=True)

        self.lin = nn.Linear(2*hidden_size, out_size)

    def forward(self, sents_tensor, lengths):
        
        emb = self.embedding(sents_tensor)  # [B, L, emb_size]
        
        packed = pack_padded_sequence(emb, lengths, batch_first=True)
        packed, _ = self.bilstm(packed)
        packed, _ = pad_packed_sequence(packed, batch_first=True)
        self.bilstm.flatten_parameters()
        scores = self.lin(packed)  # [B, L, out_size]

        return scores

    def test(self, sents_tensor, lengths, _):
        """第三个参数不会用到，加它是为了与BiLSTM_CRF保持同样的接口"""
        with torch.no_grad():
            logits = self.forward(sents_tensor, lengths)  # [B, L, out_size]
            _, batch_tagids = torch.max(logits, dim=2)

        return batch_tagids
