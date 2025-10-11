import torch.nn as nn
import torch

class NextPhrasePredictionRNN(nn.Module):
    def __init__(self, rnn_type="LSTM", vocab_size=30522, emb_dim=300, hidden_dim=256, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        
        rnn_cls = {"RNN": nn.RNN, "GRU": nn.GRU, "LSTM": nn.LSTM}[rnn_type]
        self.rnn = rnn_cls(emb_dim, hidden_dim, batch_first=True)

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, input_ids, attention_mask):
        x = self.embedding(input_ids)
        rnn_out, _ = self.rnn(x)
        
        rnn_out_normed = self.norm(rnn_out)

        mask = attention_mask.unsqueeze(-1).float()
        masked_out = rnn_out_normed * mask
        
        # summed = masked_out.sum(dim=1)
        # lengths = attention_mask.sum(dim=1).unsqueeze(1).clamp(min=1e-9)
        # mean_pooled = summed / lengths
        # seq_representation = mean_pooled.unsqueeze(1).repeat(1, input_ids.size(1), 1)
        # out = self.dropout(seq_representation)
        out = self.dropout(masked_out)
        
        logits = self.fc(out)
        
        return logits