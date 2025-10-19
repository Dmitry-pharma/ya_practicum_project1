import torch.nn as nn
import torch

class NextPhrasePredictionRNN(nn.Module):
    def __init__(self, rnn_type="LSTM", vocab_size=30522, emb_dim=300, num_layers=1, hidden_dim=256, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        
        rnn_cls = {"RNN": nn.RNN, "GRU": nn.GRU, "LSTM": nn.LSTM}[rnn_type]
        self.rnn = rnn_cls(emb_dim, hidden_dim, num_layers, batch_first=True)

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.7) #0.5
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
        
        # rnn_out_normed = self.norm(rnn_out)
        mask = attention_mask.unsqueeze(-1).float()
        # masked_out = rnn_out_normed * mask
        masked_out = rnn_out * mask
        normalized_out=self.norm(masked_out)
        # summed = masked_out.sum(dim=1)
        # lengths = attention_mask.sum(dim=1).unsqueeze(1).clamp(min=1e-9)
        # mean_pooled = summed / lengths
        # seq_representation = mean_pooled.unsqueeze(1).repeat(1, input_ids.size(1), 1)
        # out = self.dropout(seq_representation)
        # out = self.dropout(masked_out)     
        dropped_out= self.dropout(normalized_out)
        # logits = self.fc(out)
        logits = self.fc(dropped_out)
        
        # Берем ТОЛЬКО последний токен для предсказания
        # last_token_indices = attention_mask.sum(dim=1) - 1  # индексы последних токенов
        # last_hidden_states = rnn_out[torch.arange(rnn_out.size(0)), last_token_indices]  # [batch_size, hidden_dim]

        # last_hidden_normed = self.norm(last_hidden_states)
        # last_hidden_dropped = self.dropout(last_hidden_normed)

        # # Предсказание следующего токена
        # next_token_logits = self.fc(last_hidden_dropped)  # [batch_size, vocab_size]
        
        return logits #next_token_logits