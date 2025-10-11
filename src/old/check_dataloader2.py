import torch.nn as nn
from sklearn.metrics import accuracy_score
import re
import numpy as np
import pandas as pd
from datasets import load_dataset
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#---------------------------------------------------------------------------
# 1. Загрузка датасета
dataset = load_dataset("amazon_polarity", split="train[:1000]") 
texts, labels = dataset['content'], dataset['label']

# 2. Очистка текстов
def clean_text(text):
    text = text.lower()  # к нижнему регистру
    text = re.sub(r"[^a-z0-9 ]+", " ", text)  # оставить только буквы и цифры
    text = re.sub(r"\s+", " ", text).strip()  # убрать дублирующиеся пробелы
    return text
texts = [clean_text(text) for text in texts]

#---------------------------------------------------------------------------


X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, random_state=42)


from transformers import BertTokenizerFast
import torch
from torch.utils.data import Dataset, DataLoader

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

class AmazonRNNDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.encodings = tokenizer(texts, padding='max_length', truncation=True, max_length=max_len, return_tensors='pt')
        self.labels = torch.tensor(labels)#, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc=self.encodings[idx]
        return {
            'input_ids': enc['input_ids'],
            'attention_mask': enc['attention_mask'],
            'label': self.labels[idx]
        }


train_ds = AmazonRNNDataset(X_train, y_train, tokenizer)
val_ds = AmazonRNNDataset(X_val, y_val, tokenizer)


train_loader = DataLoader(train_ds, batch_size=5, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=5)


# class MeanPoolingRNN(nn.Module):
#     def __init__(self, vocab_size, emb_dim=300, hidden_dim=256, output_dim=2, pad_idx=0):
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
#         self.rnn = nn.RNN(emb_dim, hidden_dim, batch_first=True)# RNN-слой
#         self.norm = nn.LayerNorm(hidden_dim) # слой нормализации
#         self.dropout = nn.Dropout(0.5)# dropout-слой
#         self.fc = nn.Linear(hidden_dim, output_dim) # линейный слой для классификации
#         self.init_weights()


#     def init_weights(self):
#         # напишите xavier инициализацию весов
#         # if isinstance(self, nn.Linear) or isinstance(self, nn.RNN):
#         #     nn.init.xavier_uniform_(self.weight)
#         #     if self.bias is not None:
#         #         nn.init.zeros_(self.bias)
#         # nn.init.xavier_uniform_(self.fc.weight)
#         # for name, param in self.rnn.named_parameters():
#         #     if 'weight' in name:
#         #         nn.init.xavier_uniform_(param)
#         for name, param in self.named_parameters():
#             if 'weight' in name and param.dim() > 1:
#                 nn.init.xavier_uniform_(param)
#             elif 'bias' in name:
#                 nn.init.constant_(param, 0.0)


#     def forward(self, input_ids, attention_mask):
#         x = self.embedding(input_ids)
#         rnn_out, _ = self.rnn(x)# посчитайте результат rnn-слоя

#         rnn_out_normed = self.norm(rnn_out)# посчитайте нормирование для rnn_out


#         # mean pooling по attention_mask
#         mask = attention_mask.unsqueeze(2).expand_as(rnn_out)# подготовьте маску для rnn_out_normed
#         # mask = mask.unsqueeze(-1).float()
#         masked_out =  rnn_out_normed * mask# умножьте rnn_out_normed на маску
#         summed = masked_out.sum(dim=1)# посчитайте сумму masked_out по dim=1
#         lengths = attention_mask.sum(dim=1).unsqueeze(1)# посчитайте длины последовательностей
#         mean_pooled = summed / lengths# посчитайте средние, разделив суммы на длины последовательностей
#         out = self.dropout(mean_pooled)# примените dropout
#         logits = self.fc(out)# примените линейный слой 
#         return logits


# создание модели
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pad_token_id = tokenizer.pad_token_id  # добавляем определение pad_token_id
# model = MeanPoolingRNN(vocab_size=tokenizer.vocab_size, pad_idx=pad_token_id).to(device)


# создание оптимизатора и функции потерь
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)#, weight_decay=3e-3) # создайте оптимайзер с регуляризацией
# criterion = nn.CrossEntropyLoss()


# код обучения одной эпохи
# def train_epoch(model, loader):
    # model.train()
    # total_loss = 0
for batch_idx,batch in enumerate(train_loader):
    ids = batch['input_ids'].to(device)
    mask = batch['attention_mask'].to(device)
    labels = batch['label'].to(device)
    print(ids,mask,labels)
    if batch_idx == 1:  # остановимся после второго батча
        break

        # optimizer.zero_grad()# обнулите градиенты
        # logits = model(ids, mask)# посчитайте выход модели
        
        # print(f"Logits shape: {logits.shape}")  # должно быть [128, 2]
        # print(f"Labels shape: {labels.shape}")  # должно быть [128]
        # print(f"Labels dtype: {labels.dtype}")  # должно быть torch.int64
        
        # loss = criterion(logits, labels)# посчитайте функцию потерь
        # loss.backward()  # посчитайте градиенты
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # примените gradient clipping
        # optimizer.step() # обновите градиенты

        # total_loss += loss.item()

        
        
    # return total_loss / len(loader)


# код подсчёта accuracy на валидации
# def evaluate(model, loader):
#     model.eval()
#     preds, trues = [], []
#     with torch.no_grad():
#         for batch in loader:
#             ids = batch['input_ids'].to(device)
#             mask = batch['attention_mask'].to(device)
#             labels = batch['label']
#             logits = model(ids, mask)
#             preds += torch.argmax(logits, dim=1).cpu().tolist()
#             trues += labels.tolist()
#     return accuracy_score(trues, preds)


# # обучение
# for epoch in range(10):
#     loss = train_epoch(model, train_loader)
#     acc = evaluate(model, val_loader)
#     print(f"Epoch {epoch+1}: Loss = {loss:.4f}, Accuracy = {acc:.4f}")