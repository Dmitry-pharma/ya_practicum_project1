# https://code.s3.yandex.net/deep-learning/tweets.txt
import torch.nn as nn
from sklearn.metrics import accuracy_score
import re
import numpy as np
import pandas as pd
from datasets import load_dataset
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AutoTokenizer
from pathlib import Path
import html
from torch.utils.data import Dataset, DataLoader
#---------------------------------------------------------------------------
file_path = Path(__file__).parent.parent / 'data' / '1_raw_dataset_tweets.txt'
print(file_path)

# v_path="../data/raw_dataset_tweets.txt"
# v_path="C:/Users/OMEN/Documents/LLM_Test/YaPracticum/project1_text-autocomplete/data/raw_dataset_tweets.txt"
# texts_df = pd.read_csv(file_path)
# print(f"Загружено строк: {len(dtexts_df)}")

# Просто читаем все строки
with open(file_path, 'r', encoding='utf-8') as f:
    texts = f.readlines()

# Убираем символы переноса строк
texts = [text.strip() for text in texts]
texts = texts[:10]
print(f"Загружено {len(texts)} строк")
# for i, text in enumerate(texts[:5]):  # первые 5 строк
#     print(f"{i+1}: {text}")
    
def clean_text(text):
    text = text.lower()  # к нижнему регистру    
    text = html.unescape(text)  # преобразует &quot; -> ", &amp; -> &, и т.д.# 2. Декодировать HTML символы  
    text = re.sub(r'<[^>]+>', '', text)  # 3. Удалить оставшиеся HTML теги
    text = re.sub(r'@\w+', 'user', text)# заменить упоминания (@username)    
    text = re.sub(r'#\w+', '', text) # Удалить хештеги (#tag)    
    text = re.sub(r'[^\w\s,\.!?;:]', '', text) # Удалить эмодзи и специальные символы
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)# Удалить ссылки    
   
    text = re.sub(r"[^a-z0-9 ]+", " ", text)  # оставить только буквы и цифры
    text = re.sub(r"\s+", " ", text).strip()  # убрать дублирующиеся пробелы
    
    return text

print("Cleaning...")
cleaned_texts =[]
for text in tqdm(texts):
    cleaned_texts.append(clean_text(text))
    
text_pairs = [(i, texts[i], cleaned_texts[i]) for i in range(len(texts))]

#сохраняем очищенный текст в csv
texts_df = pd.DataFrame(text_pairs, columns=['rowno','text_raw','text_cleaned'])
# output_path = Path(__file__).parent.parent / 'data' / '2_cleaned_dataset_tweets.csv'
# texts_df.to_csv(output_path, index=False, encoding='utf-8')



# from transformers import BertTokenizerFast
# import torch
from torch.utils.data import Dataset, DataLoader

# tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
class TweetRNNDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=20):
        # self.encodings = tokenizer(texts, padding='max_length', truncation=True, max_length=max_len, return_tensors='pt')
        # self.labels = torch.tensor(labels)#, dtype=torch.long)
        # self.samples = []
        # self.tokenizer = tokenizer
        # self.max_len = max_len
        
        print("Preparing X, Y pairs...")
        for text in tqdm(texts):
            # Токенизируем текст
            tokens = tokenizer.tokenize(text)
            
            if len(tokens) < 2:  # пропускаем слишком короткие тексты
                continue
                
            # Для длинных текстов разбиваем на скользящие окна размером max_len
            for i in range(1, len(tokens)):
                start_idx = max(0, i - max_len)
                x_tok = tokens[start_idx:i]  # токены с start_idx до i-1        
                y_tok = tokens[start_idx+1:i+1]  # Цель: те же токены, но сдвинутые на 1 вперед
                
                if len(x_tok) < 1 or len(y_tok) < 1:
                    continue
                
                # Преобразуем токены в ID
                x_ids = tokenizer.convert_tokens_to_ids(x_tok)
                y_ids = tokenizer.convert_tokens_to_ids(y_tok)
                
                # Паддинг слева для X/Y до max_len
                x_padded = [tokenizer.pad_token_id] * (max_len - len(x_ids)) + x_ids
                y_padded = [tokenizer.pad_token_id] * (max_len - len(y_ids)) + y_ids
                
                # Создаем attention_mask (1 для реальных токенов, 0 для паддинга)
                attention_mask = [1 if token_id != tokenizer.pad_token_id else 0 for token_id in x_padded]
                
                 # Для токенов тоже делаем паддинг, чтобы они были одинаковой длины
                x_tok_padded = ['[PAD]'] * (max_len - len(x_tok)) + x_tok
                y_tok_padded = ['[PAD]'] * (max_len - len(y_tok)) + y_tok
                
                # print(i,x_tok,y_tok,x_padded,attention_mask,y_padded)
                self.samples.append({
                    'x':x_tok_padded,
                    'y':y_tok_padded,
                    'input_ids': x_padded,
                    'attention_mask': attention_mask,
                    'labels': y_padded
                })
        
        print(f"PAIRS ARE READY! Prepeared {len(self.samples)} pairs")
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'input_ids': sample['input_ids'],
            'attention_mask': sample['attention_mask'],
            'labels': sample['labels'],
            'x':sample['x'],
            'y':sample['y']
        }

# X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, random_state=42)
train, temp = train_test_split(texts_df['text_cleaned'].head(1000), test_size=0.2, random_state=42)
val, test = train_test_split(temp, test_size=1/2, random_state=42)  # val 0.1, test 0.2

train_ds = TweetRNNDataset(train, tokenizer)
val_ds = TweetRNNDataset(val, tokenizer)
test_ds = TweetRNNDataset(test, tokenizer)

train_loader = DataLoader(train_ds, batch_size=128)#, shuffle=True)
val_loader = DataLoader(val_ds)#, batch_size=128)
test_loader = DataLoader(test_ds)#, batch_size=128)

print(f"✅ Датасеты созданы: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

# # ДОБАВЛЕН КОД ДЛЯ ВЫВОДА 10 РЕЗУЛЬТАТОВ ИЗ DATALOADER
# print("\n" + "="*60)
# print("ВЫВОД 10 РЕЗУЛЬТАТОВ ИЗ DATALOADER")
# print("="*60)
n_epochs = 3

for epoch in range(n_epochs): 
    for one in (train_loader):
        print(one['input_ids'])
        
# for batch_idx, batch in enumerate(train_loader):
#     if batch_idx >= 10:
#         break
#     print(batch_idx, batch)
    
# # Функция для декодирования ID обратно в текст
# def decode_batch(batch, tokenizer):
#     """Декодирует батч обратно в читаемый текст"""
#     decoded_samples = []
#     for i in range(len(batch['input_ids'])):
#         # Декодируем input_ids (контекст)
#         input_ids = batch['input_ids'][i]
#         input_tokens = tokenizer.convert_ids_to_tokens(input_ids)
#         input_text = tokenizer.convert_tokens_to_string([t for t in input_tokens if t != tokenizer.pad_token])
        
#         # Декодируем labels (целевая последовательность)
#         label_ids = batch['labels'][i]
#         label_tokens = tokenizer.convert_ids_to_tokens(label_ids)
#         label_text = tokenizer.convert_tokens_to_string([t for t in label_tokens if t != tokenizer.pad_token])
        
#         # Attention mask
#         attention_mask = batch['attention_mask'][i]
        
#         decoded_samples.append({
#             'x':batch['x'][i],
#             'y':batch['y'][i],
#             'input_text': input_text,
#             'label_text': label_text,
#             'input_ids': input_ids.tolist(),
#             'label_ids': label_ids.tolist(),
#             'attention_mask': attention_mask.tolist()
#         })
    
#     return decoded_samples

# # Выводим 10 примеров из train_loader
# print("\n🎯 10 ПРИМЕРОВ ИЗ TRAIN DATALOADER:")
# print("-" * 60)

# batch_count = 0
# total_examples = 0

# for batch_idx, batch in enumerate(train_loader):
#     decoded_batch = decode_batch(batch, tokenizer)
    
#     for example_idx, example in enumerate(decoded_batch):
#         if total_examples >= 10:
#             break
            
#         print(f"\n📝 Пример {total_examples + 1}:")
#         print(f"   Контекст (X): '{example['input_text']}'")
#         print(f"   Цель (Y):     '{example['label_text']}'")
#         print(f"   Input IDs:    {example['input_ids']}")
#         print(f"   Label IDs:    {example['label_ids']}")
#         print(f"   Attn Mask:    {example['attention_mask']}")
#         print(f"   X:    {example['x']}")
#         print(f"   Y:    {example['y']}")
        
        
#         total_examples += 1
    
#     if total_examples >= 10:
#         break
    
#     batch_count += 1

# print(f"\n📊 Просмотрено батчей: {batch_count + 1}")
# print(f"📊 Размер батча: {batch['input_ids'].shape[1]}")

# # Дополнительно: вывод статистики по тензорам
# print("\n📈 СТАТИСТИКА ТЕНЗОРОВ:")
# print(f"   Input IDs shape: {batch['input_ids'].shape}")
# print(f"   Labels shape: {batch['labels'].shape}")
# print(f"   Attention Mask shape: {batch['attention_mask'].shape}")
# print(f"   Input IDs dtype: {batch['input_ids'].dtype}")
# print(f"   Labels dtype: {batch['labels'].dtype}")

# # Проверка на наличие паддинга
# print(f"\n🔍 ПРОВЕРКА ПАДДИНГА:")
# input_ids = batch['input_ids'][0]
# pad_count = (input_ids == tokenizer.pad_token_id).sum().item()
# real_token_count = len(input_ids) - pad_count
# print(f"   Токенов в примере: {len(input_ids)}")
# print(f"   Реальных токенов: {real_token_count}")
# print(f"   Паддинг токенов: {pad_count}")
# print(f"   Паддинг токен ID: {tokenizer.pad_token_id}")

# # Вывод нескольких примеров из validation loader для сравнения
# print("\n🔍 ДЛЯ СРАВНЕНИЯ - 2 ПРИМЕРА ИЗ VALIDATION DATALOADER:")
# print("-" * 60)

# val_examples_shown = 0
# for batch_idx, batch in enumerate(val_loader):
#     decoded_batch = decode_batch(batch, tokenizer)
    
#     for example_idx, example in enumerate(decoded_batch[:2]):  # только 2 примера
#         print(f"\n📝 Val Пример {val_examples_shown + 1}:")
#         print(f"   Контекст: '{example['input_text']}'")
#         print(f"   Цель:     '{example['label_text']}'")
        
#         val_examples_shown += 1
#         if val_examples_shown >= 2:
#             break
    
#     if val_examples_shown >= 2:
#         break

# print("\n✅ Вывод примеров завершен!")

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


# # создание модели
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# pad_token_id = tokenizer.pad_token_id  # добавляем определение pad_token_id
# model = MeanPoolingRNN(vocab_size=tokenizer.vocab_size, pad_idx=pad_token_id).to(device)


# # создание оптимизатора и функции потерь
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)#, weight_decay=3e-3) # создайте оптимайзер с регуляризацией
# criterion = nn.CrossEntropyLoss()


# # код обучения одной эпохи
# def train_epoch(model, loader):
#     model.train()
#     total_loss = 0
#     for batch in loader:
#         ids = batch['input_ids'].to(device)
#         mask = batch['attention_mask'].to(device)
#         labels = batch['label'].to(device)



#         optimizer.zero_grad()# обнулите градиенты
#         logits = model(ids, mask)# посчитайте выход модели
        
#         # print(f"Logits shape: {logits.shape}")  # должно быть [128, 2]
#         # print(f"Labels shape: {labels.shape}")  # должно быть [128]
#         # print(f"Labels dtype: {labels.dtype}")  # должно быть torch.int64
        
#         loss = criterion(logits, labels)# посчитайте функцию потерь
#         loss.backward()  # посчитайте градиенты
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # примените gradient clipping
#         optimizer.step() # обновите градиенты

#         total_loss += loss.item()

        
        
#     return total_loss / len(loader)


# # код подсчёта accuracy на валидации
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