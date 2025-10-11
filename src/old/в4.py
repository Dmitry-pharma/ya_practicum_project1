from tqdm import tqdm  # ← ДОБАВЛЕНО
import pandas as pd
import re 
from collections import Counter
from pathlib import Path
import html
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast
import torch
from torch.utils.data import Dataset, DataLoader
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
from transformers import AutoTokenizer
file_path = Path(__file__).parent.parent / 'data' / '1_raw_dataset_tweets.txt'
print(file_path)
model_dir="C:/Users/OMEN/Documents/LLM_Test/YaPracticum/project1_text-autocomplete/models"
# v_path="../data/raw_dataset_tweets.txt"
# v_path="C:/Users/OMEN/Documents/LLM_Test/YaPracticum/project1_text-autocomplete/data/raw_dataset_tweets.txt"
# texts_df = pd.read_csv(file_path)
# print(f"Загружено строк: {len(dtexts_df)}")

# Просто читаем все строки
with open(file_path, 'r', encoding='utf-8') as f:
    texts = f.readlines()

# Убираем символы переноса строк
texts = [text.strip() for text in texts]
texts = texts[:100]

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
print("Cleanning...")
cleaned_texts =[]
for text in tqdm(texts):
    cleaned_texts.append(clean_text(text))
    
text_pairs = [(i, texts[i], cleaned_texts[i]) for i in range(len(texts))]

#сохраняем очищенный текст в csv
texts_df = pd.DataFrame(text_pairs, columns=['rowno','text_raw','text_cleaned'])
# output_path = Path(__file__).parent.parent / 'data' / '2_cleaned_dataset_tweets.csv'
# texts_df.to_csv(output_path, index=False, encoding='utf-8')

print("Cleanning READY!")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

MAX_LEN = 20  # максимальная длина примера для выравнивания
print("Prepearing X, Y pairs...")
data = []
for text in tqdm(texts_df['text_cleaned']):  # df['clean'] — очищенные твиты
    tokens = tokenizer.tokenize(text)
    if len(tokens) < 2:# пропускаем слишком короткие тексты
        continue
    # Для длинных твитов разбиваем на скользящие окна размером MAX_LEN
    for i in range(1, len(tokens)):
        start_idx = max(0, i - MAX_LEN)
        x_tok = tokens[start_idx:i]# токены с start_idx до i-1        
        y_tok = tokens[start_idx+1:i+1]# Цель: те же токены, но сдвинутые на 1 вперед (следующий токен)
        if len(x_tok) < 1 or len(y_tok) < 1:
            continue
        # padding слева для X/Y до MAX_LEN
        # x_ids = [0] * (MAX_LEN-len(x_tok)) + [vocab.get(tok, 1) for tok in x_tok]
        # y_ids = [0] * (MAX_LEN-len(y_tok)) + [vocab.get(tok, 1) for tok in y_tok]
        # data.append((x_ids, y_ids))
        # data.append((x_tok,y_tok))
         # Преобразуем токены в ID
        x_ids = tokenizer.convert_tokens_to_ids(x_tok)
        y_ids = tokenizer.convert_tokens_to_ids(y_tok)                
        # Паддинг слева для X/Y до max_len
        x_padded = [tokenizer.pad_token_id] * (MAX_LEN - len(x_ids)) + x_ids
        y_padded = [tokenizer.pad_token_id] * (MAX_LEN - len(y_ids)) + y_ids                
        # Создаем attention_mask (1 для реальных токенов, 0 для паддинга)
        attention_mask = [1 if token_id != tokenizer.pad_token_id else 0 for token_id in x_padded]
        # print(i,x_tok,y_tok,x_padded, y_padded,attention_mask)
        # print()
        data.append((x_padded, y_padded,attention_mask))
print("PAIRS ARE READY!")


class TweetsDataset(Dataset):
    def __init__(self, data, targets,mask):
        # Сохраните data и targets в атрибуты
        # Ваш код здесь
        self.data = data
        self.targets = targets 
        self.mask=mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        return {'data':torch.tensor(self.data[idx], dtype=torch.long),
                'target':torch.tensor(self.targets[idx], dtype=torch.long),
                'mask':torch.tensor(self.mask[idx], dtype=torch.long)                
            }

from sklearn.model_selection import train_test_split 
train, temp = train_test_split(data, test_size=0.2, random_state=42)
val, test = train_test_split(temp, test_size=1/2, random_state=42)  # val 0.1, test 0.2

#транспонирование списка кортежей через zip
X_train, Y_train, M_train = zip(*train)
X_val, Y_val, M_val = zip(*val)
X_test, Y_test, M_test = zip(*test)

train_ds = TweetsDataset(X_train, Y_train, M_train)
val_ds = TweetsDataset(X_val, Y_val, M_val)
test_ds = TweetsDataset(X_test, Y_test, M_test)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=128)
test_loader = DataLoader(test_ds, batch_size=128)

print(f"✅ Датасеты созданы: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

# for batch_idx, batch in enumerate(train_loader):
#     if batch_idx >= 1:
#         break
#     print("BATCH Number=",batch_idx)
#     for batch_item_num, batch_item in enumerate(batch):
#         print(batch_item_num)
#         print("x=",batch['data'][batch_item_num])
#         print("y=",batch['target'][batch_item_num])
#         print("m=",batch['mask'][batch_item_num])
 
# Проверим размерности
# print("\n🔍 Проверка размерностей:")
# sample_batch = next(iter(train_loader))
# print(f"inputs shape: {sample_batch['data'].shape}")  # [128, 20]
# print(f"labels shape: {sample_batch['target'].shape}")        # [128, 20] - последовательность!
# print(f"attention_mask shape: {sample_batch['mask'].shape}")  # [128, 20]


import torch.nn as nn
from sklearn.metrics import accuracy_score

class MeanPoolingRNN(nn.Module):
    def __init__(self, vocab_size, emb_dim=300, hidden_dim=256, output_dim=2, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.RNN(emb_dim, hidden_dim, batch_first=True)# RNN-слой
        self.norm = nn.LayerNorm(hidden_dim) # слой нормализации
        self.dropout = nn.Dropout(0.5)# dropout-слой
        self.fc = nn.Linear(hidden_dim, vocab_size) # линейный слой для классификации
        self.init_weights()


    def init_weights(self):
        # напишите xavier инициализацию весов
        # if isinstance(self, nn.Linear) or isinstance(self, nn.RNN):
        #     nn.init.xavier_uniform_(self.weight)
        #     if self.bias is not None:
        #         nn.init.zeros_(self.bias)
        # nn.init.xavier_uniform_(self.fc.weight)
        # for name, param in self.rnn.named_parameters():
        #     if 'weight' in name:
        #         nn.init.xavier_uniform_(param)
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)


    def forward(self, input_ids, attention_mask):
        x = self.embedding(input_ids)
        rnn_out, _ = self.rnn(x)# посчитайте результат rnn-слоя
        
        rnn_out_normed = self.norm(rnn_out)# посчитайте нормирование для rnn_out


        # mean pooling по attention_mask
        # mask = attention_mask.unsqueeze(2).expand_as(rnn_out)# подготовьте маску для rnn_out_normed
        mask = attention_mask.unsqueeze(-1).float()  # [batch_size, seq_len, 1]
        # mask = mask.unsqueeze(-1).float()
        masked_out =  rnn_out_normed * mask# умножьте rnn_out_normed на маску
        summed = masked_out.sum(dim=1)# посчитайте сумму masked_out по dim=1
        # lengths = attention_mask.sum(dim=1).unsqueeze(1)# посчитайте длины последовательностей
        lengths = attention_mask.sum(dim=1).unsqueeze(1).clamp(min=1e-9)  # [batch_size, 1]
        mean_pooled = summed / lengths# посчитайте средние, разделив суммы на длины последовательностей
         # 5. Классификация для каждого элемента последовательности
        # Повторяем общее представление для каждого токена в последовательности
        seq_representation = mean_pooled.unsqueeze(1).repeat(1, input_ids.size(1), 1)  # [batch_size, seq_len, hidden_dim]
        
        # out = self.dropout(mean_pooled)# примените dropout
        out = self.dropout(seq_representation)
        
        logits = self.fc(out)# примените линейный слой 
        return logits


# создание модели
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pad_token_id = tokenizer.pad_token_id  # добавляем определение pad_token_id
# model = MeanPoolingRNN(vocab_size=tokenizer.vocab_size, pad_idx=pad_token_id).to(device)
model = MeanPoolingRNN(
    vocab_size=tokenizer.vocab_size,
    emb_dim=256,
    hidden_dim=512,
    pad_idx=pad_token_id
).to(device)

# создание оптимизатора и функции потерь
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)#, weight_decay=3e-3) # создайте оптимайзер с регуляризацией
criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)


# код обучения одной эпохи
def train_epoch(model, loader):
    model.train()
    total_loss = 0
    progress_bar = tqdm(loader, desc="Train", leave=False)
    for batch in progress_bar:
        ids = batch['data'].to(device)
        mask = batch['mask'].to(device)
        labels = batch['target'].to(device)



        optimizer.zero_grad()# обнулите градиенты
        logits = model(ids, mask)# посчитайте выход модели
        
        # print(f"Logits shape: {logits.shape}")  # должно быть [128, 2]
        # print(f"Labels shape: {labels.shape}")  # должно быть [128]
        # print(f"Labels dtype: {labels.dtype}")  # должно быть torch.int64
        logits_flat = logits.reshape(-1, logits.size(-1))
        labels_flat = labels.reshape(-1)
        
        # loss = criterion(logits, labels)# посчитайте функцию потерь
        loss = criterion(logits_flat, labels_flat)
        loss.backward()  # посчитайте градиенты
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # примените gradient clipping
        optimizer.step() # обновите градиенты

        total_loss += loss.item()
        # Обновляем postfix tqdm: отображаем текущий loss
        progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

        
        
    return total_loss / len(loader)


# код подсчёта accuracy на валидации
def evaluate(model, loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch['data'].to(device)
            mask = batch['mask'].to(device)
            labels = batch['target']
            logits = model(ids, mask)
            # preds += torch.argmax(logits, dim=1).cpu().tolist()
            # trues += labels.tolist()
            preds += torch.argmax(logits, dim=-1).cpu().flatten().tolist()
            trues += labels.flatten().tolist()
    return accuracy_score(trues, preds)

def save_model(model, optimizer, epoch, accuracy, loss, path):
    """Сохраняет модель и её параметры"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy,
        'loss': loss,
        'model_config': {
            'vocab_size': tokenizer.vocab_size,
            'emb_dim': 256,
            'hidden_dim': 512,
            'pad_idx': pad_token_id
        }
    }, path)

def load_model(model, optimizer, path):
    """Загружает модель и её параметры"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['accuracy'], checkpoint['loss']

def test_model(model, loader):
    """Функция для финального тестирования модели"""
    model.eval()
    all_preds = []
    all_trues = []
    test_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing"):
            ids = batch['data'].to(device)
            mask = batch['mask'].to(device)
            labels = batch['target'].to(device)
            
            logits = model(ids, mask)
            
            # Вычисление loss
            logits_flat = logits.reshape(-1, logits.size(-1))
            labels_flat = labels.reshape(-1)
            loss = criterion(logits_flat, labels_flat)
            test_loss += loss.item()
            
            # Получаем предсказания
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().flatten().tolist())
            all_trues.extend(labels.cpu().flatten().tolist())
    
    accuracy = accuracy_score(all_trues, all_preds)
    avg_loss = test_loss / len(loader)
    
    print(f"🎯 Результаты тестирования:")
    print(f"   Test Loss: {avg_loss:.4f}")
    print(f"   Test Accuracy: {accuracy:.4f}")
    print(f"   Количество примеров: {len(all_trues)}")
    
    return accuracy, avg_loss

# Списки для хранения метрик
train_losses = []
val_accuracies = []

# обучение
print("🎯 Начало обучения (предсказание последовательностей)...")
epoch_loop = tqdm(range(10), desc="Обучение", unit="эпоха")
for epoch in epoch_loop:
    loss = train_epoch(model, train_loader)
    acc = evaluate(model, val_loader)
    train_losses.append(loss)
    val_accuracies.append(acc)
    # Обновляем описание tqdm: отображаем текущие loss и accuracy
    epoch_loop.set_postfix({
        "Loss": f"{loss:.4f}",
        "Acc": f"{acc:.4f}"
    })
    # print(f"Epoch {epoch+1}: Loss = {loss:.4f}, Accuracy = {acc:.4f}")
    # if (epoch + 1) % 10 == 0:
    #     print(f"\nЭпоха {epoch+1}: Loss = {loss:.4f}, Accuracy = {acc:.4f}")

# Сохраняем финальную модель
final_model_path = model_dir / 'final_model.pth'
save_model(model, optimizer, epoch, val_acc, train_loss, final_model_path)
print(f"💾 Сохранена финальная модель")

import matplotlib.pyplot as plt

# Создаём график
plt.figure(figsize=(12, 5))

# График потерь (loss)
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', color='tab:blue')#marker='o'
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

# График точности (accuracy)
plt.subplot(1, 2, 2)
plt.plot(range(1, len(val_accuracies) + 1), val_accuracies,  label='Validation Accuracy', color='tab:orange')#marker='s',
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
plt.show()