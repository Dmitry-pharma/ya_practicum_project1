from tqdm import tqdm  
import pandas as pd
import re 
from collections import Counter
from pathlib import Path
import html
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast
import torch
import json
import random
from datetime import datetime
# tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
from transformers import AutoTokenizer
#-----------parameters---------------------------
file_path = Path(__file__).parent.parent / 'data' / '1_raw_dataset_tweets.txt'
print(file_path)
model_dir = Path('C:/Users/OMEN/Documents/LLM_Test/YaPracticum/project1_text-autocomplete/models')  # или любой другой путь
model_dir.mkdir(exist_ok=True)
MAX_LEN = 20  # максимальная длина примера для выравнивания
limit=1000 #кол-во сообщений на входе (до разиения на выборки)
v_hidden_dim=512 #размер скрытого состояния
v_emb_dim=300 #размер входных эмбеддингов
v_batch_size=128
rnn_type="LSTM"  #"GRU"  "LSTM"  "RNN"
# v_path="../data/raw_dataset_tweets.txt"
# v_path="C:/Users/OMEN/Documents/LLM_Test/YaPracticum/project1_text-autocomplete/data/raw_dataset_tweets.txt"
# texts_df = pd.read_csv(file_path)
# print(f"Загружено строк: {len(dtexts_df)}")
#-------------------------------------------------------------------
# Просто читаем все строки
with open(file_path, 'r', encoding='utf-8') as f:
    texts = f.readlines()

# Убираем символы переноса строк
texts = [text.strip() for text in texts]
texts = texts[:limit]

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

print("Cleanning READY!")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


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

train_loader = DataLoader(train_ds, batch_size=v_batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=v_batch_size)
test_loader = DataLoader(test_ds, batch_size=v_batch_size)

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

class NextPhrasePredictionRNN(nn.Module):
    def __init__(self, rnn_type="RNN", vocab_size=30522, emb_dim=300, hidden_dim=256, output_dim=2, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        
        rnn_cls = {"RNN": nn.RNN, "GRU": nn.GRU, "LSTM": nn.LSTM}[rnn_type]
        self.rnn = rnn_cls(emb_dim, hidden_dim, batch_first=True) 

        # self.rnn = nn.RNN(emb_dim, hidden_dim, batch_first=True)# RNN-слой
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
model = NextPhrasePredictionRNN(
    rnn_type="LSTM",
    vocab_size=tokenizer.vocab_size,
    emb_dim=v_emb_dim,
    hidden_dim=v_hidden_dim,
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
def evaluate_xxx(model, loader):
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
# функция замера лосса и accuracy

def evaluate(model, loader):
    model.eval()
    preds, trues = [], []
    # correct, total = 0, 0
    # sum_loss = 0
    total_loss=0
    with torch.no_grad():
        for batch in loader:
            ids = batch['data'].to(device)
            mask = batch['mask'].to(device)
            labels = batch['target']
            logits = model(ids,mask)# выход модели для входа ids
            logits_flat = logits.reshape(-1, logits.size(-1))
            labels_flat = labels.reshape(-1)
            loss = criterion(logits_flat,labels_flat)# функция потерь
            total_loss += loss.item()
            # preds = torch.argmax(x_output, dim=1)# предсказанные токены
            preds += torch.argmax(logits, dim=-1).cpu().flatten().tolist()
            trues += labels.cpu().flatten().tolist()
            # correct += (preds == y_batch).sum().item()# количество верно угаданных токенов
            # total += y_batch.size(0)# размер батча
            # sum_loss += loss.item()# суммарная функция потерь
    
    # лосс и accuracy
    accuracy = accuracy_score(trues, preds)
    avg_loss = total_loss / len(loader)
    # avg_loss = sum_loss / len(loader)
    # accuracy = correct / total
    return accuracy, avg_loss 

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
            'emb_dim': v_emb_dim,
            'hidden_dim': v_hidden_dim,
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
#------------------------------------------------------------------------------------
def analyze_predictions(model, loader, tokenizer, num_examples=5):
    """Анализирует предсказания модели и показывает примеры"""
    model.eval()
    
    bad_cases = []  # неправильные предсказания
    good_cases = []  # правильные предсказания
    
    with torch.no_grad():
        for batch in loader:
            ids = batch['data'].to(device)
            mask = batch['mask'].to(device)
            labels = batch['target'].to(device)
            
            logits = model(ids, mask)
            preds = torch.argmax(logits, dim=-1)
            
            # Обрабатываем каждый пример в батче
            for i in range(ids.size(0)):
                # Получаем ненулевые токены (игнорируем паддинг)
                non_pad_indices = mask[i].bool()
                
                if non_pad_indices.sum() == 0:
                    continue
                
                # Берем только реальные токены (не паддинг)
                input_ids_real = ids[i][non_pad_indices]
                preds_real = preds[i][non_pad_indices]
                labels_real = labels[i][non_pad_indices]
                
                # Конвертируем в токены
                input_tokens = tokenizer.convert_ids_to_tokens(input_ids_real.cpu())
                true_tokens = tokenizer.convert_ids_to_tokens(labels_real.cpu())
                pred_tokens = tokenizer.convert_ids_to_tokens(preds_real.cpu())
                
                # Анализируем каждый токен в последовательности
                # Используем минимальную длину чтобы избежать IndexError
                min_length = min(len(input_tokens)-1, len(true_tokens), len(pred_tokens))
                
                for j in range(min_length):
                    # Пропускаем если выходим за границы массивов
                    if j >= len(true_tokens) or j >= len(pred_tokens):
                        continue
                        
                    context = input_tokens[:j+1]  # контекст до текущего токена
                    true_tok = true_tokens[j]
                    pred_tok = pred_tokens[j]
                    
                    # Пропускаем специальные токены и паддинг
                    skip_tokens = ['[PAD]', '[CLS]', '[SEP]', '<pad>', '<cls>', '<sep>', '']
                    if true_tok in skip_tokens or pred_tok in skip_tokens:
                        continue
                    
                    # Пропускаем если токены одинаковые но это специальные токены
                    if true_tok == pred_tok and true_tok in skip_tokens:
                        continue
                    
                    if true_tok != pred_tok:
                        bad_cases.append((context, true_tok, pred_tok))
                    else:
                        good_cases.append((context, true_tok, pred_tok))
            
            # Ограничиваем сбор примеров для экономии памяти
            if len(bad_cases) > 200 and len(good_cases) > 200:
                break
    
    # Проверяем что есть примеры для показа
    if not bad_cases and not good_cases:
        print("❌ Не найдено примеров для анализа. Возможные причины:")
        print("   - Все предсказания правильные")
        print("   - Проблемы с данными или токенизацией")
        print("   - Слишком строгая фильтрация токенов")
        return [], []
    
    # Выбираем случайные примеры
    random.seed(42)
    
    # Безопасный sampling с проверкой на пустые списки
    bad_cases_sampled = []
    good_cases_sampled = []
    
    if bad_cases:
        bad_cases_sampled = random.sample(bad_cases, min(num_examples, len(bad_cases)))
    if good_cases:
        good_cases_sampled = random.sample(good_cases, min(num_examples, len(good_cases)))
    
    # Выводим результаты
    print("\n" + "="*60)
    print("🔍 АНАЛИЗ ПРЕДСКАЗАНИЙ НА ТЕСТОВОЙ ВЫБОРКЕ")
    print("="*60)
    
    if bad_cases_sampled:
        print(f"\n❌ Примеры НЕПРАВИЛЬНЫХ предсказаний ({len(bad_cases_sampled)} из {len(bad_cases)}):")
        print("-" * 50)
        for i, (context, true_tok, pred_tok) in enumerate(bad_cases_sampled, 1):
            # Показываем последние 5 токенов контекста (или все если меньше)
            context_to_show = context[-5:] if len(context) > 5 else context
            context_str = ' '.join(context_to_show)
            
            print(f"{i}. Контекст: ...{context_str}")
            print(f"   Истинный токен: '{true_tok}' | Предсказанный: '{pred_tok}'")
            print(f"   Статус: {'🚫 ОШИБКА' if true_tok != pred_tok else '✅ ВЕРНО'}")
            
            # Дополнительная информация о токенах
            true_len = len(true_tok)
            pred_len = len(pred_tok)
            if true_len != pred_len:
                print(f"   Разница длины: {true_len} vs {pred_len}")
            print()
    else:
        print(f"\n✅ Нет неправильных предсказаний для показа!")
    
    if good_cases_sampled:
        print(f"\n✅ Примеры ПРАВИЛЬНЫХ предсказаний ({len(good_cases_sampled)} из {len(good_cases)}):")
        print("-" * 50)
        for i, (context, true_tok, pred_tok) in enumerate(good_cases_sampled, 1):
            context_to_show = context[-5:] if len(context) > 5 else context
            context_str = ' '.join(context_to_show)
            
            print(f"{i}. Контекст: ...{context_str}")
            print(f"   Истинный токен: '{true_tok}' | Предсказанный: '{pred_tok}'")
            print(f"   Статус: {'✅ ВЕРНО' if true_tok == pred_tok else '🚫 ОШИБКА'}")
            print()
    else:
        print(f"\n❌ Нет правильных предсказаний для показа!")
    
    # Статистика
    total_predictions = len(bad_cases) + len(good_cases)
    if total_predictions > 0:
        accuracy = len(good_cases) / total_predictions * 100
        print(f"\n📊 СТАТИСТИКА ПРЕДСКАЗАНИЙ:")
        print(f"   Всего предсказаний: {total_predictions}")
        print(f"   Правильных: {len(good_cases)} ({accuracy:.2f}%)")
        print(f"   Неправильных: {len(bad_cases)} ({100-accuracy:.2f}%)")
        
        # Дополнительная статистика
        if bad_cases:
            avg_context_length = sum(len(context) for context, _, _ in bad_cases) / len(bad_cases)
            print(f"   Средняя длина контекста при ошибках: {avg_context_length:.1f} токенов")
    else:
        print(f"\n📊 Не удалось собрать статистику предсказаний")
    
    return bad_cases, good_cases

# Дополнительная функция для анализа конкретных примеров
def show_detailed_examples(model, test_loader, tokenizer, num_examples=3):
    """Показывает детальные примеры работы модели"""
    model.eval()
    
    print("\n" + "="*60)
    print("🔬 ДЕТАЛЬНЫЙ АНАЛИЗ РАБОТЫ МОДЕЛИ")
    print("="*60)
    
    examples_shown = 0
    with torch.no_grad():
        for batch in test_loader:
            if examples_shown >= num_examples:
                break
                
            ids = batch['data'].to(device)
            mask = batch['mask'].to(device)
            labels = batch['target'].to(device)
            
            logits = model(ids, mask)
            preds = torch.argmax(logits, dim=-1)
            
            for i in range(min(num_examples - examples_shown, ids.size(0))):
                print(f"\n📝 Пример {examples_shown + 1}:")
                print("-" * 40)
                
                # Получаем ненулевые токены
                non_pad_indices = mask[i].bool()
                input_ids_real = ids[i][non_pad_indices]
                preds_real = preds[i][non_pad_indices]
                labels_real = labels[i][non_pad_indices]
                
                # Конвертируем в текст
                input_tokens = tokenizer.convert_ids_to_tokens(input_ids_real.cpu())
                input_text = tokenizer.convert_tokens_to_string(input_tokens)
                
                true_tokens = tokenizer.convert_ids_to_tokens(labels_real.cpu())
                true_text = tokenizer.convert_tokens_to_string(true_tokens)
                
                pred_tokens = tokenizer.convert_ids_to_tokens(preds_real.cpu())
                pred_text = tokenizer.convert_tokens_to_string(pred_tokens)
                
                print(f"Входной текст: {input_text}")
                print(f"Ожидаемый вывод: {true_text}")
                print(f"Предсказанный вывод: {pred_text}")
                
                # Сравнение по токенам
                print("\nСравнение по токенам:")
                min_len = min(len(true_tokens), len(pred_tokens))
                for j in range(min_len):
                    status = "✅" if true_tokens[j] == pred_tokens[j] else "❌"
                    print(f"  {status} Позиция {j}: '{true_tokens[j]}' vs '{pred_tokens[j]}'")
                
                examples_shown += 1
                print()
#-----------------------------------------------------------------------------------------
# Списки для хранения метрик
train_losses = []
val_accuracies = []
val_losses = []
best_val_acc = 0
best_epoch = 0
patience = 5
patience_counter = 0
# обучение
print("🎯 Начало обучения (предсказание последовательностей)...")
epoch_loop = tqdm(range(10), desc="Обучение", unit="эпоха")
for epoch in epoch_loop:
    # Обучение
    train_loss  = train_epoch(model, train_loader)    
    train_losses.append(train_loss)
    # Валидация
    val_acc, val_loss = evaluate(model, val_loader)
    val_accuracies.append(val_acc)
    val_losses.append(val_loss)
    # Сохранение лучшей модели
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch + 1
        patience_counter = 0        
        # Сохраняем лучшую модель
        best_model_path = model_dir / 'best_model.pth'
        save_model(model, optimizer, epoch, val_acc, val_loss, best_model_path)
        print(f"💾 Сохранена лучшая модель с accuracy: {val_acc:.4f}")
    else:
        patience_counter += 1
    # Обновляем описание tqdm: отображаем текущие loss и accuracy
    epoch_loop.set_postfix({
        "Loss": f"{train_loss:.4f}",
        "Val Acc": f"{val_acc:.4f}",
        "Best Acc": f"{best_val_acc:.4f}",
        "Patience": f"{patience_counter}/{patience}"
    })
    # print(f"Epoch {epoch+1}: Loss = {loss:.4f}, Accuracy = {acc:.4f}")
    # if (epoch + 1) % 10 == 0:
    #     print(f"\nЭпоха {epoch+1}: Loss = {loss:.4f}, Accuracy = {acc:.4f}")
    # Ранняя остановка
    if patience_counter >= patience:
        print(f"🛑 Ранняя остановка на эпохе {epoch + 1}")
        break

# Сохраняем финальную модель
final_model_path = model_dir / 'final_model.pth'
save_model(model, optimizer, epoch, val_acc, train_loss, final_model_path)
print(f"💾 Сохранена финальная модель")

import matplotlib.pyplot as plt

# Загружаем лучшую модель для тестирования
print(f"🔄 Загружаем лучшую модель из эпохи {best_epoch}...")
checkpoint = torch.load(model_dir / 'best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Финальное тестирование на тестовом наборе
print("\n🧪 Начинаем финальное тестирование...")
test_accuracy, test_loss = test_model(model, test_loader)



# # Сохраняем результаты тестирования
# results = {
#     'best_validation_accuracy': best_val_acc,
#     'best_epoch': best_epoch,
#     'test_accuracy': test_accuracy,
#     'test_loss': test_loss,
#     'final_training_epochs': epoch + 1
# }
results = {
    # Основные метрики
    'best_validation_accuracy': best_val_acc,
    'best_epoch': best_epoch,
    'test_accuracy': test_accuracy,
    'test_loss': test_loss,
    'final_training_epochs': epoch + 1,
    
    # Статистика по выборкам
    'dataset_statistics': {
        'number_of_input_samples':limit,
        'train_samples': len(train_ds),
        'validation_samples': len(val_ds),
        'test_samples': len(test_ds),
        'total_samples': len(train_ds) + len(val_ds) + len(test_ds),
        'train_ratio': len(train_ds) / (len(train_ds) + len(val_ds) + len(test_ds)),
        'validation_ratio': len(val_ds) / (len(train_ds) + len(val_ds) + len(test_ds)),
        'test_ratio': len(test_ds) / (len(train_ds) + len(val_ds) + len(test_ds))
    },
    
    # Параметры модели
    'model_architecture': {
        'model_type': 'NextPhrasePredictionRNN',
        'rnn_type': rnn_type,  # или 'LSTM', 'GRU' если измените
        'vocab_size': tokenizer.vocab_size,
        'embedding_dim': v_emb_dim,
        'hidden_dim': v_hidden_dim,
        'max_sequence_length': MAX_LEN,
        'dropout_rate': 0.5
    },
    
    # Параметры обучения
    'training_parameters': {
        'batch_size': v_batch_size,
        'learning_rate': 3e-3,
        'optimizer': 'AdamW',
        'loss_function': 'CrossEntropyLoss',
        'early_stopping_patience': patience,
        'gradient_clip_norm': 1.0
    },
    
    # Дополнительная статистика
    'additional_statistics': {
        'total_training_steps': len(train_loader) * (epoch + 1),
        'final_train_loss': train_losses[-1] if train_losses else None,
        'final_validation_loss': val_losses[-1] if val_losses else None,
        'training_time_epochs': epoch + 1
    }
}

timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M')
results_filename = f'training_results_{timestamp}.json'
with open(model_dir / results_filename, 'w') as f:
    json.dump(results, f, indent=2)
print(f"💾 Результаты сохранены в: {results_filename}")

print(f"\n📊 Итоговые результаты:")
print(f"   Лучшая валидационная accuracy: {best_val_acc:.4f}")
print(f"   Лучшая эпоха: {best_epoch}")
print(f"   Тестовая accuracy: {test_accuracy:.4f}")
print(f"   Тестовая loss: {test_loss:.4f}")

#--------------------------------------------------------------------------------------------
# Добавляем анализ предсказаний:
print("\n" + "="*60)
print("Начинаем детальный анализ предсказаний...")
print("="*60)

# Анализ случайных примеров
bad_cases, good_cases = analyze_predictions(model, test_loader, tokenizer, num_examples=5)

# Детальный анализ конкретных примеров
show_detailed_examples(model, test_loader, tokenizer, num_examples=3)

# Дополнительная статистика по типам ошибок
def analyze_error_patterns(bad_cases, tokenizer):
    """Анализирует паттерны ошибок"""
    print("\n" + "="*60)
    print("📈 АНАЛИЗ ПАТТЕРНОВ ОШИБОК")
    print("="*60)
    
    if not bad_cases:
        print("Нет ошибок для анализа! 🎉")
        return
    
    # Анализ самых частых ошибок
    error_pairs = [(true, pred) for _, true, pred in bad_cases]
    error_counter = Counter(error_pairs)
    
    print("\n🔝 Топ-10 самых частых ошибок:")
    for (true_tok, pred_tok), count in error_counter.most_common(10):
        print(f"  '{true_tok}' → '{pred_tok}': {count} раз")
    
    # Анализ по длине токенов
    length_errors = []
    for _, true_tok, pred_tok in bad_cases:
        length_diff = abs(len(true_tok) - len(pred_tok))
        length_errors.append(length_diff)
    
    if length_errors:
        avg_length_diff = sum(length_errors) / len(length_errors)
        print(f"\n📏 Средняя разница в длине токенов при ошибках: {avg_length_diff:.2f}")

# Вызываем анализ паттернов ошибок
analyze_error_patterns(bad_cases, tokenizer)

#--------------------------------------------------------------------------------------------

# Визуализация результатов
plt.figure(figsize=(15, 5))

# График потерь
plt.subplot(1, 3, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', color='tab:blue', linewidth=2,marker=None)
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss', color='tab:red', linewidth=2,marker=None)
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

# График точности
plt.subplot(1, 3, 2)
plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy', color='tab:orange', linewidth=2,marker=None)
plt.axhline(y=best_val_acc, color='red', linestyle='--', label=f'Best Acc: {best_val_acc:.4f}')
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.grid(True)
plt.legend()

# График сравнения метрик
plt.subplot(1, 3, 3)
metrics = ['Best Val Acc', 'Test Acc']
values = [best_val_acc, test_accuracy]
colors = ['lightblue', 'lightgreen']
bars = plt.bar(metrics, values, color=colors)
plt.title('Final Metrics Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{value:.4f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n✅ Обучение завершено!")
print(f"📁 Модели сохранены в: {model_dir}")
print(f"📊 Графики сохранены в: training_results.png")