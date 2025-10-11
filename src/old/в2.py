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

# v_path="../data/raw_dataset_tweets.txt"
# v_path="C:/Users/OMEN/Documents/LLM_Test/YaPracticum/project1_text-autocomplete/data/raw_dataset_tweets.txt"
# texts_df = pd.read_csv(file_path)
# print(f"Загружено строк: {len(dtexts_df)}")

# Просто читаем все строки
with open(file_path, 'r', encoding='utf-8') as f:
    texts = f.readlines()

# Убираем символы переноса строк
texts = [text.strip() for text in texts]
texts = texts[:1000]

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
for text in tqdm(texts_df['text_cleaned'].head(10)):  # df['clean'] — очищенные твиты
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
        print(i,x_tok,y_tok,x_padded, y_padded,attention_mask)
        print()
        data.append((x_padded, y_padded,attention_mask))
print("PAIRS ARE READY!")
# from sklearn.model_selection import train_test_split

# train, temp = train_test_split(data, test_size=0.2, random_state=42)
# val, test = train_test_split(temp, test_size=1/2, random_state=42)  # val 0.1, test 0.2

# # Проверка одного примера
# if len(train) > 0:
#     for i in range(10):
#         sample_x, sample_y, attention_mask = train[i]
#         x=[tokenizer.convert_ids_to_tokens(tok_id) for tok_id in list(sample_x)]
#         y=[tokenizer.convert_ids_to_tokens(tok_id) for tok_id in list(sample_y)]
                
#         print(f"Sample context: {x}")
#         print(f"Sample target: {y}")
#         print(f"Attention mask: {attention_mask}")
        
#     # print(f"Decoded context: {tokenizer.decode(sample_x)}")
#     # print(f"Decoded target: {tokenizer.decode(sample_y)}")
 
# #---------------------------------------------------------------------------------------------------------      