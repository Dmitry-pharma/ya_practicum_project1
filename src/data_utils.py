from tqdm import tqdm  
import pandas as pd
import re 
from pathlib import Path
import html
from transformers import AutoTokenizer

def clean_text(text):
    text = text.lower()  # к нижнему регистру  
    text = html.unescape(text)# преобразует &quot; -> ", &amp; -> &, и т.д.# 2. Декодировать HTML символы 
    text = re.sub(r'<[^>]+>', '', text)# 3. Удалить оставшиеся HTML теги
    text = re.sub(r'@\w+', 'user', text)# заменить упоминания (@username)  
    text = re.sub(r'#\w+', '', text)# Удалить хештеги (#tag)  
    text = re.sub(r'[^\w\s,\.!?;:]', '', text)# Удалить эмодзи и специальные символы
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)# Удалить ссылки 
    text = re.sub(r"[^a-z0-9 ]+", " ", text)# оставить только буквы и цифры
    text = re.sub(r"\s+", " ", text).strip()# убрать дублирующиеся пробелы
    return text

def load_and_clean_data(file_path, limit=10000):
    with open(file_path, 'r', encoding='utf-8') as f:
        texts = f.readlines()
    
    texts = [text.strip() for text in texts]
    texts = texts[:limit]
    
    print(f"Загружено {len(texts)} строк")
    
    print("Cleaning...")
    cleaned_texts = []
    for text in tqdm(texts):
        cleaned_texts.append(clean_text(text))
    #для целей отладки
    text_pairs = [(i, texts[i], cleaned_texts[i]) for i in range(len(texts))]
    texts_df = pd.DataFrame(text_pairs, columns=['rowno','text_raw','text_cleaned'])
    
    print("Cleaning READY!")
    return texts_df

def prepare_training_pairs(texts_df, tokenizer, MAX_LEN=20):
    print("Preparing X, Y pairs...")
    data = []
    
    for text in tqdm(texts_df['text_cleaned']):
        tokens = tokenizer.tokenize(text)
        if len(tokens) < 2:
            continue
            
        for i in range(1, len(tokens)):
            start_idx = max(0, i - MAX_LEN)
            x_tok = tokens[start_idx:i]
            y_tok = tokens[start_idx+1:i+1]
            
            if len(x_tok) < 1 or len(y_tok) < 1:
                continue
                
            x_ids = tokenizer.convert_tokens_to_ids(x_tok)
            y_ids = tokenizer.convert_tokens_to_ids(y_tok)
            
            x_padded = [tokenizer.pad_token_id] * (MAX_LEN - len(x_ids)) + x_ids
            y_padded = [tokenizer.pad_token_id] * (MAX_LEN - len(y_ids)) + y_ids
            attention_mask = [1 if token_id != tokenizer.pad_token_id else 0 for token_id in x_padded]
            
            data.append((x_padded, y_padded, attention_mask))
    
    print("PAIRS ARE READY!")
    return data