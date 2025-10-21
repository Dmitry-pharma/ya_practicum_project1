from tqdm import tqdm  
import pandas as pd
import re 
from pathlib import Path
import html
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import pickle

from src.next_token_dataset import TweetsDataset

def clean_text(text):
    text = text.lower()  # к нижнему регистру  
    text = html.unescape(text)# преобразует &quot; -> ", &amp; -> &, и т.д.# 2. Декодировать HTML символы 
    text = re.sub(r'<[^>]+>', '', text)# 3. Удалить оставшиеся HTML теги
    # text = re.sub(r'@\w+', 'user', text)# заменить упоминания (@username)  
    text = re.sub(r'@\w+', '', text)# заменить упоминания (@username)  
    
    
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


def prepare_training_pairs3(texts_df, tokenizer, MAX_LEN=20):
    print("Preparing X, Y pairs...")
    data = []
    
    # for rowno, text in tqdm(zip(texts_df['rowno'], texts_df['text_cleaned']), total=len(texts_df)):
    for text in tqdm( texts_df['text_cleaned']):
    
        # print(rowno, text)
        #если значение не текст, то пропускаем его обработку
        if text is None or isinstance(text, float):
            continue
        
        tokens = tokenizer.tokenize(text)
        
        if len(tokens) < 2:
            continue
        
        
        for i in range(0, len(tokens) - MAX_LEN, MAX_LEN // 2):  # Перекрытие 50%
            chunk = tokens[i:i + MAX_LEN + 1]  # +1 для целевого токена
            
            if len(chunk) < 2:
                continue
                
        
            start_idx = max(0, i - MAX_LEN)
            x_tok = chunk[start_idx:i]
            y_tok = chunk[start_idx+1:i+1]
            
            if len(x_tok) < 1 or len(y_tok) < 1:
                continue
            
            # print(x_tok,y_tok)    
            
            x_ids = tokenizer.convert_tokens_to_ids(x_tok)
            y_ids = tokenizer.convert_tokens_to_ids(y_tok)
            
            x_padded = [tokenizer.pad_token_id] * (MAX_LEN - len(x_ids)) + x_ids
            y_padded = [tokenizer.pad_token_id] * (MAX_LEN - len(y_ids)) + y_ids
            attention_mask = [1 if token_id != tokenizer.pad_token_id else 0 for token_id in x_padded]
            
            data.append((x_padded, y_padded, attention_mask))
    
    print("PAIRS ARE READY!")
    return data




# Функция для сохранения датасетов
def save_file(dataset, file_path):
    """Сохраняет датасет в файл"""
    # with open(file_path, 'wb') as f:
    #     pickle.dump(dataset, f)
    # print(f"💾 Датасет сохранен: {file_path}")
    # Создаем CSV файл с тем же именем но другим расширением
    # csv_path = file_path.with_suffix('.csv')
    
    try:
        # Преобразуем датасет в DataFrame для CSV
        if hasattr(dataset, '__getitem__') and hasattr(dataset, '__len__'):
            # Для TweetsDataset
            data_for_csv = []
            for i in range(min(1000, len(dataset))):  # Ограничиваем для больших датасетов
                try:
                    item = dataset[i]
                    row = {
                        'index': i,
                        'input_ids': item['data'].tolist(),
                        'target_ids': item['target'].tolist(),
                        'attention_mask': item['mask'].tolist()
                    }
                    data_for_csv.append(row)
                except Exception as e:
                    continue
            
            df = pd.DataFrame(data_for_csv)
            df.to_csv(file_path, index=False, encoding='utf-8')
            print(f"💾 Примеры датасета сохранены в CSV: {file_path} ({len(df)} строк)")
            
        else:
            print(f"⚠️  Не удалось создать CSV для датасета типа {type(dataset)}")
            
    except Exception as e:
        print(f"❌ Ошибка при сохранении CSV: {e}")

def save_samples(texts_df, file_path):
    """Сохраняет таблицу с текстами в CSV файл"""
    texts_df[['rowno', 'text_raw', 'text_cleaned']].to_csv(file_path, index=False, encoding='utf-8')
    print(f"Таблица сохранена: {file_path}")

def load_samples(file_path):
    """Сохраняет таблицу с текстами в CSV файл"""
    # texts_df[['rowno', 'text_raw', 'text_cleaned']].to_csv(file_path, index=False, encoding='utf-8')
    texts_df= pd.read_csv(file_path, encoding='utf-8')
    print(f"Таблица загружена: {file_path}")
    return texts_df

# Функция для загрузки датасетов
def load_datasets(file_path):
    """Загружает датасеты из указанной директории"""
    
    with open(file_path, 'rb') as f:
        v_dataset = pickle.load(f)
        
    # print(f"📂 Датасеты загружены из: {config['datasets_dir']}")
    return v_dataset

# Пример загрузки (для использования в других скриптах)
# train_ds_loaded, val_ds_loaded, test_ds_loaded = load_datasets(config)    
# samples_preparation(file_path=config['file_path'],limit=config['limit'],tokenizer=tokenizer,MAX_LEN=config['MAX_LEN'],batch_size=config['batch_size'])
def samples_preparation(data_dir, source_file, limit):
       
    file_path = data_dir / source_file
    texts_df = load_and_clean_data(file_path, limit)

    # Разделение данных
    train, temp = train_test_split(texts_df, test_size=0.2, random_state=42)
    val, test = train_test_split(temp, test_size=1/2, random_state=42)
    
    # print(test)
    
    # Сохраняем все выборки
    save_samples(train, data_dir / 'train_dataset.csv')
    save_samples(val, data_dir / 'val_dataset.csv')
    save_samples(test, data_dir / 'test_dataset.csv')
    return train,val,test
    
def dataset_preparation(data, tokenizer, MAX_LEN=20, batch_size=128, shuffle=True):         
    X, Y, M = zip(*prepare_training_pairs3(data, tokenizer, MAX_LEN))
   
    data_ds = TweetsDataset(X, Y, M)
    data_loader = DataLoader(data_ds, batch_size=batch_size, shuffle=shuffle)
    print(f"✅ Датасет создан: Size={len(data_ds)}")
    return data_ds, data_loader

# --------------------------------------------
# import os
# from torch.utils.data import Dataset
# import torch

# class TweetsDataset(Dataset):
#     def __init__(self, data, targets, mask):
#         self.data = data
#         self.targets = targets 
#         self.mask = mask

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):      
#         return {
#             'data': torch.tensor(self.data[idx], dtype=torch.long),
#             'target': torch.tensor(self.targets[idx], dtype=torch.long),
#             'mask': torch.tensor(self.mask[idx], dtype=torch.long)
#         }
        

# current_dir = Path(os.getcwd())
# v_file_path=Path(current_dir) /'data'/ 'test_dataset.csv'
# print(v_file_path)
# #формируем/загружаем данные
# test= load_samples(v_file_path )        
# transformer_model_name = "distilgpt2"

# tokenizer = AutoTokenizer.from_pretrained(transformer_model_name)
# test_ds,  test_loader = dataset_preparation(test, tokenizer, MAX_LEN=20, batch_size=10)

# for i in range(len(test_ds)):
#         print(test_ds[i])


