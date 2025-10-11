import pandas as pd
import re 
from collections import Counter
from pathlib import Path
import html
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast
import torch
from torch.utils.data import Dataset, DataLoader

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

print(f"Загружено {len(texts)} строк")
for i, text in enumerate(texts[:5]):  # первые 5 строк
    print(f"{i+1}: {text}")
    
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

cleaned_texts = [clean_text(text) for text in texts]
text_pairs = [(i, texts[i], cleaned_texts[i]) for i in range(len(texts))]

#сохраняем очищенный текст в csv
texts_df = pd.DataFrame(text_pairs, columns=['rowno','text_raw','text_cleaned'])
output_path = Path(__file__).parent.parent / 'data' / '2_cleaned_dataset_tweets.csv'
texts_df.to_csv(output_path, index=False, encoding='utf-8')

print("READY!")

# разбиение на тренировочную и валидационную выборки
val_size = 0.05

train_texts, val_texts = train_test_split(cleaned_texts[:max_texts_count], test_size=val_size, random_state=42)
print(f"Train texts: {len(train_texts)}, Val texts: {len(val_texts)}")

# класс датасета
class MaskedBertDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_len=7):
        # self.samples - список пар (x, y)
        # x - токенизированный текст с пропущенным токеном
        # y - пропущенный токен
        self.samples = []

        for vi,line in enumerate(texts):
            # token_ids =  tokenizer.encode(line)#, padding='max_length', truncation=True, max_length=seq_len, return_tensors='pt')# токенизируйте строку line
            token_ids = tokenizer.encode(line, add_special_tokens=False, max_length=512, truncation=True)
            # if vi<=10:
            #     print(vi,line,token_ids)
            
            # если строка слишком короткая, то пропускаем её
            # print(len(token_ids) , seq_len)
            if len(token_ids) < seq_len:
                # print("...do not use this string...")
                continue


            # проходимся по всем токенам в последовательности
            for i in range(1, len(token_ids) - 1):
                '''
                context - список из seq_len // 2 токенов до i-го токена, токена tokenizer.mask_token_id, и seq_len // 2 токенов после i-го токена
                '''
                # соберите контекст вокруг i-го токена
                context = token_ids[max(0, i - seq_len//2): i] + [tokenizer.mask_token_id] + token_ids[i+1: i+1+seq_len//2]
               
                # если контекст слишком короткий, то пропускаем его
                if len(context) < seq_len:
                    continue


                target = token_ids[i]# возьмите i-ый токен последовательности

                # if vi<=10:
                #     print(i,max(0, i - seq_len//2),i+1+seq_len//2)
                #     print(i,context,target)
                
                self.samples.append((context, target))
           
    def __len__(self):
        return len(self.samples)# верните размер датасета


    def __getitem__(self, idx):
        x, y = self.samples[idx]# получите контекст и таргет для элемента с индексом idx
        return torch.tensor(x), torch.tensor(y)


# загружаем токенизатор
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")


# тренировочный и валидационный датасеты
train_dataset = MaskedBertDataset(train_texts, tokenizer, seq_len=seq_len)
val_dataset = MaskedBertDataset(val_texts, tokenizer, seq_len=seq_len)


# даталоадеры
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

print("Train dataset size:",len(train_dataset)," Val dataset size: ",len(val_dataset))
print("Train loader size: ",len(train_loader)," Val loader size: ", len(val_loader)) 

# Проверка одного примера
# if len(train_dataset) > 0:
#     sample_x, sample_y = train_dataset[0]
#     print(f"Sample context: {sample_x}")
#     print(f"Sample target: {sample_y}")
#     print(f"Decoded context: {tokenizer.decode(sample_x)}")
#     print(f"Decoded target: {tokenizer.decode(sample_y)}")
 
#---------------------------------------------------------------------------------------------------------      

