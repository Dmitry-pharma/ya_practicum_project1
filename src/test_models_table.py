import torch
from torch.utils.data import DataLoader
from pathlib import Path
from transformers import AutoTokenizer, GPT2LMHeadModel
import random
import pandas as pd
from data_utils import load_and_clean_data, prepare_training_pairs
from next_token_dataset import TweetsDataset
from lstm_model import NextPhrasePredictionRNN
from eval_lstm import vevaluate
from sklearn.model_selection import train_test_split
import evaluate

def create_test_dataset(file_path, limit=1000, max_len=20):
    """Создает тестовую выборку на 1000 записей"""
    print("📊 Создание тестовой выборки...")
    
    # Загрузка и очистка данных
    texts_df = load_and_clean_data(file_path, limit)
    
    # Используем BertTokenizer для LSTM модели
    lstm_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Подготовка данных для LSTM
    data = prepare_training_pairs(texts_df, lstm_tokenizer, max_len)
    
    # Используем все данные для тестирования (не разделяем на train/test)
    test_data = data
    
    X_test, Y_test, M_test = zip(*test_data)
    
    # Создаем dataset и loader для LSTM
    test_ds = TweetsDataset(X_test, Y_test, M_test)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    
    print(f"✅ Тестовая выборка создана: {len(test_ds)} примеров")
    
    return test_loader, lstm_tokenizer, texts_df

def load_lstm_model(model_path, device):
    """Загружает LSTM модель"""
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint['model_config']
    
    model = NextPhrasePredictionRNN(
        rnn_type="LSTM",
        vocab_size=model_config['vocab_size'],
        emb_dim=model_config['emb_dim'],
        hidden_dim=model_config['hidden_dim'],
        pad_idx=model_config['pad_idx']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, model_config

def generate_lstm_completion(model, tokenizer, text, device, max_length=20, num_tokens=5):
    """Генерирует продолжение текста с помощью LSTM модели"""
    model.eval()
    
    # Токенизируем входной текст
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = inputs['input_ids'].to(device)
    
    generated_tokens = []
    
    with torch.no_grad():
        for _ in range(num_tokens):
            # Получаем предсказание следующего токена
            outputs = model(input_ids, torch.ones_like(input_ids))
            next_token_logits = outputs[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)
            
            # Добавляем предсказанный токен к входу
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            generated_tokens.append(next_token.item())
            
            # Останавливаемся если достигли максимальной длины или специального токена
            if next_token.item() == tokenizer.sep_token_id or len(generated_tokens) >= num_tokens:
                break
    
    # Декодируем сгенерированные токены
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return generated_text

def generate_gpt2_completion(model, tokenizer, text, device, max_length=50, num_tokens=10):
    """Генерирует продолжение текста с помощью GPT2 модели"""
    model.eval()
    
    # Токенизируем входной текст
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    with torch.no_grad():
        # Генерируем продолжение
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=input_ids.shape[1] + num_tokens,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.1,
            no_repeat_ngram_size=2
        )
    
    # Извлекаем сгенерированную часть
    generated_ids = outputs[0][input_ids.shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return generated_text

def calculate_rouge(prediction, reference):
    """Вычисляет ROUGE метрики между предсказанием и эталоном"""
    try:
        rouge = evaluate.load('rouge')
        results = rouge.compute(
            predictions=[prediction],
            references=[reference],
            use_stemmer=True
        )
        return {
            'rouge1': round(results['rouge1'], 4),
            'rouge2': round(results['rouge2'], 4),
            'rougeL': round(results['rougeL'], 4)
        }
    except Exception as e:
        return {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}

def get_actual_continuation(full_text, context_length=0.7):
    """Извлекает фактическое продолжение текста"""
    words = full_text.split()
    if len(words) <= 5:  # Слишком короткий текст
        return ""
    
    split_point = max(5, int(len(words) * context_length))
    context = " ".join(words[:split_point])
    continuation = " ".join(words[split_point:])
    
    return context, continuation

def create_comparison_table(texts_df, lstm_model, lstm_tokenizer, gpt2_model, gpt2_tokenizer, device, num_examples=15):
    """Создает таблицу сравнения моделей"""
    print("\n" + "="*60)
    print("📊 СОЗДАНИЕ ТАБЛИЦЫ СРАВНЕНИЯ МОДЕЛЕЙ")
    print("="*60)
    
    # Выбираем случайные примеры
    random.seed(42)
    sample_texts = random.sample(texts_df['text_raw'].tolist(), min(num_examples, len(texts_df)))
    
    results = []
    
    for i, full_text in enumerate(sample_texts):
        print(f"Обработка примера {i+1}/{len(sample_texts)}...")
        
        try:
            # Получаем контекст и фактическое продолжение
            context, actual_continuation = get_actual_continuation(full_text)
            
            if not context or not actual_continuation:
                continue
            
            # Генерируем продолжения моделями
            lstm_continuation = generate_lstm_completion(
                lstm_model, lstm_tokenizer, context, device, num_tokens=10
            )
            
            gpt2_continuation = generate_gpt2_completion(
                gpt2_model, gpt2_tokenizer, context, device, num_tokens=15
            )
            
            # Вычисляем ROUGE метрики
            lstm_rouge = calculate_rouge(lstm_continuation, actual_continuation)
            gpt2_rouge = calculate_rouge(gpt2_continuation, actual_continuation)
            
            results.append({
                'Исходное сообщение': full_text[:100] + "..." if len(full_text) > 100 else full_text,
                'Начало сообщения': context[:80] + "..." if len(context) > 80 else context,
                'Фактическое продолжение': actual_continuation[:80] + "..." if len(actual_continuation) > 80 else actual_continuation,
                'LSTM продолжение': lstm_continuation[:80] + "..." if len(lstm_continuation) > 80 else lstm_continuation,
                'DistilGPT2 продолжение': gpt2_continuation[:80] + "..." if len(gpt2_continuation) > 80 else gpt2_continuation,
                'ROUGE-L LSTM': lstm_rouge['rougeL'],
                'ROUGE-L DistilGPT2': gpt2_rouge['rougeL'],
                'ROUGE-1 LSTM': lstm_rouge['rouge1'],
                'ROUGE-1 DistilGPT2': gpt2_rouge['rouge1']
            })
            
        except Exception as e:
            print(f"❌ Ошибка при обработке примера {i+1}: {e}")
            continue
    
    # Создаем DataFrame
    df = pd.DataFrame(results)
    
    # Переупорядочиваем столбцы для лучшего отображения
    column_order = [
        'Исходное сообщение',
        'Начало сообщения', 
        'Фактическое продолжение',
        'LSTM продолжение',
        'DistilGPT2 продолжение',
        'ROUGE-L LSTM',
        'ROUGE-L DistilGPT2',
        'ROUGE-1 LSTM', 
        'ROUGE-1 DistilGPT2'
    ]
    
    # Оставляем только существующие столбцы
    existing_columns = [col for col in column_order if col in df.columns]
    df = df[existing_columns]
    
    return df

def main():
    # Конфигурация
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Используется устройство: {device}")
    
    # Пути к данным и модели
    data_path = Path(__file__).parent.parent / 'data' / '1_raw_dataset_tweets.txt'
    model_path = Path(__file__).parent.parent / 'models' / 'best_model.pth'
    
    # 1) Создаем тестовую выборку на 1000 записей
    test_loader, lstm_tokenizer, texts_df = create_test_dataset(
        file_path=data_path, 
        limit=1000,
        max_len=20
    )
    
    # 2) Загружаем LSTM модель
    lstm_model = None
    if model_path.exists():
        try:
            print("🔄 Загрузка LSTM модели...")
            lstm_model, model_config = load_lstm_model(model_path, device)
            lstm_model.eval()
            print("✅ LSTM модель загружена")
        except Exception as e:
            print(f"❌ Ошибка при загрузке LSTM модели: {e}")
            return
    else:
        print(f"❌ Файл модели LSTM не найден: {model_path}")
        return
    
    # 3) Загружаем DistilGPT2 модель
    try:
        print("🔄 Загрузка DistilGPT2 модели...")
        gpt2_tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        gpt2_model = GPT2LMHeadModel.from_pretrained("distilgpt2").to(device)
        
        # Настраиваем pad token для GPT2
        if gpt2_tokenizer.pad_token is None:
            gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
            
        gpt2_model.eval()
        print("✅ DistilGPT2 модель загружена")
    except Exception as e:
        print(f"❌ Ошибка при загрузке DistilGPT2 модели: {e}")
        return
    
    # 4) Создаем таблицу сравнения
    comparison_df = create_comparison_table(
        texts_df, 
        lstm_model, 
        lstm_tokenizer, 
        gpt2_model, 
        gpt2_tokenizer, 
        device,
        num_examples=150
    )
    
    # 5) Выводим таблицу
    print("\n" + "="*80)
    print("📋 ТАБЛИЦА СРАВНЕНИЯ МОДЕЛЕЙ (15 случайных примеров)")
    print("="*80)
    
    # Настраиваем отображение pandas
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 50)
    
    # Выводим таблицу
    print(comparison_df.to_string(index=False))
    
    # 6) Сохраняем таблицу в файл
    output_path = Path(__file__).parent / "model_comparison_results.csv"
    comparison_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\n💾 Таблица сохранена в: {output_path}")
    
    # 7) Выводим средние метрики
    print("\n" + "="*60)
    print("📈 СРЕДНИЕ МЕТРИКИ КАЧЕСТВА")
    print("="*60)
    
    if 'ROUGE-L LSTM' in comparison_df.columns and 'ROUGE-L DistilGPT2' in comparison_df.columns:
        avg_rouge_l_lstm = comparison_df['ROUGE-L LSTM'].mean()
        avg_rouge_l_gpt2 = comparison_df['ROUGE-L DistilGPT2'].mean()
        avg_rouge1_lstm = comparison_df['ROUGE-1 LSTM'].mean()
        avg_rouge1_gpt2 = comparison_df['ROUGE-1 DistilGPT2'].mean()
        
        print(f"📊 LSTM модель:")
        print(f"   Средний ROUGE-L: {avg_rouge_l_lstm:.4f}")
        print(f"   Средний ROUGE-1: {avg_rouge1_lstm:.4f}")
        
        print(f"\n📊 DistilGPT2 модель:")
        print(f"   Средний ROUGE-L: {avg_rouge_l_gpt2:.4f}")
        print(f"   Средний ROUGE-1: {avg_rouge1_gpt2:.4f}")
        
        # Определяем лучшую модель
        if avg_rouge_l_lstm > avg_rouge_l_gpt2:
            winner = "LSTM"
            diff = avg_rouge_l_lstm - avg_rouge_l_gpt2
        else:
            winner = "DistilGPT2"
            diff = avg_rouge_l_gpt2 - avg_rouge_l_lstm
        
        print(f"\n🏆 Лучшая модель по ROUGE-L: {winner}")
        print(f"   Разница: {diff:.4f}")
    
    print(f"\n✅ Сравнение завершено!")

if __name__ == "__main__":
    main()