import torch
from torch.utils.data import DataLoader
from pathlib import Path
from transformers import AutoTokenizer, GPT2LMHeadModel
import random
from data_utils import load_and_clean_data, prepare_training_pairs
from next_token_dataset import TweetsDataset
from lstm_model import NextPhrasePredictionRNN
from eval_lstm import vevaluate, test_model, analyze_predictions, show_detailed_examples
from sklearn.model_selection import train_test_split

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

def test_lstm_model(model_path, test_loader, tokenizer, device):
    """Тестирует LSTM модель на тестовой выборке"""
    print("\n" + "="*60)
    print("🧠 ТЕСТИРОВАНИЕ LSTM МОДЕЛИ")
    print("="*60)
    
    # Загружаем конфигурацию модели
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint['model_config']
    
    # Создаем модель
    model = NextPhrasePredictionRNN(
        rnn_type="LSTM",
        vocab_size=model_config['vocab_size'],
        emb_dim=model_config['emb_dim'],
        hidden_dim=model_config['hidden_dim'],
        pad_idx=model_config['pad_idx']
    ).to(device)
    
    # Загружаем веса
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Критерий для оценки
    criterion = torch.nn.CrossEntropyLoss(ignore_index=model_config['pad_idx'])
    
    # Тестируем модель
    test_accuracy, test_loss = test_model(model, test_loader, criterion, device)
    
    # Детальная оценка с ROUGE метриками
    accuracy, avg_loss, rouge_metrics = vevaluate(
        model, test_loader, criterion, device, tokenizer, 
        compute_rouge=True, num_rouge_examples=100
    )
    
    print(f"\n📊 Детальные результаты LSTM:")
    print(f"   Test Accuracy: {accuracy:.4f}")
    print(f"   Test Loss: {avg_loss:.4f}")
    if rouge_metrics and 'rouge1' in rouge_metrics:
        print(f"   ROUGE-1: {rouge_metrics['rouge1']:.4f}")
        print(f"   ROUGE-2: {rouge_metrics['rouge2']:.4f}")
        print(f"   ROUGE-L: {rouge_metrics['rougeL']:.4f}")
    
    # Анализ предсказаний
    bad_cases, good_cases = analyze_predictions(model, test_loader, tokenizer, device)
    
    # Детальные примеры
    show_detailed_examples(model, test_loader, tokenizer, num_examples=3)
    
    return {
        'accuracy': accuracy,
        'loss': avg_loss,
        'rouge_metrics': rouge_metrics,
        'model_type': 'LSTM'
    }

def prepare_transformer_dataset(texts_df, tokenizer, max_length=20, num_examples=100):
    """Подготавливает данные для трансформера в формате next-token prediction"""
    print("🔧 Подготовка данных для трансформера...")
    
    data = []
    for text in texts_df['text_cleaned'][:num_examples]:
        if len(text.strip()) < 5:  # Пропускаем слишком короткие тексты
            continue
            
        # Токенизируем текст
        inputs = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
        input_ids = inputs['input_ids'][0]
        
        if len(input_ids) < 2:  # Нужен как минимум 1 токен для контекста и 1 для предсказания
            continue
        
        # Создаем примеры next-token prediction
        for i in range(len(input_ids) - 1):
            context = input_ids[:i+1]
            target = input_ids[i+1]
            
            # Паддинг до max_length
            if len(context) < max_length:
                padding = torch.full((max_length - len(context),), tokenizer.pad_token_id)
                context = torch.cat([padding, context])
            else:
                context = context[-max_length:]
            
            attention_mask = (context != tokenizer.pad_token_id).long()
            
            data.append({
                'input_ids': context,
                'attention_mask': attention_mask,
                'labels': target.unsqueeze(0)  # Добавляем dimension для совместимости
            })
    
    print(f"✅ Подготовлено {len(data)} примеров для трансформера")
    return data

def test_transformer_model(model_name, texts_df, device, max_length=20, num_examples=100):
    """Тестирует трансформер модель (DistilGPT2) на тестовой выборке"""
    print("\n" + "="*60)
    print("🤖 ТЕСТИРОВАНИЕ TRANSFORMER МОДЕЛИ (DistilGPT2)")
    print("="*60)
    
    # Загружаем токенизатор и модель
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    
    # Добавляем pad token если его нет
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    
    # Подготавливаем данные
    transformer_data = prepare_transformer_dataset(texts_df, tokenizer, max_length, num_examples)
    
    if not transformer_data:
        print("❌ Не удалось подготовить данные для трансформера")
        return None
    
    total_correct = 0
    total_tokens = 0
    all_predictions = []
    all_references = []
    
    with torch.no_grad():
        for i, example in enumerate(transformer_data):
            input_ids = example['input_ids'].unsqueeze(0).to(device)  # Add batch dimension
            attention_mask = example['attention_mask'].unsqueeze(0).to(device)
            target = example['labels'].to(device)
            
            # Получаем предсказания
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Берем предсказание для последнего токена
            next_token_logits = logits[:, -1, :]
            predicted_token = torch.argmax(next_token_logits, dim=-1)
            
            # Сравниваем с целевым токеном
            if predicted_token.item() == target.item():
                total_correct += 1
            total_tokens += 1
            
            # Сохраняем для анализа
            try:
                true_token_text = tokenizer.decode(target.cpu(), skip_special_tokens=True)
                pred_token_text = tokenizer.decode(predicted_token.cpu(), skip_special_tokens=True)
                
                if true_token_text.strip() and pred_token_text.strip():
                    all_references.append(true_token_text)
                    all_predictions.append(pred_token_text)
            except Exception as e:
                continue
            
            if (i + 1) % 50 == 0:
                print(f"Обработано {i + 1}/{len(transformer_data)} примеров")
    
    # Вычисляем accuracy
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    
    print(f"\n📊 Результаты Transformer ({model_name}):")
    print(f"   Test Accuracy: {accuracy:.4f}")
    print(f"   Обработано токенов: {total_tokens}")
    print(f"   Правильных предсказаний: {total_correct}")
    
    # Вычисляем ROUGE метрики
    rouge_metrics = None
    if all_predictions and all_references:
        try:
            import evaluate
            rouge = evaluate.load('rouge')
            rouge_results = rouge.compute(
                predictions=all_predictions,
                references=all_references,
                use_stemmer=True,
                use_aggregator=True
            )
            
            rouge_metrics = {
                'rouge1': round(rouge_results['rouge1'], 4),
                'rouge2': round(rouge_results['rouge2'], 4),
                'rougeL': round(rouge_results['rougeL'], 4),
                'rougeLsum': round(rouge_results['rougeLsum'], 4),
                'num_examples': len(all_references)
            }
            
            print(f"   ROUGE-1: {rouge_metrics['rouge1']:.4f}")
            print(f"   ROUGE-2: {rouge_metrics['rouge2']:.4f}")
            print(f"   ROUGE-L: {rouge_metrics['rougeL']:.4f}")
        except Exception as e:
            print(f"❌ Ошибка при вычислении ROUGE: {e}")
            rouge_metrics = None
    else:
        print("❌ Не удалось собрать достаточно данных для ROUGE метрик")
    
    # Показываем несколько примеров предсказаний
    print(f"\n🔍 Примеры предсказаний Transformer:")
    examples_to_show = min(5, len(all_predictions))
    for i in range(examples_to_show):
        print(f"   Пример {i+1}:")
        print(f"      Ожидалось: '{all_references[i]}'")
        print(f"      Предсказано: '{all_predictions[i]}'")
        status = "✅" if all_references[i] == all_predictions[i] else "❌"
        print(f"      Статус: {status}")
        print()
    
    return {
        'accuracy': accuracy,
        'rouge_metrics': rouge_metrics,
        'model_type': 'Transformer',
        'model_name': model_name
    }

def compare_models(results_lstm, results_transformer):
    """Сравнивает результаты двух моделей"""
    print("\n" + "="*60)
    print("📈 СРАВНЕНИЕ МОДЕЛЕЙ")
    print("="*60)
    
    if results_lstm:
        print(f"\n🧠 {results_lstm['model_type']}:")
        print(f"   Accuracy: {results_lstm['accuracy']:.4f}")
        if results_lstm.get('rouge_metrics'):
            print(f"   ROUGE-1: {results_lstm['rouge_metrics']['rouge1']:.4f}")
    else:
        print(f"\n🧠 LSTM: Результаты недоступны")
    
    if results_transformer:
        print(f"\n🤖 {results_transformer['model_type']} ({results_transformer['model_name']}):")
        print(f"   Accuracy: {results_transformer['accuracy']:.4f}")
        if results_transformer.get('rouge_metrics'):
            print(f"   ROUGE-1: {results_transformer['rouge_metrics']['rouge1']:.4f}")
    else:
        print(f"\n🤖 Transformer: Результаты недоступны")
    
    # Определяем лучшую модель по accuracy
    if results_lstm and results_transformer:
        if results_lstm['accuracy'] > results_transformer['accuracy']:
            winner = "LSTM"
            diff = results_lstm['accuracy'] - results_transformer['accuracy']
        else:
            winner = "Transformer"
            diff = results_transformer['accuracy'] - results_lstm['accuracy']
        
        print(f"\n🏆 Лучшая модель по accuracy: {winner}")
        print(f"   Разница: {diff:.4f}")
    else:
        print(f"\n🏆 Невозможно определить лучшую модель - недостаточно данных")

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
    
    # 2) Тестируем LSTM модель
    results_lstm = None
    if model_path.exists():
        try:
            results_lstm = test_lstm_model(model_path, test_loader, lstm_tokenizer, device)
        except Exception as e:
            print(f"❌ Ошибка при тестировании LSTM модели: {e}")
    else:
        print(f"❌ Файл модели LSTM не найден: {model_path}")
    
    # 3) Тестируем Transformer модель (DistilGPT2)
    transformer_model_name = "distilgpt2"
    try:
        results_transformer = test_transformer_model(
            model_name=transformer_model_name,
            texts_df=texts_df,
            device=device,
            max_length=20,
            num_examples=200  # Ограничиваем для скорости
        )
    except Exception as e:
        print(f"❌ Ошибка при тестировании Transformer модели: {e}")
        results_transformer = None
    
    # Сравниваем результаты
    compare_models(results_lstm, results_transformer)
    
    print(f"\n✅ Тестирование завершено!")

if __name__ == "__main__":
    main()