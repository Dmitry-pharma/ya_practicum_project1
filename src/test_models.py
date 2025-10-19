import torch
from torch.utils.data import DataLoader
from pathlib import Path
from transformers import AutoTokenizer, GPT2LMHeadModel,BertTokenizerFast
import random
from sklearn.model_selection import train_test_split
import evaluate
from sklearn.metrics import accuracy_score
#src
from src.data_utils import load_and_clean_data, prepare_training_pairs,dataset_preparation
from src.next_token_dataset import TweetsDataset
from src.lstm_model import NextPhrasePredictionRNN
from src.eval_lstm import vevaluate, test_model, analyze_predictions, show_detailed_examples

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
        pad_idx=model_config['pad_idx'],
        num_layers=model_config['num_layers']
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

def prepare_transformer_dataset_old(texts_df, tokenizer, max_length=20, num_examples=100):
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

def prepare_transformer_dataset(test_ds, tokenizer, max_length=20, num_examples=100):
    """Подготавливает данные для трансформера в формате next-token prediction из TweetsDataset"""
    print("🔧 Подготовка данных для трансформера из test_ds...")
    
    data = []
    examples_processed = 0
    
    for i in range(min(len(test_ds), num_examples)):
        if examples_processed >= num_examples:
            break
            
        # Получаем пример из test_ds
        batch = test_ds[i]
        input_ids = batch['data']  # [max_length]
        targets = batch['target']  # [max_length]
        attention_mask = batch['mask']  # [max_length]
        
        # Получаем реальные токены (игнорируем паддинг)
        non_pad_indices = (attention_mask == 1).nonzero(as_tuple=True)[0]
        
        if len(non_pad_indices) < 2:  # Нужно как минимум 2 токена
            continue
        
        # Конвертируем в список токенов
        real_input_ids = input_ids[non_pad_indices].tolist()
        real_targets = targets[non_pad_indices].tolist()
        
        # Создаем примеры next-token prediction
        for j in range(len(real_input_ids) - 1):
            if examples_processed >= num_examples:
                break
                
            context = real_input_ids[:j+1]  # Токены контекста
            target = real_targets[j]        # Следующий токен
            
            # Паддинг слева до max_length
            if len(context) < max_length:
                padding = [tokenizer.pad_token_id] * (max_length - len(context))
                padded_context = padding + context
            else:
                padded_context = context[-max_length:]
            
            # Создаем attention_mask
            padded_attention_mask = [1 if token_id != tokenizer.pad_token_id else 0 
                                   for token_id in padded_context]
            
            data.append({
                'input_ids': torch.tensor(padded_context, dtype=torch.long),
                'attention_mask': torch.tensor(padded_attention_mask, dtype=torch.long),
                'labels': torch.tensor([target], dtype=torch.long)  # Для совместимости с transformers
            })
            
            examples_processed += 1
    
    print(f"✅ Подготовлено {len(data)} примеров для трансформера из test_ds")
    return data

def prepare_transformer_dataset_from_loader(test_loader, tokenizer, max_length=20, num_examples=100):
    """Подготавливает данные для трансформера из DataLoader"""
    print("🔧 Подготовка данных для трансформера из test_loader...")
    
    data = []
    examples_processed = 0
    
    for batch in test_loader:
        if examples_processed >= num_examples:
            break
            
        input_ids_batch = batch['data']      # [batch_size, max_length]
        targets_batch = batch['target']      # [batch_size, max_length]
        attention_mask_batch = batch['mask'] # [batch_size, max_length]
        
        # Обрабатываем каждый пример в батче
        for i in range(input_ids_batch.size(0)):
            if examples_processed >= num_examples:
                break
                
            input_ids = input_ids_batch[i]    # [max_length]
            targets = targets_batch[i]        # [max_length]
            attention_mask = attention_mask_batch[i]  # [max_length]
            
            # Получаем реальные токены (игнорируем паддинг)
            non_pad_indices = (attention_mask == 1).nonzero(as_tuple=True)[0]
            
            if len(non_pad_indices) < 2:
                continue
            
            real_input_ids = input_ids[non_pad_indices].tolist()
            real_targets = targets[non_pad_indices].tolist()
            
            # Создаем примеры next-token prediction
            for j in range(len(real_input_ids) - 1):
                if examples_processed >= num_examples:
                    break
                    
                context = real_input_ids[:j+1]
                target = real_targets[j]
                
                # Паддинг слева
                if len(context) < max_length:
                    padding = [tokenizer.pad_token_id] * (max_length - len(context))
                    padded_context = padding + context
                else:
                    padded_context = context[-max_length:]
                
                padded_attention_mask = [1 if token_id != tokenizer.pad_token_id else 0 
                                       for token_id in padded_context]
                
                data.append({
                    'input_ids': torch.tensor(padded_context, dtype=torch.long),
                    'attention_mask': torch.tensor(padded_attention_mask, dtype=torch.long),
                    'labels': torch.tensor([target], dtype=torch.long)
                })
                
                examples_processed += 1
    
    print(f"✅ Подготовлено {len(data)} примеров для трансформера")
    return data



def test_transformer_model_old(model_name, texts_df, device, max_length=20, num_examples=100):
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


def test_transformer_with_generation_old(model_name, test_loader, device, max_length=20, num_examples=50):
    """Тестирует трансформер модель в режиме генерации текста"""
    print("\n" + "="*60)
    print("🤖 ТЕСТИРОВАНИЕ TRANSFORMER МОДЕЛИ (РЕЖИМ ГЕНЕРАЦИИ)")
    print("="*60)
    
    # Загружаем токенизатор и модель
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    
    # Подготавливаем данные
    transformer_data = prepare_transformer_dataset_from_loader(
        test_loader, tokenizer, max_length=max_length, num_examples=num_examples
    )
    
    if not transformer_data:
        print("❌ Не удалось подготовить данные для трансформера")
        return None
    
    all_predictions = []
    all_references = []
    all_contexts = []
    
    print(f"🧪 Генерация текста на {len(transformer_data)} примерах...")
    
    with torch.no_grad():
        for i, example in enumerate(transformer_data):
            input_ids = example['input_ids'].unsqueeze(0).to(device)
            attention_mask = example['attention_mask'].unsqueeze(0).to(device)
            target = example['labels'].to(device)
            
            try:
                # Генерируем продолжение
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=input_ids.shape[1] + 1,  # +1 токен
                    num_return_sequences=1,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                
                # Декодируем результаты
                context_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
                generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
                continuation = generated_text[len(context_text):].strip()
                
                true_token_text = tokenizer.decode(target.cpu(), skip_special_tokens=True)
                
                if true_token_text.strip() and continuation.strip():
                    all_contexts.append(context_text)
                    all_references.append(true_token_text)
                    all_predictions.append(continuation)
                    
                    # Показываем первый пример
                    if i == 0:
                        print(f"\n🔍 Пример генерации:")
                        print(f"   Контекст: '{context_text}'")
                        print(f"   Ожидалось: '{true_token_text}'")
                        print(f"   Сгенерировано: '{continuation}'")
                        
            except Exception as e:
                continue
            
            if (i + 1) % 10 == 0:
                print(f"   Обработано {i + 1}/{len(transformer_data)} примеров")
    
    # Вычисляем ROUGE метрики
    rouge_metrics = None
    if all_predictions and all_references:
        try:
            
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
                'num_examples': len(all_references)
            }
            
            print(f"\n📊 Результаты генерации ({model_name}):")
            print(f"   ROUGE-1: {rouge_metrics['rouge1']:.4f}")
            print(f"   ROUGE-2: {rouge_metrics['rouge2']:.4f}")
            print(f"   ROUGE-L: {rouge_metrics['rougeL']:.4f}")
            
        except Exception as e:
            print(f"❌ Ошибка при вычислении ROUGE: {e}")
    
    # Показываем примеры генерации
    print(f"\n🔍 Примеры генерации:")
    examples_to_show = min(3, len(all_predictions))
    for i in range(examples_to_show):
        print(f"   Пример {i+1}:")
        print(f"      Контекст: '{all_contexts[i]}'")
        print(f"      Ожидалось: '{all_references[i]}'")
        print(f"      Сгенерировано: '{all_predictions[i]}'")
        print()
    
    return {
        'rouge_metrics': rouge_metrics,
        'model_type': 'Transformer (Generation)',
        'model_name': model_name,
        'num_examples': len(all_predictions)
    }

def test_transformer_with_generation(model_name, test_loader, device, max_length=20, num_examples=50):
    """Тестирует трансформер модель в режиме генерации текста"""
    print("\n" + "="*60)
    print("🤖 ТЕСТИРОВАНИЕ TRANSFORMER МОДЕЛИ (РЕЖИМ ГЕНЕРАЦИИ)")
    print("="*60)
    
    # Загружаем токенизатор и модель
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    
    # Подготавливаем данные
    transformer_data = prepare_transformer_dataset_from_loader(
        test_loader, tokenizer, max_length=max_length, num_examples=num_examples
    )
    
    if not transformer_data:
        print("❌ Не удалось подготовить данные для трансформера")
        return None
    
    all_predictions = []
    all_references = []
    all_contexts = []
    
    print(f"🧪 Генерация текста на {len(transformer_data)} примерах...")
    
    with torch.no_grad():
        for i, example in enumerate(transformer_data):
            input_ids = example['input_ids'].unsqueeze(0).to(device)
            attention_mask = example['attention_mask'].unsqueeze(0).to(device)
            target = example['labels'].to(device)
            
            try:
                # Генерируем продолжение
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=input_ids.shape[1] + 1,  # +1 токен
                    num_return_sequences=1,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                
                # Декодируем результаты
                context_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
                generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
                continuation = generated_text[len(context_text):].strip()
                
                true_token_text = tokenizer.decode(target.cpu(), skip_special_tokens=True)
                
                if true_token_text.strip() and continuation.strip():
                    all_contexts.append(context_text)
                    all_references.append(true_token_text)
                    all_predictions.append(continuation)
                    
                    # Показываем первый пример
                    if i >= 0: #'{context_text}
                        print(f"\n🔍 Пример '{i} генерации:")
                        print(f"   Контекст: '{context_text}'")
                        print(f"   Ожидалось: '{true_token_text}'")
                        print(f"   Сгенерировано: '{continuation}'")
                        
            except Exception as e:
                continue
            
            if (i + 1) % 10 == 0:
                print(f"   Обработано {i + 1}/{len(transformer_data)} примеров")
    
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
                'num_examples': len(all_references)
            }
            
            print(f"\n📊 Результаты генерации ({model_name}):")
            print(f"   ROUGE-1: {rouge_metrics['rouge1']:.4f}")
            print(f"   ROUGE-2: {rouge_metrics['rouge2']:.4f}")
            print(f"   ROUGE-L: {rouge_metrics['rougeL']:.4f}")
            
        except Exception as e:
            print(f"❌ Ошибка при вычислении ROUGE: {e}")
    
    # Показываем примеры генерации
    print(f"\n🔍 Примеры генерации:")
    examples_to_show = min(3, len(all_predictions))
    for i in range(examples_to_show):
        print(f"   Пример {i+1}:")
        print(f"      Контекст: '{all_contexts[i]}'")
        print(f"      Ожидалось: '{all_references[i]}'")
        print(f"      Сгенерировано: '{all_predictions[i]}'")
        print()
    
    # Вычисляем accuracy для генерации (сравниваем точное совпадение токенов)
    accuracy = 0
    if all_predictions and all_references:
        correct = sum(1 for pred, ref in zip(all_predictions, all_references) if pred == ref)
        accuracy = correct / len(all_references)
        print(f"📊 Accuracy генерации (точное совпадение): {accuracy:.4f}")
    
    return {
        'accuracy': accuracy,
        'rouge_metrics': rouge_metrics,
        'model_type': 'Transformer',
        'model_name': model_name
    }

def test_transformer_model_new__(model_name, texts_df, device, max_length=20, num_examples=100):
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
    # transformer_data = prepare_transformer_dataset(texts_df, tokenizer, max_length, num_examples)
    training_pairs= prepare_training_pairs(texts_df, tokenizer, max_length)
    
    if not training_pairs:
        print("❌ Не удалось подготовить данные для трансформера")
        return None
    
    # Ограничиваем количество примеров
    if len(training_pairs) > num_examples:
        training_pairs = training_pairs[:num_examples]
        print(f"📝 Используем {num_examples} примеров из {len(training_pairs)}")
    
    
    total_correct = 0
    total_tokens = 0
    all_predictions = []
    all_references = []
    
    with torch.no_grad():
        for i, (x_padded, y_padded, attention_mask) in enumerate(training_pairs):
            # Преобразуем в тензоры
            input_ids = torch.tensor(x_padded).unsqueeze(0).to(device)
            attention_mask_tensor = torch.tensor(attention_mask).unsqueeze(0).to(device)
            targets = torch.tensor(y_padded).to(device)
            # input_ids = example['input_ids'].unsqueeze(0).to(device)  # Add batch dimension
            # attention_mask = example['attention_mask'].unsqueeze(0).to(device)
            # target = example['labels'].to(device)
            
            # Получаем предсказания
            outputs = model(input_ids, attention_mask=attention_mask_tensor)
            logits = outputs.logits
            
            # Берем предсказание для последнего токена
            # next_token_logits = logits[:, -1, :]
            predicted_tokens = torch.argmax(logits, dim=-1).squeeze(0)
            
            # Сравниваем каждый токен (игнорируем паддинг)
            for j in range(len(targets)):
                if attention_mask[j] == 1:  # Только реальные токены
                    if predicted_tokens[j].item() == targets[j].item():
                        total_correct += 1
                    total_tokens += 1
                    
                    # Сохраняем для анализа (только не-pad токены)
                    try:
                        true_token_text = tokenizer.decode([targets[j].item()], skip_special_tokens=True)
                        pred_token_text = tokenizer.decode([predicted_tokens[j].item()], skip_special_tokens=True)
                        
                        if true_token_text.strip() and pred_token_text.strip():
                            all_references.append(true_token_text)
                            all_predictions.append(pred_token_text)
                    except Exception as e:
                        continue
            
            if (i + 1) % 50 == 0:
                print(f"Обработано {i + 1}/{len(training_pairs)} примеров")
            
            # Сохраняем для анализа
            try:
                true_token_text = tokenizer.decode(targets.cpu(), skip_special_tokens=True)
                pred_token_text = tokenizer.decode(predicted_tokens.cpu(), skip_special_tokens=True)
                
                if true_token_text.strip() and pred_token_text.strip():
                    all_references.append(true_token_text)
                    all_predictions.append(pred_token_text)
            except Exception as e:
                continue
            
            if (i + 1) % 50 == 0:
                print(f"Обработано {i + 1}/{len(training_pairs)} примеров")
    
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
    examples_to_show = min(100, len(all_predictions))
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

def test_transformer_model_new(model_name, texts_df, device, max_length=20, num_examples=100):
    """Упрощенная версия тестирования трансформера"""
    print("\n" + "="*60)
    print("🤖 ТЕСТИРОВАНИЕ TRANSFORMER МОДЕЛИ")
    print("="*60)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    
    training_pairs = prepare_training_pairs(texts_df, tokenizer, max_length)
    
    if not training_pairs:
        print("❌ Не удалось подготовить данные")
        return None
    
    training_pairs = training_pairs[:num_examples]
    
    total_correct = 0
    total_tokens = 0
    
    print(f"🧪 Тестирование на {len(training_pairs)} примерах...")
    
    with torch.no_grad():
        for i, (x_padded, y_padded, attention_mask) in enumerate(training_pairs):
            input_ids = torch.tensor(x_padded).unsqueeze(0).to(device)
            attention_mask_tensor = torch.tensor(attention_mask).unsqueeze(0).to(device)
            
            # Получаем реальные токены
            non_pad_indices = (attention_mask_tensor[0] == 1).nonzero(as_tuple=True)[0]
            
            if len(non_pad_indices) < 2:
                continue
            
            targets = torch.tensor(y_padded).to(device)
            
            try:
                # Генерация
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask_tensor,
                    max_length=input_ids.shape[1] + len(non_pad_indices),
                    num_return_sequences=1,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                
                # Сравнение
                generated_sequence = generated[0, input_ids.shape[1]:]
                
                for j in range(min(len(generated_sequence), len(non_pad_indices))):
                    generated_token = generated_sequence[j]
                    target_token = targets[j]
                    
                    if generated_token.item() == target_token.item():
                        total_correct += 1
                    total_tokens += 1
                    
            except Exception as e:
                print(f"❌ Ошибка для примера {i}: {e}")
                continue
            
            if (i + 1) % 10 == 0:
                print(f"Обработано {i + 1}/{len(training_pairs)} примеров")
    
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    
    print(f"\n📊 Результаты:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Токенов: {total_tokens}")
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
    
    return {
        'accuracy': accuracy,
        'rouge_metrics': rouge_metrics,
        'model_type': 'Transformer',
        'model_name': model_name
    }

def evaluate_transformer_forward(model_name, data, device, compute_rouge=False, num_rouge_examples=50):
    """Оценивает трансформер модель через прямой forward pass"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    model.eval()
    
    
    _, data_loader = dataset_preparation(data, tokenizer, MAX_LEN=20, batch_size=128)
    
    preds, trues = [], []
    total_loss = 0
    
    # Для ROUGE метрик
    references, predictions = [], []
    
    print(f"🧪 Оценка трансформера (forward) на {len(data_loader)} батчах...")
    
    with torch.no_grad():
        examples_processed = 0
        for batch_idx, batch in enumerate(data_loader):
            ids = batch['data'].to(device)
            mask = batch['mask'].to(device)
            labels = batch['target'].to(device)
            
            # Прямой forward pass
            outputs = model(
                input_ids=ids,
                attention_mask=mask,
                labels=labels  # Для автоматического вычисления loss
            )
            
            logits = outputs.logits
            loss = outputs.loss
            
            total_loss += loss.item()
            
            # Предсказываем токены
            batch_preds = torch.argmax(logits, dim=-1)
            
            # Сохраняем предсказания и истинные значения
            preds.extend(batch_preds.cpu().flatten().tolist())
            trues.extend(labels.cpu().flatten().tolist())
            
            # Собираем данные для ROUGE
            if compute_rouge and examples_processed < num_rouge_examples:
                for i in range(ids.size(0)):
                    if examples_processed >= num_rouge_examples:
                        break
                    
                    # Получаем реальные токены (игнорируем паддинг)
                    non_pad_indices = mask[i].bool()
                    if non_pad_indices.sum() == 0:
                        continue
                    
                    input_ids_real = ids[i][non_pad_indices]
                    preds_real = batch_preds[i][non_pad_indices]
                    labels_real = labels[i][non_pad_indices]
                    
                    # Конвертируем в текст
                    try:
                        # Входной текст (контекст)
                        input_tokens = tokenizer.convert_ids_to_tokens(input_ids_real.cpu())
                        input_text = tokenizer.convert_tokens_to_string(input_tokens)
                        
                        # Целевой текст (ожидаемое продолжение)
                        true_tokens = tokenizer.convert_ids_to_tokens(labels_real.cpu())
                        true_text = tokenizer.convert_tokens_to_string(true_tokens)
                        
                        # Предсказанный текст
                        pred_tokens = tokenizer.convert_ids_to_tokens(preds_real.cpu())
                        pred_text = tokenizer.convert_tokens_to_string(pred_tokens)
                        
                        if true_text.strip() and pred_text.strip():
                            references.append(true_text)
                            predictions.append(pred_text)
                            examples_processed += 1
                            
                    except Exception as e:
                        print(f"⚠️ Ошибка при обработке примера для ROUGE: {e}")
                        continue
            
            if (batch_idx + 1) % 10 == 0:
                print(f"   Обработано батчей: {batch_idx + 1}/{len(data_loader)}")
    
    # Вычисляем метрики
    accuracy = accuracy_score(trues, preds) if trues and preds else 0
    avg_loss = total_loss / len(data_loader)
    
    print(f"📊 Метрики трансформера (forward):")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Loss: {avg_loss:.4f}")
    print(f"   Примеров: {len(preds)}")
    
    # Вычисляем ROUGE метрики
    rouge_metrics = None
    if compute_rouge and references and predictions:
        try:
            import evaluate
            rouge = evaluate.load('rouge')
            rouge_results = rouge.compute(
                predictions=predictions,
                references=references,
                use_stemmer=True,
                use_aggregator=True
            )
            
            rouge_metrics = {
                'rouge1': round(rouge_results['rouge1'], 4),
                'rouge2': round(rouge_results['rouge2'], 4),
                'rougeL': round(rouge_results['rougeL'], 4),
                'rougeLsum': round(rouge_results['rougeLsum'], 4),
                'num_examples': len(references)
            }
            
        except Exception as e:
            print(f"❌ Ошибка при вычислении ROUGE: {e}")
            rouge_metrics = {'error': str(e)}
    
    return accuracy, avg_loss, rouge_metrics


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
        # results_transformer = test_transformer_model(
        #     model_name=transformer_model_name,
        #     texts_df=texts_df,
        #     device=device,
        #     max_length=20,
        #     num_examples=200  # Ограничиваем для скорости
        # )
        results_transformer = test_transformer_with_generation(
            model_name=transformer_model_name, 
            test_loader=test_loader, 
            device=device, 
            max_length=20, 
            num_examples=200
        )
    except Exception as e:
        print(f"❌ Ошибка при тестировании Transformer модели: {e}")
        results_transformer = None
    
    # Сравниваем результаты
    compare_models(results_lstm, results_transformer)
    
    print(f"\n✅ Тестирование завершено!")

if __name__ == "__main__":
    main()