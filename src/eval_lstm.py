import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from collections import Counter
import random
import evaluate  # pip install evaluate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def vevaluate(model, loader, criterion, device, tokenizer, compute_rouge=False, num_rouge_examples=50):
    model.eval()
    preds, trues = [], []
    total_loss = 0
    # Для ROUGE метрик
    references,predictions = [],[]
    
    with torch.no_grad():
        examples_processed = 0
        for batch in loader:
            ids = batch['data'].to(device)
            mask = batch['mask'].to(device)
            labels = batch['target'].to(device)
            
            logits = model(ids, mask)
            logits_flat = logits.reshape(-1, logits.size(-1))
            labels_flat = labels.reshape(-1)
            
            loss = criterion(logits_flat, labels_flat)
            total_loss += loss.item()
            
            batch_preds = torch.argmax(logits, dim=-1)
            
            # preds += torch.argmax(logits, dim=-1).cpu().flatten().tolist()
            preds += batch_preds.cpu().flatten().tolist()
            trues += labels.cpu().flatten().tolist()
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
                        
                        # Для задачи предсказания следующего токена сравниваем продолжения
                        # Reference - ожидаемое продолжение, Prediction - сгенерированное продолжение
                        if true_text.strip() and pred_text.strip():
                            references.append(true_text)
                            predictions.append(pred_text)
                            examples_processed += 1
                            
                    except Exception as e:
                        print(f"⚠️ Ошибка при обработке примера для ROUGE: {e}")
                        continue
    
    accuracy = accuracy_score(trues, preds)
    avg_loss = total_loss / len(loader)
    # Вычисляем ROUGE метрики если требуется
    rouge_metrics = None
    if compute_rouge and references and predictions:
        try:
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
            
            # print(f"📊 ROUGE метрики ({len(references)} примеров):")
            # print(f"   ROUGE-1: {rouge_metrics['rouge1']:.4f}")
            # print(f"   ROUGE-2: {rouge_metrics['rouge2']:.4f}")
            # print(f"   ROUGE-L: {rouge_metrics['rougeL']:.4f}")
            
        except Exception as e:
            print(f"❌ Ошибка при вычислении ROUGE: {e}")
            rouge_metrics = {'error': str(e)}
    return accuracy, avg_loss, rouge_metrics

def vevaluate2(model, loader, criterion, device, tokenizer, compute_rouge=False, num_rouge_examples=50):
    """
    Валидация модели с вычислением accuracy, loss и ROUGE метрик
    
    Args:
        model: модель для оценки
        loader: DataLoader с валидационными данными
        criterion: функция потерь
        device: устройство (cpu/cuda)
        tokenizer: токенизатор
        compute_rouge: вычислять ROUGE метрики
        num_rouge_examples: количество примеров для ROUGE
    
    Returns:
        accuracy, avg_loss, rouge_metrics
    """
    model.eval()
    preds, trues = [], []
    total_loss = 0
    references, predictions = [], []
    
    with torch.no_grad():
        examples_processed = 0
        
        for batch_idx, batch in enumerate(loader):
            ids = batch['data'].to(device)
            mask = batch['mask'].to(device)
            labels = batch['target'].to(device)
            
            # Прямой проход
            logits = model(ids, mask)
            logits_flat = logits.reshape(-1, logits.size(-1))
            labels_flat = labels.reshape(-1)
            
            # Вычисление потерь
            loss = criterion(logits_flat, labels_flat)
            total_loss += loss.item()
            
            # Предсказания
            batch_preds = torch.argmax(logits, dim=-1)
            
            # Сбор предсказаний и истинных значений (ИСКЛЮЧАЯ PAD ТОКЕНЫ)
            non_pad_mask = (labels != tokenizer.pad_token_id)
            preds_non_pad = batch_preds[non_pad_mask].cpu().flatten().tolist()
            trues_non_pad = labels[non_pad_mask].cpu().flatten().tolist()
            
            preds.extend(preds_non_pad)
            trues.extend(trues_non_pad)
            
            # Сбор данных для ROUGE
            if compute_rouge and examples_processed < num_rouge_examples:
                batch_size = ids.size(0)
                
                for i in range(batch_size):
                    if examples_processed >= num_rouge_examples:
                        break
                    
                    try:
                        # Находим границу между контекстом и продолжением
                        non_pad_indices = mask[i].bool()
                        if non_pad_indices.sum() == 0:
                            continue
                            
                        # Последний токен контекста (предпоследний в полной последовательности)
                        context_end_idx = non_pad_indices.sum().item() - 1
                        if context_end_idx <= 0:
                            continue
                        
                        # Контекст (входные данные)
                        context_tokens = ids[i][:context_end_idx]
                        context_text = tokenizer.decode(context_tokens.cpu(), skip_special_tokens=True)
                        
                        # Ожидаемое продолжение (последний токен)
                        expected_next_token = labels[i][context_end_idx]
                        expected_text = tokenizer.decode([expected_next_token.item()], skip_special_tokens=True)
                        
                        # Предсказанное продолжение
                        predicted_next_token = batch_preds[i][context_end_idx]
                        predicted_text = tokenizer.decode([predicted_next_token.item()], skip_special_tokens=True)
                        
                        # Проверяем что тексты не пустые
                        if expected_text.strip() and predicted_text.strip():
                            references.append(expected_text)
                            predictions.append(predicted_text)
                            examples_processed += 1
                            
                            # Логируем первый пример для отладки
                            # if examples_processed == 1:
                            #     print(f"🔍 Пример ROUGE:")
                            #     print(f"   Контекст: '{context_text}'")
                            #     print(f"   Ожидалось: '{expected_text}'")
                            #     print(f"   Предсказано: '{predicted_text}'")
                            
                    except Exception as e:
                        print(f"⚠️ Ошибка при обработке примера {examples_processed} для ROUGE: {e}")
                        continue
    
    # Вычисление accuracy (только для не-pad токенов)
    accuracy = accuracy_score(trues, preds) if trues and preds else 0.0
    avg_loss = total_loss / len(loader)
    
    # ROUGE метрики
    rouge_metrics = None
    if compute_rouge and references and predictions:
        try:
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
            
            # print(f"📊 ROUGE метрики ({len(references)} примеров):")
            # print(f"   ROUGE-1: {rouge_metrics['rouge1']:.4f}")
            # print(f"   ROUGE-2: {rouge_metrics['rouge2']:.4f}")
            # print(f"   ROUGE-L: {rouge_metrics['rougeL']:.4f}")
            
        except Exception as e:
            print(f"❌ Ошибка при вычислении ROUGE: {e}")
            rouge_metrics = {'error': str(e)}
    elif compute_rouge:
        print("⚠️ Не удалось вычислить ROUGE: нет валидных примеров")
        rouge_metrics = {'error': 'No valid examples'}
    
    # Статистика
    print(f"📊 Валидация: Accuracy={accuracy:.4f}, Loss={avg_loss:.4f}, "
          f"Примеров={len(preds)}, ROUGE примеров={len(references) if references else 0}")
    
    return accuracy, avg_loss, rouge_metrics

def vevaluate3(model, loader, criterion, device, tokenizer, compute_rouge=False, num_rouge_examples=50):
    model.eval()
    preds, trues = [], []
    total_loss = 0
    total_tokens = 0  # ← ДОБАВЛЕНО: счетчик реальных токенов
    # Для ROUGE метрик
    references,predictions = [],[]
    
    with torch.no_grad():
        examples_processed = 0
        for batch in loader:
            ids = batch['data'].to(device)
            mask = batch['mask'].to(device)
            labels = batch['target'].to(device)
            
            logits = model(ids, mask)
            
            logits_flat = logits.reshape(-1, logits.size(-1))
            labels_flat = labels.reshape(-1)
            
            #!!! Создаем mask для исключения pad-токенов
            loss_mask = (labels_flat != tokenizer.pad_token_id)
            if loss_mask.sum() == 0:  # если в батче только pad-токены
                continue
            #!!! Применяем mask
            logits_masked = logits_flat[loss_mask]
            labels_masked = labels_flat[loss_mask]
            
            #!!! Считаем loss только по реальным токенам
            loss = criterion(logits_masked, labels_masked)
            total_loss += loss.item() * loss_mask.sum().item()  # взвешенный loss
            total_tokens += loss_mask.sum().item()  # общее количество реальных токенов
            
            # loss = criterion(logits_flat, labels_flat)
            # total_loss += loss.item()
            
            batch_preds = torch.argmax(logits, dim=-1)
            #!!!добавлено
            non_pad_mask = (labels != tokenizer.pad_token_id)
            
            # preds += torch.argmax(logits, dim=-1).cpu().flatten().tolist()
            # preds += batch_preds.cpu().flatten().tolist()
            # trues += labels.cpu().flatten().tolist()
            
            #!!!добавлено
            preds_non_pad = batch_preds[non_pad_mask].cpu().tolist()
            trues_non_pad = labels[non_pad_mask].cpu().tolist()            
            preds.extend(preds_non_pad)
            trues.extend(trues_non_pad)
            
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
                        
                        # Для задачи предсказания следующего токена сравниваем продолжения
                        # Reference - ожидаемое продолжение, Prediction - сгенерированное продолжение
                        if true_text.strip() and pred_text.strip():
                            references.append(true_text)
                            predictions.append(pred_text)
                            examples_processed += 1
                            
                    except Exception as e:
                        print(f"⚠️ Ошибка при обработке примера для ROUGE: {e}")
                        continue
    
    # accuracy = accuracy_score(trues, preds)
    # avg_loss = total_loss / len(loader)
    #!!!!исправлено
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    accuracy = accuracy_score(trues, preds) if trues and preds else 0.0
    # Вычисляем ROUGE метрики если требуется
    rouge_metrics = None
    if compute_rouge and references and predictions:
        try:
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
            
            # print(f"📊 ROUGE метрики ({len(references)} примеров):")
            # print(f"   ROUGE-1: {rouge_metrics['rouge1']:.4f}")
            # print(f"   ROUGE-2: {rouge_metrics['rouge2']:.4f}")
            # print(f"   ROUGE-L: {rouge_metrics['rougeL']:.4f}")
            
        except Exception as e:
            print(f"❌ Ошибка при вычислении ROUGE: {e}")
            rouge_metrics = {'error': str(e)}
    return accuracy, avg_loss, rouge_metrics


def test_model(model, loader, criterion, device):
    model.eval()
    all_preds, all_trues = [], []
    test_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing"):
            ids = batch['data'].to(device)
            mask = batch['mask'].to(device)
            labels = batch['target'].to(device)
            
            logits = model(ids, mask)
            logits_flat = logits.reshape(-1, logits.size(-1))
            labels_flat = labels.reshape(-1)
            
            loss = criterion(logits_flat, labels_flat)
            test_loss += loss.item()
            
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

def analyze_predictions(model, loader, tokenizer, device, num_examples=5):
    model.eval()
    bad_cases, good_cases = [], []
    
    with torch.no_grad():
        for batch in loader:
            ids = batch['data'].to(device)
            mask = batch['mask'].to(device)
            labels = batch['target'].to(device)
            
            logits = model(ids, mask)
            preds = torch.argmax(logits, dim=-1)
            
            for i in range(ids.size(0)):
                non_pad_indices = mask[i].bool()
                if non_pad_indices.sum() == 0:
                    continue
                
                input_ids_real = ids[i][non_pad_indices]
                preds_real = preds[i][non_pad_indices]
                labels_real = labels[i][non_pad_indices]
                
                input_tokens = tokenizer.convert_ids_to_tokens(input_ids_real.cpu())
                true_tokens = tokenizer.convert_ids_to_tokens(labels_real.cpu())
                pred_tokens = tokenizer.convert_ids_to_tokens(preds_real.cpu())
                
                min_length = min(len(input_tokens)-1, len(true_tokens), len(pred_tokens))
                
                for j in range(min_length):
                    if j >= len(true_tokens) or j >= len(pred_tokens):
                        continue
                        
                    context = input_tokens[:j+1]
                    true_tok = true_tokens[j]
                    pred_tok = pred_tokens[j]
                    
                    skip_tokens = ['[PAD]', '[CLS]', '[SEP]', '<pad>', '<cls>', '<sep>', '']
                    if true_tok in skip_tokens or pred_tok in skip_tokens:
                        continue
                    if true_tok == pred_tok and true_tok in skip_tokens:
                        continue
                    
                    if true_tok != pred_tok:
                        bad_cases.append((context, true_tok, pred_tok))
                    else:
                        good_cases.append((context, true_tok, pred_tok))
            
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

def analyze_error_patterns(bad_cases, tokenizer):
    if not bad_cases:
        print("Нет ошибок для анализа! 🎉")
        return
    
    error_pairs = [(true, pred) for _, true, pred in bad_cases]
    error_counter = Counter(error_pairs)
    
    print("\n🔝 Топ-10 самых частых ошибок:")
    for (true_tok, pred_tok), count in error_counter.most_common(10):
        print(f"  '{true_tok}' → '{pred_tok}': {count} раз")
    
    length_errors = [abs(len(true_tok) - len(pred_tok)) for _, true_tok, pred_tok in bad_cases]
    if length_errors:
        avg_length_diff = sum(length_errors) / len(length_errors)
        print(f"\n📏 Средняя разница в длине токенов при ошибках: {avg_length_diff:.2f}")