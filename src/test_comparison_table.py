def compare_models_lstm_vs_gpt2():
    """Сравнение LSTM и distilGPT2 моделей с контролем длины генерации"""
    
    # Загрузка тестовой выборки
    current_dir = Path.cwd()
    v_file_path = current_dir / 'data' / 'test_dataset.csv'
    test_df = load_samples(v_file_path)
    
    # Инициализация моделей и токенизаторов
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # LSTM модель 1
    model_path = current_dir / 'models' / 'lstm_50K_best_model.pth'
    lstm_tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint['model_config']
    
    lstm_model = NextPhrasePredictionRNN(
        rnn_type="LSTM",
        vocab_size=model_config['vocab_size'],
        emb_dim=model_config['emb_dim'],
        hidden_dim=model_config['hidden_dim'],
        pad_idx=model_config['pad_idx']
    ).to(device)
    lstm_model.load_state_dict(checkpoint['model_state_dict'])
    lstm_model.eval()
    
    # GRU модель 2
    model_path2 = current_dir / 'models' / 'gru_MaxL10NumEp100Lim500000NumL2Hidd128Emb300Lr1E-4WDec005_best_model.pth'
    # lstm_tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    checkpoint2 = torch.load(model_path2, map_location=device)
    model_config2 = checkpoint2['model_config']
    
    lstm_model2 = NextPhrasePredictionRNN(
        rnn_type="GRU",
        vocab_size=model_config2['vocab_size'],
        emb_dim=model_config2['emb_dim'],
        hidden_dim=model_config2['hidden_dim'],
        pad_idx=model_config2['pad_idx'],
        num_layers=model_config2['num_layers']
    ).to(device)
    lstm_model2.load_state_dict(checkpoint2['model_state_dict'])
    lstm_model2.eval()
    
    
    # distilGPT2 модель
    gpt2_tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    if gpt2_tokenizer.pad_token is None:
        gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
    
    gpt2_model = GPT2LMHeadModel.from_pretrained("distilgpt2").to(device)
    gpt2_model.eval()
    
    # Загрузка ROUGE метрики
    rouge = evaluate.load('rouge')
    
    def calculate_expected_tokens_count(original_phrase, truncated_phrase, tokenizer):
        """Вычисляет количество токенов в ожидаемом продолжении"""
        full_tokens = tokenizer.tokenize(original_phrase)
        truncated_tokens = tokenizer.tokenize(truncated_phrase)
        expected_tokens_count = len(full_tokens) - len(truncated_tokens)
        return max(expected_tokens_count, 1)  # минимум 1 токен
    
    # Функция для генерации продолжения LSTM с контролем длины
    def generate_lstm_completion(lstm_model, prompt_text, expected_tokens_count):
        start_time = time.time()
        
        inputs = lstm_tokenizer(prompt_text, return_tensors='pt', truncation=True, max_length=50)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        current_input = input_ids.clone()
        current_mask = attention_mask.clone()
        generated_tokens = []
        
        with torch.no_grad():
            for step in range(expected_tokens_count):
                logits = lstm_model(current_input, current_mask)
                next_token_logits = logits[0, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1)
                next_token_id = next_token.item()
                
                generated_tokens.append(next_token_id)
                next_token_tensor = next_token.unsqueeze(0).unsqueeze(0)
                current_input = torch.cat([current_input, next_token_tensor], dim=1)
                
                new_mask = torch.ones(1, 1, device=device, dtype=torch.long)
                current_mask = torch.cat([current_mask, new_mask], dim=1)
                
                # Останавливаемся на EOS токене
                if next_token_id == lstm_tokenizer.eos_token_id:
                    break
        
        completion = lstm_tokenizer.decode(generated_tokens, skip_special_tokens=True)
        end_time = time.time()
        generation_time = end_time - start_time
        
        return completion, generation_time, len(generated_tokens)
    
    # Функция для генерации продолжения GPT2 с контролем длины
    def generate_gpt2_completion(prompt_text, expected_tokens_count):
        start_time = time.time()
        
        inputs = gpt2_tokenizer(prompt_text, return_tensors='pt', truncation=True, max_length=50)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = gpt2_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=input_ids.shape[1] + expected_tokens_count,
                num_return_sequences=1,
                do_sample=False,  # greedy decoding
                pad_token_id=gpt2_tokenizer.pad_token_id,
                eos_token_id=gpt2_tokenizer.eos_token_id,
                early_stopping=False,
            )
        
        # Извлекаем сгенерированную часть
        generated_tokens = outputs[0][input_ids.shape[1]:]
        
        # Обрезаем до нужного количества токенов
        if len(generated_tokens) > expected_tokens_count:
            generated_tokens = generated_tokens[:expected_tokens_count]
        
        completion = gpt2_tokenizer.decode(generated_tokens, skip_special_tokens=True)
        end_time = time.time()
        generation_time = end_time - start_time
        
        return completion, generation_time, len(generated_tokens)
    
    # Подготовка данных для сравнения
    results = []
    
    print("🔍 Начало сравнения моделей...")
    print(f"📊 Тестовая выборка: {len(test_df)} примеров")
    
    # for idx, row in test_df.iterrows():
    progress_bar = tqdm(test_df.iterrows(), total=len(test_df), desc="📊 Обработка примеров")
    for idx, row in progress_bar:
        if idx>=200:
            break
        progress_bar.set_postfix({
        "Пример": f"{idx + 1}/{len(test_df)}"
        # ,
        # "Текст": f"{row['text_cleaned'][:30]}..." if len(str(row['text_cleaned'])) > 30 else str(row['text_cleaned'])
    })
        
        original_phrase = row['text_cleaned']
        
        # Создаем урезанную фразу (первые 5-7 слов)
        words = original_phrase.split()
        if len(words) > 7:
            truncated_phrase = ' '.join(words[:7])
        else:
            truncated_phrase = ' '.join(words[:-1]) if len(words) > 1 else original_phrase
        
        # Извлекаем ожидаемое продолжение
        expected_continuation = original_phrase[len(truncated_phrase):].strip()
        
        # Вычисляем количество токенов в ожидаемом продолжении для каждой модели
        lstm_expected_tokens = calculate_expected_tokens_count(original_phrase, truncated_phrase, lstm_tokenizer)
        gpt2_expected_tokens = calculate_expected_tokens_count(original_phrase, truncated_phrase, gpt2_tokenizer)
        
        # if idx<5:
        #     tqdm.write(f"\n📝 Пример {idx + 1}/{len(test_df)}")
        #     tqdm.write(f"   Оригинал: '{original_phrase}'")
        #     tqdm.write(f"   Урезано: '{truncated_phrase}'")
        #     tqdm.write(f"   Ожидаемое: '{expected_continuation}'")
        #     tqdm.write(f"   Ожидаемое токенов - LSTM: {lstm_expected_tokens}, GPT2: {gpt2_expected_tokens}")
        
        # Генерация продолжений с контролем длины
        lstm_completion, lstm_time, lstm_actual_tokens = generate_lstm_completion(lstm_model, truncated_phrase, lstm_expected_tokens)
        lstm_completion2, lstm_time2, lstm_actual_tokens2 = generate_lstm_completion(lstm_model2, truncated_phrase, lstm_expected_tokens)
        gpt2_completion, gpt2_time, gpt2_actual_tokens = generate_gpt2_completion(truncated_phrase, gpt2_expected_tokens)
        
        # if idx<5:
        #     tqdm.write(f"   LSTM: '{lstm_completion}' ({lstm_actual_tokens} токенов, {lstm_time:.2f}с)")
        #     tqdm.write(f"   GRU: '{lstm_completion2}' ({lstm_actual_tokens2} токенов, {lstm_time2:.2f}с)")
        #     tqdm.write(f"   GPT2: '{gpt2_completion}' ({gpt2_actual_tokens} токенов, {gpt2_time:.2f}с)")
        
        # Вычисляем ROUGE метрики
        lstm_rouge = rouge.compute(
            predictions=[lstm_completion],
            references=[expected_continuation],
            use_stemmer=True
        ) if lstm_completion.strip() and expected_continuation.strip() else {'rougeL': 0.0}

        lstm_rouge2 = rouge.compute(
            predictions=[lstm_completion2],
            references=[expected_continuation],
            use_stemmer=True
        ) if lstm_completion2.strip() and expected_continuation.strip() else {'rougeL': 0.0}

        
        gpt2_rouge = rouge.compute(
            predictions=[gpt2_completion],
            references=[expected_continuation],
            use_stemmer=True
        ) if gpt2_completion.strip() and expected_continuation.strip() else {'rougeL': 0.0}
        
        results.append({
            'Оригинальная фраза': original_phrase,
            'Урезанная фраза': truncated_phrase,
            'Ожидаемое продолжение': expected_continuation,
            # 'Ожидаемое токенов LSTM': lstm_expected_tokens,
            # 'Ожидаемое токенов GPT2': gpt2_expected_tokens,
            'Продолжение фразы модели LSTM': lstm_completion,
            'Продолжение фразы модели GRU': lstm_completion2,
            
            # 'Фактическое токенов LSTM': lstm_actual_tokens,
            'Продолжение фразы модели distilGPT2': gpt2_completion,
            # 'Фактическое токенов GPT2': gpt2_actual_tokens,
            'rouge LSTM': round(lstm_rouge['rougeL'], 4),
            'rouge GRU': round(lstm_rouge2['rougeL'], 4),
            'rouge distilGPT2': round(gpt2_rouge['rougeL'], 4),
            'время генерации ответа LSTM': round(lstm_time, 4),
            'время генерации ответа GRU': round(lstm_time2, 4),
            'время генерации ответа distilGPT2': round(gpt2_time, 4)
        })
    
    # Создаем DataFrame с результатами
    results_df = pd.DataFrame(results)
    
    # Сохраняем в CSV
    # output_path = current_dir / 'results' / f'model_comparison_controlled_length_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    output_path = current_dir / 'results' / 'model_comparison.csv'
    output_path.parent.mkdir(exist_ok=True)
    results_df.to_csv(output_path, index=False, encoding='utf-8')
    
    # Вычисляем общую статистику
    total_stats = {
        'Средний ROUGE-L LSTM': results_df['rouge LSTM'].mean(),
        'Средний ROUGE-L GRU': results_df['rouge GRU'].mean(),       
        'Средний ROUGE-L GPT2': results_df['rouge distilGPT2'].mean(),
        'Общее время LSTM (сек)': results_df['время генерации ответа LSTM'].sum(),
        'Общее время LSTM (сек)': results_df['время генерации ответа GRU'].sum(),
        'Общее время GPT2 (сек)': results_df['время генерации ответа distilGPT2'].sum(),
        'Среднее время LSTM (сек)': results_df['время генерации ответа LSTM'].mean(),
        'Среднее время GRU (сек)': results_df['время генерации ответа GRU'].mean(),        
        'Среднее время GPT2 (сек)': results_df['время генерации ответа distilGPT2'].mean(),
        # 'Совпадение длины LSTM (%)': (results_df['Фактическое токенов LSTM'] == results_df['Ожидаемое токенов LSTM']).mean() * 100,
        # 'Совпадение длины GPT2 (%)': (results_df['Фактическое токенов GPT2'] == results_df['Ожидаемое токенов GPT2']).mean() * 100,
        'Количество примеров': len(results_df)
    }
    
    # Определяем лучшую модель
    # if total_stats['Средний ROUGE-L LSTM'] > total_stats['Средний ROUGE-L GPT2']:
    #     best_model = "LSTM"
    #     advantage = total_stats['Средний ROUGE-L LSTM'] - total_stats['Средний ROUGE-L GPT2']
    # else:
    #     best_model = "distilGPT2"
    #     advantage = total_stats['Средний ROUGE-L GPT2'] - total_stats['Средний ROUGE-L LSTM']
    
    model_scores = {
    'LSTM': total_stats['Средний ROUGE-L LSTM'],
    'GRU': total_stats['Средний ROUGE-L GRU'], 
    'distilGPT2': total_stats['Средний ROUGE-L GPT2']
    }
    # Находим лучшую модель    
    # Сортируем модели по убыванию ROUGE-L
    sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)

    
    
    # Вывод результатов
    print("\n" + "="*80)
    print("📊 ИТОГОВАЯ СТАТИСТИКА СРАВНЕНИЯ МОДЕЛЕЙ")
    print("="*80)
    print(f"🤖 LSTM модель:")
    print(f"   Средний ROUGE-L: {total_stats['Средний ROUGE-L LSTM']:.4f}")
    # print(f"   Совпадение длины: {total_stats['Совпадение длины LSTM (%)']:.1f}%")
    print(f"   Общее время генерации: {total_stats['Общее время LSTM (сек)']:.2f} сек")
    print(f"   Среднее время на пример: {total_stats['Среднее время LSTM (сек)']:.4f} сек")
    #---------------------------------------------------------------------------------------
    print(f"🤖 GRU модель:")
    print(f"   Средний ROUGE-L: {total_stats['Средний ROUGE-L GRU']:.4f}")
    # print(f"   Совпадение длины: {total_stats['Совпадение длины LSTM (%)']:.1f}%")
    print(f"   Общее время генерации: {total_stats['Общее время GRU (сек)']:.2f} сек")
    print(f"   Среднее время на пример: {total_stats['Среднее время GRU (сек)']:.4f} сек")
    #---------------------------------------------------------------------------------------    
    print(f"\n🤖 distilGPT2 модель:")
    print(f"   Средний ROUGE-L: {total_stats['Средний ROUGE-L GPT2']:.4f}")
    # print(f"   Совпадение длины: {total_stats['Совпадение длины GPT2 (%)']:.1f}%")
    print(f"   Общее время генерации: {total_stats['Общее время GPT2 (сек)']:.2f} сек")
    print(f"   Среднее время на пример: {total_stats['Среднее время GPT2 (сек)']:.4f} сек")
    #---------------------------------------------------------------------------------------
    # print(f"\n🏆 ЛУЧШАЯ МОДЕЛЬ: {best_model}")
    # print(f"   Преимущество в ROUGE-L: {advantage:.4f}")
    print(f"\n🏆 РЕЙТИНГ МОДЕЛЕЙ:")
    [print(f"   {i}. {m}: {s:.4f}" + (f" (отставание: {sorted_models[0][1]-s:.4f})" if i>1 else "")) 
    for i, (m, s) in enumerate(sorted_models, 1)]
    
    # if total_stats['Общее время LSTM (сек)'] < total_stats['Общее время GPT2 (сек)']:
    #     time_advantage = total_stats['Общее время GPT2 (сек)'] - total_stats['Общее время LSTM (сек)']
    #     print(f"   LSTM быстрее на: {time_advantage:.2f} сек")
    # else:
    #     time_advantage = total_stats['Общее время LSTM (сек)'] - total_stats['Общее время GPT2 (сек)']
    #     print(f"   GPT2 быстрее на: {time_advantage:.2f} сек")
    
    
    
    print(f"\n💾 Результаты сохранены в: {output_path}")
    
    # Сохраняем статистику в отдельный файл
    stats_df = pd.DataFrame([total_stats])
    stats_path = current_dir / 'results' / f'model_stats_controlled_length_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    stats_df.to_csv(stats_path, index=False)
    
    return results_df, total_stats, best_model


# Запуск сравнения
if __name__ == "__main__":
    results, stats, best_model = compare_models_lstm_vs_gpt2()