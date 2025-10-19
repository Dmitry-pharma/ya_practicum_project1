import torch
from tqdm import tqdm
from pathlib import Path
import json
from datetime import datetime
# from sklearn.model_selection import train_test_split
# from torch.utils.data import DataLoader
# from next_token_dataset import TweetsDataset
# import matplotlib.pyplot as plt
# import random
# from collections import Counter
import numpy as np
# from transformers import AutoTokenizer
from transformers import BertTokenizerFast

#src
from data_utils import samples_preparation #load_and_clean_data, prepare_training_pairs
from lstm_model import NextPhrasePredictionRNN
from eval_lstm import vevaluate, test_model, analyze_predictions, analyze_error_patterns, show_detailed_examples
from visualization import create_visualization

# tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
# tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(loader, desc="Train", leave=False)
    
    for batch in progress_bar:
        ids = batch['data'].to(device)
        mask = batch['mask'].to(device)
        labels = batch['target'].to(device)

        optimizer.zero_grad()
        logits = model(ids, mask)
        
        logits_flat = logits.reshape(-1, logits.size(-1))
        labels_flat = labels.reshape(-1)
        
        loss = criterion(logits_flat, labels_flat)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
    return total_loss / len(loader)

def save_model(model, optimizer, epoch, accuracy, loss, path, tokenizer, config):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy,
        'loss': loss,       
        'model_config': {
            'vocab_size': config['vocab_size'],
            'emb_dim': config['emb_dim'],
            'hidden_dim': config['hidden_dim'],
            'pad_idx': config['pad_idx']           
        }
    }, path)


# def load_model(model, optimizer, path):
#     """Загружает модель и её параметры"""
#     checkpoint = torch.load(path)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     return checkpoint['epoch'], checkpoint['accuracy'], checkpoint['loss']



def main():
    # Конфигурация
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    config = {
        'file_path': Path(__file__).parent.parent / 'data' / '1_raw_dataset_tweets.txt',
        'model_dir': Path(__file__).parent.parent / 'models',
        'MAX_LEN': 20,
        'limit': 100,
        'hidden_dim': 64,#512
        'emb_dim': 300,
        'batch_size': 256,
        'rnn_type': "LSTM",
        'tokenizer': tokenizer,
        'pad_idx':tokenizer.pad_token_id,
        'vocab_size': tokenizer.vocab_size,
        'compute_rouge': True
    }
    
    
    print('Config: ',config)
    
    config['model_dir'].mkdir(exist_ok=True)
    
    # Загрузка и подготовка данных
    
    
    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader=samples_preparation(
                file_path=config['file_path'],
                limit=config['limit'],
                tokenizer=tokenizer,
                MAX_LEN=config['MAX_LEN'],
                batch_size=config['batch_size']
                )
    
    # Модель и обучение
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # pad_token_id = tokenizer.pad_token_id
    
    model = NextPhrasePredictionRNN(
        rnn_type=config['rnn_type'],
        vocab_size=tokenizer.vocab_size,
        emb_dim=config['emb_dim'],
        hidden_dim=config['hidden_dim'],
        pad_idx=config['pad_idx']#pad_token_id
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=config['pad_idx'])#pad_token_id)
    
    # Обучение (основной цикл)
    train_losses, val_accuracies, val_losses, val_rouge_metrics = [], [], [], []
    best_val_acc, best_epoch, patience_counter = 0, 0, 0
    patience = 5
    
    print("🎯 Начало обучения...")
    epoch_loop = tqdm(range(10), desc="Обучение", unit="эпоха")
    
    for epoch in epoch_loop:
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        
        val_acc, val_loss, val_rouge = vevaluate(model, val_loader, criterion, device, tokenizer, compute_rouge = config['compute_rouge'], num_rouge_examples=50)
         
             
        val_accuracies.append(val_acc)
        val_losses.append(val_loss)
        val_rouge_metrics.append(val_rouge)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            
            best_model_path = config['model_dir'] / 'best_model.pth'
            save_model(model, optimizer, epoch, val_acc, val_loss, best_model_path, 
                      tokenizer, config)
            print(f"💾 Сохранена лучшая модель с accuracy: {val_acc:.4f}")
        else:
            patience_counter += 1
        # Обновляем прогресс-бар с ROUGE метриками
        rouge_info = ""
        if val_rouge and 'rouge1' in val_rouge:
            rouge_info = f" | R1: {val_rouge['rouge1']:.3f}"
            
        # epoch_loop.set_postfix({
        #     "Loss": f"{train_loss:.4f}",
        #     "Val Acc": f"{val_acc:.4f}",
        #     "Best Acc": f"{best_val_acc:.4f}",
        #     "Patience": f"{patience_counter}/{patience}"
        # })
        
        postfix_data = {
        "Loss": f"{train_loss:.4f}",
        "Val Acc": f"{val_acc:.4f}",
        "Best Acc": f"{best_val_acc:.4f}",
        "Patience": f"{patience_counter}/{patience}"
        }
        # Добавляем ROUGE метрики если они есть
        if val_rouge and 'rouge1' in val_rouge:
            postfix_data["R1"] = f"{val_rouge['rouge1']:.3f}"
            postfix_data["R2"] = f"{val_rouge['rouge2']:.3f}"
    
        epoch_loop.set_postfix(postfix_data)
        
        if patience_counter >= patience:
            print(f"🛑 Ранняя остановка на эпохе {epoch + 1}")
            break
        
    # После цикла обучения добавляем анализ ROUGE метрик
    if config['compute_rouge']:
        print("\n" + "="*60)
        print("📈 АНАЛИЗ ROUGE МЕТРИК ПО ЭПОХАМ")
        print("="*60)

        for epoch, (acc, rouge) in enumerate(zip(val_accuracies, val_rouge_metrics)):
            if rouge and 'rouge1' in rouge:
                print(f"Эпоха {epoch+1}: Acc={acc:.4f} | ROUGE-1={rouge['rouge1']:.4f} | ROUGE-2={rouge['rouge2']:.4f} | ROUGE-L={rouge['rougeL']:.4f}")
        
    # Тестирование и анализ
    print(f"🔄 Загружаем лучшую модель из эпохи {best_epoch}...")    
    checkpoint = torch.load(config['model_dir'] / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_accuracy, test_loss = test_model(model, test_loader, criterion, device)
             
    
    # Сохранение результатов
    results = {
        'best_validation_accuracy': best_val_acc,
        'best_epoch': best_epoch,
        'test_accuracy': test_accuracy,
        'test_loss': test_loss,
        'final_training_epochs': epoch + 1,
        'dataset_statistics': {
            'number_of_input_samples': config['limit'],
            'train_samples': len(train_ds),
            'validation_samples': len(val_ds),
            'test_samples': len(test_ds),
            'total_samples': len(train_ds) + len(val_ds) + len(test_ds),
        },
        'model_architecture': {
            'model_type': 'NextPhrasePredictionRNN',
            'rnn_type': config['rnn_type'],
            'vocab_size': tokenizer.vocab_size,
            'embedding_dim': config['emb_dim'],
            'hidden_dim': config['hidden_dim'],
            'max_sequence_length': config['MAX_LEN'],
        }
    }
    
    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M')
    results_filename = f'training_results_{timestamp}.json'
    with open(config['model_dir'] / results_filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Результаты сохранены в: {results_filename}")
    
    # Анализ предсказаний
    bad_cases, good_cases = analyze_predictions(model, test_loader, tokenizer, device)
    analyze_error_patterns(bad_cases, tokenizer)
    
        
            
    # Детальный анализ конкретных примеров    
    show_detailed_examples(model, test_loader, tokenizer, num_examples=3)
    # Визуализация результатов
    # create_visualization(train_losses, val_losses, val_accuracies, best_val_acc, test_accuracy, config)
    create_visualization(
        train_losses=train_losses,
        val_losses=val_losses,
        val_accuracies=val_accuracies,
        best_val_acc=best_val_acc,
        test_accuracy=test_accuracy,
        val_rouge_metrics=val_rouge_metrics,
        #test_rouge_metrics=test_rouge,  # если есть тестовые ROUGE метрики
        config=config
    )
    print(f"\n✅ Обучение завершено!")
    print(f"📁 Модели сохранены в: {config['model_dir']}")
    print(f"📊 Графики сохранены в: training_results.png")


if __name__ == "__main__":
    main()