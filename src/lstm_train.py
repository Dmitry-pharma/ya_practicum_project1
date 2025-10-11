import torch
from tqdm import tqdm
from pathlib import Path
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random
from collections import Counter
import numpy as np
from data_utils import load_and_clean_data, prepare_training_pairs
from next_token_dataset import TweetsDataset
from lstm_model import NextPhrasePredictionRNN
from eval_lstm import vevaluate, test_model, analyze_predictions, analyze_error_patterns, show_detailed_examples
from transformers import AutoTokenizer
from transformers import BertTokenizerFast
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
        'limit': 50000,
        'hidden_dim': 512,
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
    # tokenizer=config['tokenizer']
    # tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    
    texts_df = load_and_clean_data(config['file_path'], config['limit'])
    data = prepare_training_pairs(texts_df, tokenizer, config['MAX_LEN'])
    
    # Разделение данных
    train, temp = train_test_split(data, test_size=0.2, random_state=42)
    val, test = train_test_split(temp, test_size=1/2, random_state=42)
    
    X_train, Y_train, M_train = zip(*train)
    X_val, Y_val, M_val = zip(*val)
    X_test, Y_test, M_test = zip(*test)
    
    train_ds = TweetsDataset(X_train, Y_train, M_train)
    val_ds = TweetsDataset(X_val, Y_val, M_val)
    test_ds = TweetsDataset(X_test, Y_test, M_test)
    
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'])
    test_loader = DataLoader(test_ds, batch_size=config['batch_size'])
    
    print(f"✅ Датасеты созданы: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")
    
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
    
        
    # def create_visualization(train_losses, val_losses, val_accuracies, best_val_acc, test_accuracy, config):
    #     """Создает визуализацию результатов обучения"""
    #     plt.figure(figsize=(15, 5))

    #     # График потерь
    #     plt.subplot(1, 3, 1)
    #     plt.plot(range(1, len(train_losses) + 1), train_losses, 
    #              label='Train Loss', color='tab:blue', linewidth=2, marker=None)
    #     plt.plot(range(1, len(val_losses) + 1), val_losses, 
    #              label='Val Loss', color='tab:red', linewidth=2, marker=None)
    #     plt.title('Training and Validation Loss')
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Loss')
    #     plt.grid(True)
    #     plt.legend()

    #     # График точности
    #     plt.subplot(1, 3, 2)
    #     plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, 
    #              label='Validation Accuracy', color='tab:orange', linewidth=2, marker=None)
    #     plt.axhline(y=best_val_acc, color='red', linestyle='--', label=f'Best Acc: {best_val_acc:.4f}')
    #     plt.title('Validation Accuracy')
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Accuracy')
    #     plt.ylim(0, 1)
    #     plt.grid(True)
    #     plt.legend()

    #     # График сравнения метрик
    #     plt.subplot(1, 3, 3)
    #     metrics = ['Best Val Acc', 'Test Acc']
    #     values = [best_val_acc, test_accuracy]
    #     colors = ['lightblue', 'lightgreen']
    #     bars = plt.bar(metrics, values, color=colors)
    #     plt.title('Final Metrics Comparison')
    #     plt.ylabel('Accuracy')
    #     plt.ylim(0, 1)
    #     for bar, value in zip(bars, values):
    #         plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
    #                  f'{value:.4f}', ha='center', va='bottom')

    #     plt.tight_layout()  
    #     plt.savefig(config['model_dir'] / 'training_results.png', dpi=150, bbox_inches='tight')
    #     plt.show()
    def create_visualization(train_losses, val_losses, val_accuracies, best_val_acc, test_accuracy, 
                        val_rouge_metrics=None, config=None):
        """Создает визуализацию результатов обучения с ROUGE метриками"""
    
        # Определяем layout в зависимости от того, вычислялись ли ROUGE метрики
        compute_rouge = config and config.get('compute_rouge', False)
        has_rouge_data = val_rouge_metrics and any(m and 'rouge1' in m for m in val_rouge_metrics)
    
        if compute_rouge and has_rouge_data:
            # Создаем фигуру с 2 рядами графиков
            fig = plt.figure(figsize=(16, 10))
            n_rows = 2
            n_cols = 3
        else:
            # Создаем фигуру с 1 рядом графиков (без ROUGE)
            fig = plt.figure(figsize=(15, 5))
            n_rows = 1
            n_cols = 3
    
        # Первый ряд: основные метрики
        # График потерь
        plt.subplot(n_rows, n_cols, 1)
        plt.plot(range(1, len(train_losses) + 1), train_losses, 
                 label='Train Loss', color='tab:blue', linewidth=2, marker=None)
        plt.plot(range(1, len(val_losses) + 1), val_losses, 
                 label='Val Loss', color='tab:red', linewidth=2, marker=None)
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # График точности
        plt.subplot(n_rows, n_cols, 2)
        plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, 
                 label='Validation Accuracy', color='tab:orange', linewidth=2, marker=None)
        plt.axhline(y=best_val_acc, color='red', linestyle='--', label=f'Best Acc: {best_val_acc:.4f}')
        plt.title('Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.legend()

        # График сравнения метрик
        plt.subplot(n_rows, n_cols, 3)
        metrics = ['Best Val Acc', 'Test Acc']
        values = [best_val_acc, test_accuracy]
        colors = ['lightblue', 'lightgreen']
        bars = plt.bar(metrics, values, color=colors, edgecolor='black', alpha=0.7)
        plt.title('Final Metrics Comparison')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                     f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
        # Второй ряд: ROUGE метрики (только если compute_rouge=True и есть данные)
        if compute_rouge and has_rouge_data:
            # Извлекаем ROUGE scores
            rouge1_scores = [m['rouge1'] if m and 'rouge1' in m else 0 for m in val_rouge_metrics]
            rouge2_scores = [m['rouge2'] if m and 'rouge2' in m else 0 for m in val_rouge_metrics]
            rougeL_scores = [m['rougeL'] if m and 'rougeL' in m else 0 for m in val_rouge_metrics]
        
            epochs = range(1, len(rouge1_scores) + 1)
        
            # График ROUGE метрик по эпохам
            plt.subplot(n_rows, n_cols, 4)
            plt.plot(epochs, rouge1_scores, label='ROUGE-1', color='purple', linewidth=2, marker='o', markersize=4)
            plt.plot(epochs, rouge2_scores, label='ROUGE-2', color='green', linewidth=2, marker='s', markersize=4)
            plt.plot(epochs, rougeL_scores, label='ROUGE-L', color='orange', linewidth=2, marker='^', markersize=4)
            plt.title('ROUGE Metrics Over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('ROUGE Score')
            plt.ylim(0, 1)
            plt.grid(True, alpha=0.3)
            plt.legend()
        
            # График сравнения ROUGE метрик
            plt.subplot(n_rows, n_cols, 5)
            rouge_names = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
            rouge_final = [rouge1_scores[-1], rouge2_scores[-1], rougeL_scores[-1]]
            colors_rouge = ['purple', 'green', 'orange']
            bars_rouge = plt.bar(rouge_names, rouge_final, color=colors_rouge, edgecolor='black', alpha=0.7)
            plt.title('Final ROUGE Metrics')
            plt.ylabel('ROUGE Score')
            plt.ylim(0, 1)
            for bar, value in zip(bars_rouge, rouge_final):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                         f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
            # График связи Accuracy и ROUGE-1
            plt.subplot(n_rows, n_cols, 6)
            #  Берем столько эпох, сколько есть данных для обоих метрик
            min_len = min(len(val_accuracies), len(rouge1_scores))
            acc_subset = val_accuracies[:min_len]
            rouge1_subset = rouge1_scores[:min_len]
        
            plt.scatter(acc_subset, rouge1_subset, alpha=0.6, color='red', s=50)
            plt.xlabel('Accuracy')
            plt.ylabel('ROUGE-1')
            plt.title('Accuracy vs ROUGE-1 Correlation')
            plt.grid(True, alpha=0.3)
        
            # Добавляем линию тренда
            if len(acc_subset) > 1:
                z = np.polyfit(acc_subset, rouge1_subset, 1)
                p = np.poly1d(z)
                plt.plot(acc_subset, p(acc_subset), "r--", alpha=0.8, 
                        label=f'Trend: y={z[0]:.3f}x+{z[1]:.3f}')
                plt.legend()
    
        elif compute_rouge and not has_rouge_data:
            # Если ROUGE был включен, но данных нет - показываем информационное сообщение
            for i in range(4, 7):
                plt.subplot(n_rows, n_cols, i)
                plt.text(0.5, 0.5, 'No ROUGE data available\nCheck evaluation function', 
                        ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
                plt.axis('off')

        plt.tight_layout()
    
        # Сохраняем график
        if config and 'model_dir' in config:
            plt.savefig(config['model_dir'] / 'training_results.png', dpi=150, bbox_inches='tight')
        else:
            plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
    
        plt.show()
            
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