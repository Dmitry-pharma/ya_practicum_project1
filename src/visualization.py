
import matplotlib.pyplot as plt
import numpy as np

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
        