import evaluate

# Загружаем метрику
rouge = evaluate.load("rouge")

# Списки с предсказанными и эталонными суммаризациями (референсными)
predictions = [
    "The cat sat on the mat and looked around."
]

references = [
    "A cat was sitting on the mat and observing the surroundings."
]

# Вычисляем метрику
results = rouge.compute(predictions=predictions, references=references)

# Печатаем значения
for key, value in results.items():
    print(f"{key}: {value:.4f}")