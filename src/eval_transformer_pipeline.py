from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

def evaluate_transformer(model_name="distilgpt2", test_texts=None):
    """
    Оценка трансформер модели для сравнения с LSTM
    """
    print(f"🧪 Оценка трансформер модели: {model_name}")
    
    # Загрузка модели и токенизатора
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Создание pipeline для генерации текста
    text_generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )
    
    # Примеры для тестирования
    if test_texts is None:
        test_texts = [
            "I love to",
            "The weather is",
            "I think that",
            "In my opinion",
            "The best way to"
        ]
    
    print("\n🔍 Тестирование трансформера на примерах:")
    print("=" * 50)
    
    for i, prompt in enumerate(test_texts, 1):
        try:
            # Генерация продолжения
            result = text_generator(
                prompt,
                max_length=len(prompt.split()) + 5,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id
            )
            
            generated_text = result[0]['generated_text']
            continuation = generated_text[len(prompt):].strip()
            
            print(f"{i}. Промпт: '{prompt}'")
            print(f"   Продолжение: '{continuation}'")
            print()
            
        except Exception as e:
            print(f"❌ Ошибка при генерации для '{prompt}': {e}")
    
    # Оценка perplexity (если есть тестовые данные)
    if test_texts:
        try:
            perplexity = calculate_perplexity(model, tokenizer, test_texts, device)
            print(f"📊 Perplexity модели: {perplexity:.2f}")
        except:
            print("⚠️  Не удалось вычислить perplexity")
    
    return model, tokenizer

def calculate_perplexity(model, tokenizer, texts, device, max_length=512):
    """
    Вычисление perplexity на наборе текстов
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for text in tqdm(texts, desc="Calculating perplexity"):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            
            total_loss += loss.item() * inputs["input_ids"].size(1)
            total_tokens += inputs["input_ids"].size(1)
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))
    
    return perplexity.item()

if __name__ == "__main__":
    evaluate_transformer("distilgpt2")