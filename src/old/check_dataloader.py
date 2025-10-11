
import torch

from torch.utils.data import Dataset, DataLoader


# 1. Реализуйте класс ToyDataset

class ToyDataset(Dataset):
    def __init__(self, data, targets):
        # Сохраните data и targets в атрибуты
        # Ваш код здесь
        self.data = data
        self.targets = targets 
        pass


    def __len__(self):
        # Верните длину датасета
        # Ваш код здесь
        return len(self.data)
        pass


    def __getitem__(self, idx):
        # Верните пару (вход, метка), соответствующую индексу idx
        # Ваш код здесь
        return {'data':self.data[idx],
                'target':self.targets[idx]                
            }
        pass


# 2. Подготовьте входные списки
raw_data = list(range(10)) # data = [0, 1, 2, ..., 9]
raw_targets = [2 * x for x in raw_data]  # targets = [0, 2, 4, ..., 18]


# 3. Создайте объект ToyDataset

dataset = ToyDataset(data=raw_data, targets=raw_targets)# Ваш код здесь)
#dataset    emb = out.mean(dim=1).cpu().numpy()  # [batch, hidden]= MyDataset(data=list(range(10)), targets=[2*x for x in range(10)])

# 4. Оберните датасет в DataLoader с batch_size=4 и shuffle=True

#dataloader = DataLoader(# Ваш код здесь)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

# Переберите первые два батча и выведите их на экран
# for batch_idx, (batch_inputs, batch_targets) in enumerate(dataloader):
#     print(f"Батч {batch_idx + 1}:")
#     print("Inputs: ", batch_inputs)
#     print("Targets:", batch_targets)
#     if batch_idx == 1:  # остановимся после второго батча
#         break
for batch in dataloader:
    print(batch)