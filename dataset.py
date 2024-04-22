import torch
import numpy as np
from devices import device
import spacy
import pickle
from errors import IncorrectDatasetSize

class Dataset:
    def __init__(self, path: str, max_len: int, batch_size: int, seq_len: int, num_of_datasets=2) -> None:
        if max_len % batch_size*seq_len*num_of_datasets != 0: raise IncorrectDatasetSize(max_len, batch_size, seq_len, num_of_datasets)

        self.tokenizer_output = {}
        self.tokenizer_input = []
        self.nlp = spacy.load('ru_core_news_lg')
        self.path = path
        self.max_len = max_len
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_of_datasets = num_of_datasets

    def create_data(self):
        '''
        Создание ещё не обработанного датасета
        '''

        # Получение текста из файла. Преобразование его в список состоящий из слов
        print("-"*25, 'CREATE DATA', "-"*25)
        text = self.get_text_from_path(self.path)
        text = self.nlp(text)

        # Изменение размера переменной text
        text = text[:self.max_len-1]
        if text[-1] != '.': text = np.append(text, '.')
        if len(text) != self.max_len: text = np.append(text, '[END]')

        # Изменение размера и добавление [START] и [END] токенов
        data = self.insert_st_ed(text) 
        self.data = data[:self.max_len-1]
        if self.data[-1] != '.': self.data = np.append(self.data, '.')
        if len(self.data) != self.max_len: self.data = np.append(self.data, '[END]')
        print(f"Итоговый размер датасета: '{len(self.data)}'.")

        del text
        del data

    def get_data(self):
        '''
        Получение готового датасета.
        '''

        print("-"*25, 'GET DATA', "-"*25)
        return self.data
    
    def creating_batches(self):
        '''
        Создание батчей из датасета.
        '''

        print("-"*25, 'CREATING BATCHES', "-"*25)
        self.data = self.data.reshape((self.max_len // (self.batch_size*self.num_of_datasets*self.seq_len), self.batch_size, self.num_of_datasets, self.seq_len))

    def conversion_to_tokens(self):
        '''
        Преобразование текста в токены. Создание токенайзера в виде словаря - 'tokenizer_output' и списка - 'tokenizer_input'.
        '''

        print("-"*25, 'CONVERSION TO TOKENS', "-"*25)
        for i in range(len(self.data)):
            self.tokenizer_output[i] = str(self.data[i])
            self.tokenizer_input.append(str(self.data[i]))
            self.data[i] = i

    def get_text_from_path(self, path: str) -> str:
        '''
        Получение текста из заданного пути.
        '''

        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
            text = text.replace('\\ufeff', '')
        if text[-1] != '.': text += '.'
        return text
    
    def insert_st_ed(self, data: list[str]) -> np.array:
        '''
        Включение в датасет стоп сигналов: [START] и [END].
        '''

        length = len(data)
        prepared_data = np.array('[START]')
        for i in range(length):
            prepared_data = np.append(prepared_data, data[i])
            if data[i] == '.':
                prepared_data = np.append(prepared_data, '[END]')
                if i != length-1:
                    prepared_data = np.append(prepared_data, '[START]')
        return prepared_data
    

# Создание датасета.
data = Dataset('data/dataset.txt', 90000, 25, 10)
data.create_data()
data.conversion_to_tokens()
dataset = data.creating_batches()
dataset = data.get_data()
dataset = dataset.astype(int)

# Преобразование датасета в тензор. Создание словаря и списка для токенизации.
dataset = torch.tensor(dataset, dtype=torch.long).to(device)
tokenizer_output = data.tokenizer_output
tokenizer_input = data.tokenizer_input

# Сохранение tokenizer_output.
with open('tokenizer_output.pkl', 'wb') as f:
    pickle.dump(tokenizer_output, f)

# Сохранение tokenizer_input.
with open('tokenizer_input.txt', 'w', encoding="utf-8") as f:
    f.writelines(f"{item}" for item in tokenizer_input)
    
# Сохранение dataset.
torch.save(dataset, 'data/dataset.pt')

