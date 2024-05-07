import numpy as np
import spacy
from ..global_variables.errors import IncorrectDatasetSize

class Dataset:
    def __init__(self, file_path: str, max_len: int) -> None:

        self.tokenizer_output = {}
        self.tokenizer_input = []
        self.nlp = spacy.load('ru_core_news_lg')

        self.file_path = file_path
        self.max_len = max_len

    def create_data(self):
        '''
        FIRST STEP: Создание ещё не обработанного датасета
        '''

        # Получение текста из файла. Преобразование его в список состоящий из слов
        print("-"*25, 'CREATE DATA', "-"*25)
        text = self.get_text_from_path(self.file_path)
        text = self.nlp(text)
        print(f'Размер raw_data: {len(text)}')
        
        # Изменение размера переменной text
        text = text[:self.max_len-1]
        if text[-1] != '.': text = np.append(text, '.')
        if len(text) != self.max_len: text = np.append(text, '[END]')

        # Изменение размера и добавление [START] и [END] токенов
        data = self.insert_st_ed(text) 
        self.data = data[:self.max_len-1]
        if self.data[-1] != '.': self.data = np.append(self.data, '.')
        if len(self.data) != self.max_len: self.data = np.append(self.data, '[END]')
        print(f"Предварительный размер необработанного датасета: '{len(self.data)}'.")

        del text
        del data

    def get_data(self):
        '''
        FOURTH STEP: Получение готового датасета.
        '''

        print("-"*25, 'GET DATA', "-"*25)
        return self.data
    
    def creating_batches(self, data_property: str, batch_size: int, seq_len: int, step=1, num_of_datasets=2):
        '''
        THIRD STEP: Создание батчей из датасета.
        Input: 
            data_property - Свойство датасета.
            batch_size - Размер партии.
            seq_len - Длинна последовательности (Длина фразы).
            step - Шаг создания фраз.
            num_of_datasets - Количество датасетов участвующих в обучении или проверки, по умолчанию 'обучение - 2'.
        Output:
            dataset with shape(length // (batch_size*num_of_datasets*seq_len), batch_size, num_of_datasets, seq_len)
        '''

        if self.max_len % batch_size*seq_len*num_of_datasets != 0: raise IncorrectDatasetSize(self.max_len, batch_size, seq_len, num_of_datasets)
        print("-"*25, 'CREATING BATCHES', "-"*25)

        # Нет смысловой связки между словами
        if data_property == 'without meaning':
            self.data = self.data.reshape((self.max_len // (batch_size*num_of_datasets*seq_len), batch_size, num_of_datasets, seq_len))

        # Разделение текста на фразы (есть смысловая зависимость)
        elif data_property == 'phrase training': 
            prepared_data = np.array([])
            for i in range(0, len(self.data), step):
                if len(self.data[i:i+seq_len]) < seq_len: break
                prepared_data = np.append(prepared_data, self.data[i:i+seq_len])

            if len(prepared_data) % (batch_size*num_of_datasets*seq_len):
                length = len(prepared_data) - (len(prepared_data) % (batch_size*num_of_datasets*seq_len))
                prepared_data = prepared_data[:length]

            self.data = prepared_data.reshape((len(prepared_data) // (batch_size*num_of_datasets*seq_len), batch_size, num_of_datasets, seq_len))

    def conversion_to_tokens(self):
        '''
        SECOND STEP: Преобразование текста в токены. Создание токенайзера в виде словаря - 'tokenizer_output' и списка - 'tokenizer_input'.
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