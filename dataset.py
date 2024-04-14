import torch
from torch import nn
from torchtext.data import get_tokenizer
from sklearn.preprocessing import LabelEncoder
import numpy as np
from errors import InvalidDataType, TextLengthError, DatasetBatchSizeError


class Dataset:
    def __init__(self, path: str, batch_size: int, max_words: int, method: str) -> None:
        if not isinstance(path, str): raise InvalidDataType(path, str)
        elif not isinstance(batch_size, int): raise InvalidDataType(batch_size, int)
        elif not isinstance(max_words, int): raise InvalidDataType(max_words, int)
        elif not isinstance(method, str): raise InvalidDataType(method, str)

        self.le = LabelEncoder()
        text = self.get_text_from_path(path)

        # Division into sentences
        self.tokenizer = get_tokenizer("basic_english")
        text = self.tokenizer(text)
        text = text[:max_words-1]
        if text[-1] != '.': text = np.append(text, '.')
        if len(text) != max_words: raise TextLengthError(len(text), max_words)

        data = self.insert_st_ed(text) 
        self.le.fit(text)
        if method == 'phrases': 
            text, dataset_x, dataset_y = self.creating_datasets_from_phrases(data, batch_size)
            self.dataset_x, self.dataset_y = self.creating_tensors_x_y(text, dataset_x, dataset_y)
            del text, dataset_x, dataset_y

    def get_datasets(self):
        return (self.dataset_x, self.dataset_y)

    def processing_sentence(self, sentence: str):
        sentence = self.tokenizer(sentence)
        if sentence[-1] != '.': sentence = np.append(sentence, '.')
        print(sentence)
        sentence = torch.tensor(self.le.transform(sentence))
        return sentence

    def get_text_from_path(self, path: str) -> str:
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
            text = text.replace('\\ufeff', '')
        if text[-1] != '.': text += '.'
        return text
    
    def insert_st_ed(self, data: list[str]):
        length = len(data)
        prepared_data = np.array(['[START]'])
        for i in range(length):
            prepared_data = np.append(prepared_data, data[i])
            if data[i] == '.':
                prepared_data = np.append(prepared_data, '[END]')
                if i != length-1:
                    prepared_data = np.append(prepared_data, '[START]')
        return prepared_data

    def creating_datasets_from_phrases(self, data: np.array, batch_size: int):
        length = len(data)
        if length % batch_size != 0: length -= length % batch_size - 10
        text = data[:length]
        dataset_x, dataset_y, data = [], [], []
        flag = True
        for i in range(length):
            data.append(text[i])
            if len(data) == batch_size:
                if flag:
                    dataset_x.append(data)
                    flag = False
                else:
                    dataset_y.append(data)
                    flag = True
                data = []
        dataset_x = np.array(dataset_x)
        dataset_y = np.array(dataset_y)
        print(f"Размер словаря: '{length}'.")
        print(f"Размер dataset_x: '{dataset_x.shape}'.")
        print(f"Размер dataset_y: '{dataset_y.shape}'.")
        return text, dataset_x, dataset_y
    
    def creating_tensors_x_y(self, text: np.array, dataset_x: np.array, dataset_y: np.array):

        # Transfer into tokens
        dataset_x = np.array([list(self.le.transform(i)) for i in dataset_x])
        dataset_y = np.array([list(self.le.transform(i)) for i in dataset_y])

        # Transfer back into words
        # data_orig_x = np.array([list(self.le.inverse_transform(i)) for i in dataset_x])
        # data_orig_y = np.array([list(self.le.inverse_transform(i)) for i in dataset_y])

        # Creating Tensors(x, y)
        dataset_x = torch.tensor(dataset_x)
        dataset_y = torch.tensor(dataset_y)
        return dataset_x, dataset_y