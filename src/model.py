import os 
import sys
import torch
from torch import nn
import spacy
import pickle
from .datasets.creating_datasets import Dataset
from .model_components.training_model import fit, save_model
from .model_components.transformer import Transformer
from .global_variables.devices import device
from .global_variables.paths import PATH_FOLDER

class Model:
    '''
    Класс для создания модели 'Transformer'.  
    '''

    def __init__(self, num_tokens: int, dim_model=512, num_heads=8, 
                 num_encoder_layers=6, num_decoder_layers=6, dropout_p=0.1, 
                 path_dataset='dataset.pt', path_tok_inp='tokenizer_input.pkl', 
                 path_tok_out='tokenizer_output.pkl', path_save_model='transformer') -> None:
        
        # Настройка модели, оптимизатора и функции потери.
        self.model = Transformer(num_tokens=num_tokens, 
                                 dim_model=dim_model, 
                                 num_heads=num_heads, 
                                 num_encoder_layers=num_encoder_layers, 
                                 num_decoder_layers=num_decoder_layers,
                                 dropout_p=dropout_p).to(device)
        self.opt = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.loss_fn = nn.CrossEntropyLoss()

        # Пути к датасетам.
        PATH_DATASET = os.path.join(PATH_FOLDER, path_dataset)
        PATH_TOK_INP = os.path.join(PATH_FOLDER, path_tok_inp)
        PATH_TOK_OUT = os.path.join(PATH_FOLDER, path_tok_out)
        self.PATH_SAVE_MODEL = os.path.join(PATH_FOLDER, path_save_model)

        # Загрузка датасета и токенайзеров.
        self.dataset = torch.load(PATH_DATASET)
        self.nlp = spacy.load('ru_core_news_sm')
        with open(PATH_TOK_INP, 'rb') as f:
            self.tokenizer_input = pickle.load(f)
        with open(PATH_TOK_OUT, 'rb') as f:
            self.tokenizer_output = pickle.load(f)


    def train_save_model(self, epochs: int, add_train=False) -> None:
        '''
        Тренировка модели и её сохранение.
        '''
        if add_train: self.model.load_state_dict(torch.load(self.PATH_SAVE_MODEL))
        train_loss_list, validation_loss_list = fit(self.model, self.opt, self.loss_fn, self.dataset, self.dataset, epochs)
        save_model(self.model, path=self.PATH_SAVE_MODEL)

    
    def internal_predict(self, model: Transformer, input_sequence: torch.tensor, max_length: int, SOS_token=2, EOS_token=3) -> list[int]:
        '''
        Предсказание модели для функции predict.
        Input:
            model - Сама модель.
            input_sequence - Входной текст.
            max_length - Максимальный выходной размер текста.
        '''
        
        self.model.eval()
        y_input = torch.tensor([[SOS_token]], dtype=torch.long, device=device)
        num_tokens = len(input_sequence[0])

        for _ in range(max_length):
            tgt_mask = self.model.get_tgt_mask(y_input.size(1)).to(device)
            pred = self.model(input_sequence, y_input, tgt_mask)
            
            next_item = pred.topk(1)[1].view(-1)[-1].item() 
            next_item = torch.tensor([[next_item]], device=device)

            y_input = torch.cat((y_input, next_item), dim=1)
            if next_item.view(-1).item() == EOS_token:
                break

        return y_input.view(-1).tolist()


    def predict(self, sentence: str, max_length: int) -> str:
        '''
        Предсказание модели.
        Input:
            sentence - Входное предложение.
            max_length - Длина выходной последовательности.
        Output:
            sentence - Предсказание модели.
        '''

        # 1. Загрузка модели и токенайзера
        self.model.load_state_dict(torch.load(self.PATH_SAVE_MODEL))

        # 2. Подготовка входного текста 
        sentence = self.nlp(sentence)
        sentence_processed = []
        for i in sentence:
            if str(i)in self.tokenizer_input:
                sentence_processed.append(self.tokenizer_input.index(str(i)))
            else:
                self.tokenizer_input.append(str(i))
                sentence_processed.append(self.tokenizer_input.index(str(i)))
        sentence_processed = torch.tensor([sentence_processed], dtype=torch.long).to(device)
        print(sentence_processed)

        # 3. Model predict 
        result = self.internal_predict(self.model, sentence_processed, max_length=max_length)
        print(result)
        sentence = ''
        for i in result:
            sentence += ' ' + self.tokenizer_output[i]

        return sentence
    

def create_dataset(max_len: int, data_property: str, batch_size: int, 
                   seq_len: int, step=1, num_of_datasets=2, path_raw_data='raw_data.txt', 
                   path_tok_inp='tokenizer_input.pkl', path_tok_out='tokenizer_output.pkl', path_dataset='dataset.pt') -> None:
    '''
    Создание датасета для обучения модели.
    Input:
        max_len - Максимальное количество слов в датасета.
        data_property - Структура датасета: 'without meaning', 'phrase training'.
        batch_size - Размер партии.
        seq_len - Длина последовательности.
        step - Шаг создания фраз.
        num_of_datasets - Количество датасетов участвующих в обучении или проверки, по умолчанию 'обучение - 2'.
        path_raw_data - Путь к еще не обработанным данным, которые станут частью датасета.
    '''

    # Создание датасета.  
    PATH_RAW_DATA = os.path.join(PATH_FOLDER, path_raw_data)  
    data = Dataset(PATH_RAW_DATA, max_len)
    data.create_data()
    data.conversion_to_tokens()
    dataset = data.creating_batches(data_property, batch_size, seq_len, step, num_of_datasets) # (25, 100, 10)
    dataset = data.get_data()
    dataset = dataset.astype(int)
    print(f"Итоговый размер датасета: {dataset.shape}")

    # Преобразование датасета в тензор. Создание словаря и списка для токенизации.
    dataset = torch.tensor(dataset, dtype=torch.long).to(device)
    tokenizer_input = data.tokenizer_input
    tokenizer_output = data.tokenizer_output

    # Сохранение tokenizer_input.
    PATH_TOK_INP = os.path.join(PATH_FOLDER, path_tok_inp)
    with open(PATH_TOK_INP, 'wb') as f:
        pickle.dump(tokenizer_input, f)

    # Сохранение tokenizer_output.
    PATH_TOK_OUT = os.path.join(PATH_FOLDER, path_tok_out)
    with open(PATH_TOK_OUT, 'wb') as f:
        pickle.dump(tokenizer_output, f)
            
    # Сохранение dataset.
    PATH_DATASET = os.path.join(PATH_FOLDER, path_dataset)
    torch.save(dataset, PATH_DATASET)
