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
from .global_variables.paths import PATH_RAW_DATA, PATH_DATASET, PATH_SAVE_MODEL, PATH_TOK_INP, PATH_TOK_OUT

class Model:
    '''
    Класс для создания модели 'Transformer'.  
    '''

    def __init__(self, num_tokens: int, dim_model=512, num_heads=8, 
                 num_encoder_layers=6, num_decoder_layers=6, dropout_p=0.1, 
                 path_dataset=PATH_DATASET, path_save_model=PATH_SAVE_MODEL, 
                 path_tok_inp=PATH_TOK_INP, path_tok_out=PATH_TOK_OUT) -> None:
        
        # Настройка модели, оптимизатора и функции потери.
        self.model = Transformer(num_tokens=num_tokens, 
                                 dim_model=dim_model, 
                                 num_heads=num_heads, 
                                 num_encoder_layers=num_encoder_layers, 
                                 num_decoder_layers=num_decoder_layers,
                                 dropout_p=dropout_p).to(device)
        self.opt = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.loss_fn = nn.CrossEntropyLoss()

        # Загрузка датасета и токенайзеров.
        self.dataset = torch.load(path_dataset)
        self.nlp = spacy.load('ru_core_news_sm')
        with open(path_tok_inp, 'rb') as f:
            self.tokenizer_input = pickle.load(f)
        with open(path_tok_out, 'rb') as f:
            self.tokenizer_output = pickle.load(f)

        # Путь к обученной модели.
        self.path_save_model = path_save_model


    def train_save_model(self) -> None:
        '''
        Тренировка модели и её сохранение.
        '''

        train_loss_list, validation_loss_list = fit(self.model, self.opt, self.loss_fn, self.dataset, self.dataset, 1000)
        save_model(self.model, path=self.path_save_model)

    
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
        self.model.load_state_dict(torch.load(PATH_SAVE_MODEL))

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
        sentence = ''
        for i in result:
            sentence += ' ' + self.tokenizer_output[i]

        return sentence
    

def create_dataset(max_len: int, data_property: str, batch_size: int, seq_len: int, step=1, num_of_datasets=2, path_raw_data=PATH_RAW_DATA) -> None:
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
    data = Dataset(path_raw_data, max_len)
    data.create_data()
    data.conversion_to_tokens()
    dataset = data.creating_batches(data_property, batch_size, seq_len, step, num_of_datasets) # (25, 100, 10)
    dataset = data.get_data()
    dataset = dataset.astype(int)
    print(f"Итоговый размер датасета: {dataset.shape}")

    # Преобразование датасета в тензор. Создание словаря и списка для токенизации.
    dataset = torch.tensor(dataset, dtype=torch.long).to(device)
    tokenizer_output = data.tokenizer_output
    tokenizer_input = data.tokenizer_input

    # Сохранение tokenizer_output.
    tokenizer_output_path = os.path.join(sys.path[0], 'src\\datasets\\prepared_datasets\\tokenizer_output.pkl')
    with open(tokenizer_output_path, 'wb') as f:
        pickle.dump(tokenizer_output, f)

    # Сохранение tokenizer_input.
    tokenizer_input_path = os.path.join(sys.path[0], 'src\\datasets\\prepared_datasets\\tokenizer_input.pkl')
    with open(tokenizer_input_path, 'wb') as f:
        pickle.dump(tokenizer_input, f)
            
    # Сохранение dataset.
    dataset_path = os.path.join(sys.path[0], 'src\\datasets\\prepared_datasets\\dataset.pt')
    torch.save(dataset, dataset_path)
