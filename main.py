import torch
import spacy
import pickle
from src.global_variables.devices import device
from src.global_variables.paths import PATH_SAVE_MODEL, PATH_TOK_INP, PATH_TOK_OUT
from src.model.transformer import Transformer


def predict(model: Transformer, input_sequence: torch.tensor, max_length=15, SOS_token=2, EOS_token=3):
    '''
    Предсказание модели
    Input:
        model - Сама модель.
        input_sequence - Входной текст.
        max_length - Максимальный выходной размер текста.
    '''
    
    model.eval()
    y_input = torch.tensor([[SOS_token]], dtype=torch.long, device=device)
    num_tokens = len(input_sequence[0])

    for _ in range(max_length):
        tgt_mask = model.get_tgt_mask(y_input.size(1)).to(device)
        pred = model(input_sequence, y_input, tgt_mask)
        
        next_item = pred.topk(1)[1].view(-1)[-1].item() 
        next_item = torch.tensor([[next_item]], device=device)

        y_input = torch.cat((y_input, next_item), dim=1)
        if next_item.view(-1).item() == EOS_token:
            break

    return y_input.view(-1).tolist()


def main():
    # 1. Загрузка модели и токенайзера
    model = Transformer(num_tokens=90000, dim_model=512, num_heads=8, num_encoder_layers=6, num_decoder_layers=6, dropout_p=0.1).to(device)
    model.load_state_dict(torch.load(PATH_SAVE_MODEL))
    nlp = spacy.load('ru_core_news_sm')

    with open(PATH_TOK_INP, 'rb') as f:
        tokenizer_input = pickle.load(f)

    with open(PATH_TOK_OUT, 'rb') as f:
        tokenizer_output = pickle.load(f)

    # 2. Подготовка входного текста 
    sentence = 'высокоуровневый язык программирования.'
    sentence = nlp(sentence)
    sentence_processed = []
    for i in sentence:
        if str(i)in tokenizer_input:
            sentence_processed.append(tokenizer_input.index(str(i)))
        else:
            tokenizer_input.append(str(i))
            sentence_processed.append(tokenizer_input.index(str(i)))
    sentence_processed = torch.tensor([sentence_processed], dtype=torch.long).to(device)

    # 3. Model predict 
    result = predict(model, sentence_processed)
    sentence = ''
    for i in result:
        sentence += ' ' + tokenizer_output[i]
    print(sentence)

if __name__ == '__main__':
    main()