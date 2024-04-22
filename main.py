import torch
from devices import device
import spacy
import pickle

def predict(model, input_sequence, max_length=15, SOS_token=2, EOS_token=3):
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
    model = torch.load('data/transformer')
    tokenizer_input = []
    nlp = spacy.load('ru_core_news_sm')

    with open('data/tokenizer_input.txt', 'r', encoding='utf-8') as f:
        tokenizer_input.append(f.readlines())
    with open('data/tokenizer_output.pkl', 'rb') as f:
        tokenizer_output = pickle.load(f)
    tokenizer_input = tokenizer_input[0]

    sentence = input()
    sentence = nlp(sentence)
    
    sentence_processed = []
    for i in sentence:
        if f'{str(i)}\n'in tokenizer_input:
            sentence_processed.append(tokenizer_input.index(f'{str(i)}\n'))
        else:
            tokenizer_input.append(f'{str(i)}\n')
            sentence_processed.append(tokenizer_input.index(f'{str(i)}\n'))
    print(sentence_processed)

    sentence_processed = torch.tensor([sentence_processed], dtype=torch.long).to(device)
    result = predict(model, sentence_processed)
    sentence = ''
    for i in result:
        sentence += ' ' + tokenizer_output[i]
    print(sentence)

if __name__ == '__main__':
    main()