import os
import sys
sys.path.append(os.path.join(sys.path[0], '..'))

import torch
from torch import nn
from transformer import Transformer
from global_variables.devices import device
from global_variables.paths import PATH_DATASET, PATH_SAVE_MODEL

dataset = torch.load(PATH_DATASET)

# Загрузка модели, настройка оптимизатора и функции потерь.
model = Transformer(num_tokens=90000, dim_model=512, num_heads=8, num_encoder_layers=6, num_decoder_layers=6, dropout_p=0.1).to(device)
opt = torch.optim.SGD(model.parameters(), lr=0.01)
# opt = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()


def train_loop(model: Transformer, opt: torch.optim, loss_fn: nn, dataloader):
    '''
    Обновление весов модели.

    Input:
        model - Обучаемая модель
        opt - Оптимизатор
        loss_fn - Функция потерь
        dataloader - Датасет для обучения модели
    
    Output:
        Training loss
    '''

    model.train()
    total_loss = 0
    
    for batch in dataloader:
        X, y = batch[:, 0], batch[:, 1]
        X, y = torch.tensor(X, dtype=torch.long).to(device), torch.tensor(y, dtype=torch.long).to(device)

        y_input = y[:,:-1]
        y_expected = y[:,1:]

        sequence_length = y_input.size(1)
        tgt_mask = model.get_tgt_mask(sequence_length).to(device)

        pred = model(X, y_input, tgt_mask)
        pred = pred.permute(1, 2, 0)    

        loss = loss_fn(pred, y_expected)
        opt.zero_grad()
        loss.backward()
        opt.step()
    
        total_loss += loss.detach().item()
        
    return total_loss / len(dataloader)


def validation_loop(model, loss_fn, dataloader):
    '''
    Цикл проверки модели.

    Input:
        model - Обучаемая модель
        loss_fn - Функция потерь
        dataloader - Датасет для обучения модели

    Output:
        Validation loss
    '''

    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            X, y = batch[:, 0], batch[:, 1]
            X, y = torch.tensor(X, dtype=torch.long, device=device), torch.tensor(y, dtype=torch.long, device=device)

            y_input = y[:,:-1]
            y_expected = y[:,1:]
            
            sequence_length = y_input.size(1)
            tgt_mask = model.get_tgt_mask(sequence_length).to(device)

            pred = model(X, y_input, tgt_mask)

            pred = pred.permute(1, 2, 0)      
            loss = loss_fn(pred, y_expected)
            total_loss += loss.detach().item()
        
    return total_loss / len(dataloader)


def fit(model, opt, loss_fn, train_dataloader, val_dataloader, epochs):
    '''
    Обучение модели: 'train_loop', 'validation_loop'.

    Input:
        model - Обучаемая модель
        opt - Оптимизатор
        loss_fn - Функция потерь
        train_dataloader - Датасет для обучения модели
        val_dataloader - Датасет для проверки качества модели
        epochs - Количество эпох обучения

    Output:
        Списки Train loss и Validation loss
    '''

    train_loss_list, validation_loss_list = [], []
    
    print("Training and validating model")
    for epoch in range(epochs):
        print("-"*25, f"Epoch {epoch + 1}","-"*25)
        
        train_loss = train_loop(model, opt, loss_fn, train_dataloader)
        train_loss_list += [train_loss]
        
        validation_loss = validation_loop(model, loss_fn, val_dataloader)
        validation_loss_list += [validation_loss]
        
        print(f"Training loss: {train_loss:.4f}")
        print(f"Validation loss: {validation_loss:.4f}")
        print()
        
    return train_loss_list, validation_loss_list


train_loss_list, validation_loss_list = fit(model, opt, loss_fn, dataset, dataset, 1)

# Сохранение модели
torch.save(model.state_dict(), PATH_SAVE_MODEL)
