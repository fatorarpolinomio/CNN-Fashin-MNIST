"""
Trabalho de IA 2026 - Objetivo 2 (CNN com Fashion MNIST)

Integrantes:
    Lucas Neves Souza, Nro. USP: 15577627
    Willian Jefferson Farias, Nro. USP: 15642665

train.py: lógica de treinamento (laço de épocas, função de erro, otimizador e validação).
"""

import torch
from torch import nn

# Laço de repetição para treinamento
def train_loop(train_dataloader, model, loss_function, optimizer):
    size = len(train_dataloader.dataset)

    # Selecionando modo de treino do modelo
    model.train()

    for batch, (X, y) in enumerate(train_dataloader):

        # Computando predição e perda
        prediction = model(X)
        loss = loss_function(prediction, y)

        # Realizando Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            # Printando a loss atual
            loss, current = loss.item(), batch * train_dataloader.batch_size + len(X)
            print(f"Loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

def test_loop(test_dataloader, model, loss_function, validation=False):

    # Selecionando o modo de avaliação do modelo
    model.eval()
    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in test_dataloader:
            pred = model(X)
            test_loss += loss_function(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    if not validation:
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    else:
        print(f"Validation Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def model_training(
    model,
    train_dataloader,
    validation_dataloader,
    test_dataloader,
    loss_function,
    optimizer,
    epochs : int
):
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------")
        train_loop(train_dataloader, model, loss_function, optimizer)
        test_loop(validation_dataloader, model, loss_function, validation=True)

    test_loop(test_dataloader, model, loss_function, validation=False)
    print("Finished!")
