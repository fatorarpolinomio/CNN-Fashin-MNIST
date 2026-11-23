# Justificativas

Trabalho de IA 2026 - Objetivo 2 (CNN com Fashion MNIST)

## Integrantes:
- Lucas Neves Souza, Nro. USP: 15577627
- Willian Jefferson Farias, Nro. USP: 15642665

## Loop de treinamento

A implementação do **Loop de Treinamento** está em `train.py`. Nesse arquivo em específico, temos três funções: `train_loop`, `test_loop` e `model_training`.

As duas primeiras são combinadas e aplicadas em `model_training`, que será a nossa função principal a ser utilizada para treinar o modelo.

A justificativa para essa organização reside no uso de boas práticas de programação: construção de funções com responsabilidade única para melhor escalabilidade e, aqui neste contexto, depuração.

### train_loop

Realiza o **treinamento do modelo para uma época**. 

1) Predição & Perda:

- Fazemos a predição através da chamada `model(X)` e, logo em seguida, computamos a perda através da nossa `loss_function`

2) Backpropagation:

- Realizando atualização dos pesos `loss.backward()` optimizer.step() optimizer.zero_grad()

### test_loop
### model_training
