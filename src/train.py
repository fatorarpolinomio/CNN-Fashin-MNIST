"""
Trabalho de IA 2026 - Objetivo 2 (CNN com Fashion MNIST)

Integrantes:
    Lucas Neves Souza, Nro. USP: 15577627
    Willian Jefferson Farias, Nro. USP: 15642665

train.py: lógica de treinamento (laço de épocas, função de erro, otimizador e validação).
"""

import torch
import copy

# Classe para implementar a parada antecipada
class EarlyStopping:
    def __init__(self, model, patience=5, delta=0, verbose=False):
        self.model = model
        self.patience = patience # paciência, definida por padrão como 5, mas é possível alterar
        self.delta = delta # mudança mínima na perda
        self.verbose = verbose # booleano para indicar presença ou não de print para debug
        self.best_loss = None
        self.best_state = None
        self.no_improvement_count = 0
        self.stop_training = False

    def check_early_stop(self, val_loss):

        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            # Guardando a melhor loss que apareceu durante a validação
            self.best_loss = val_loss
            self.best_state = copy.deepcopy(self.model.state_dict())  # cópia dos melhores pesos
            # Zerando contador de ausência de melhorias, dado que uma melhoria aconteceu
            self.no_improvement_count = 0
        else:
            # Sempre que a validation loss falhar na melhoria além do delta,
            # incrementamos no_improvement_count
            self.no_improvement_count += 1

            # Se alcançarmos o limite da paciência, fazemos a parada
            if self.no_improvement_count >= self.patience:
                self.stop_training = True

                # Opção verbosa, para ajudar com depuração
                if self.verbose:
                    print("Fazendo a parada antecipada, dado que melhorias não foram observadas")

# Função para treinar modelo, se possível, com um acelerador, e.g. CUDA, MPS, MTIA ou XPU
def get_device_for_training():
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Usando o dispositivo: {device}")
    return device

# Laço de repetição para treinamento
def train_loop(train_dataloader, model, loss_function, optimizer, device):
    size = len(train_dataloader.dataset)

    # Selecionando modo de treino do modelo
    model.train()

    # Definindo perda média do treinamento modelo
    loss_treino = 0

    for batch, (X, y) in enumerate(train_dataloader):

        # Utilizando dispositivo de aceleração
        X, y = X.to(device), y.to(device)

        # Computando predição e perda
        prediction = model(X)
        loss = loss_function(prediction, y)

        # Realizando Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Incrementando para dividir depois
        loss_treino = loss_treino + loss.item()

        if batch % 100 == 0:
            # Printando a loss atual
            loss, current = loss.item(), batch * train_dataloader.batch_size + len(X)
            print(f"Perda: {loss:>7f} [{current:>5d}/{size:>5d}]")


    # Realizando a divisão da soma pela quantidade de batchs
    loss_treino /= len(train_dataloader)

    return loss_treino

def test_loop(test_dataloader, model, loss_function, device, validation=False):

    # Selecionando o modo de avaliação do modelo
    model.eval()
    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in test_dataloader:

            # Utilizando dispositivo de aceleração
            X, y = X.to(device), y.to(device)

            # Gerando predição
            pred = model(X)
            test_loss += loss_function(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    if not validation:
        print(f"Erro de Teste: \n Acurácia: {(100*correct):>0.1f}%, Perda Média: {test_loss:>8f} \n")
    else:
        print(f"Erro de Validação: \n Acurácia: {(100*correct):>0.1f}%, Perda Média: {test_loss:>8f} \n")

    return test_loss, correct



def model_training(
    model,
    train_dataloader,
    validation_dataloader,
    test_dataloader,
    loss_function,
    optimizer,
    device,
    config,
    epochs : int
):
    # Buscando os hiperparâmetros definidos no config.yaml
    cfg = config["parada"]
    patience = cfg["patience"] # Número de paciência
    delta = cfg["delta"] # mudança mínima na perda
    verbose = cfg["verbose"] # com ou sem print de depuração

    # Inicializando mecanismo de parada antecipada
    early_stopping = EarlyStopping(model, patience=patience, delta=delta, verbose=verbose)

    # Definindo lista de histórico
    historico = []

    for t in range(epochs):
        print(f"Época {t+1}\n-------------------")
        loss_treino = train_loop(train_dataloader, model, loss_function, optimizer, device)
        val_loss, val_acc = test_loop(validation_dataloader, model, loss_function, device, validation=True)

        # Verificando condição da parada antecipada
        early_stopping.check_early_stop(val_loss)

        # Atribuindo na lista para devolver depois
        historico.append({"epoca": t + 1, "loss_treino": loss_treino,
                          "loss_val": val_loss, "acc_val": val_acc})

        # Aplicando a parada, se for o caso
        if early_stopping.stop_training:
            print(f"Parando antecipadamente na época: {t+1}")
            break

    if early_stopping.best_state is not None:
        print("Restaurando os pesos da melhor época")
        model.load_state_dict(early_stopping.best_state)

    test_loop(test_dataloader, model, loss_function, device, validation=False)
    print("Finalizado!")

    return historico
