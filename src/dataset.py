"""
Trabalho de IA 2026 - Objetivo 2 (CNN com Fashion MNIST)

Integrantes:
    Lucas Neves Souza, Nro. USP: 15577627
    Willian Jefferson Farias, Nro. USP: 15642665

dataset.py: carga, pré-processamento, divisão treino/validação/teste
e filtro de classes (com remapeamento de rótulos para 0/1) da tarefa binária.
"""

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets
from torchvision.transforms import v2
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"   # <raiz>/data, não depende de onde você roda

CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# Função para normalizar imagens
def get_transform():

    transform = v2.Compose([
        v2.ToImage(), # Converte a imagem em um tensor do PyTorch
        v2.ToDtype(torch.float32, scale=True), # Converte os pixels para números decimais
        v2.Normalize((0.5,), (0.5,)) # aplica a conta (x - média) / desvio a cada pixel
    ])

    return transform

def split_train_val(training_data, seed):

    # Definindo seed
    generator = torch.Generator().manual_seed(seed)

    # Particionando dataset de treinamento
    # treino 50.000, validação 10.000 e teste 10.000,
    # ou seja, ≈ 71%, 14% e 14% do total de 70.000
    training_data, validation_data = random_split(training_data, [50000, 10000], generator)

    return training_data, validation_data

# Função para carregar o dataset
def load_fashionMNIST(seed):
    # O FashionMNIST é disponibilizado pelo torchvision, então podemos baixar diretamente:
    # Baixando dados de treinamento
    training_data = datasets.FashionMNIST(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=get_transform(),
    )

    # Baixando dados de teste
    test_data = datasets.FashionMNIST(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=get_transform(),
    )
    return training_data, test_data

# Função para converter nomes de classes em índices
def classes_para_indices(nomes, rotulo="classe"):
    # Converte uma lista de nomes de classes (como em CLASSES) nos índices 0 a 9
    # O parâmetro rotulo ("classe_0" ou "classe_1") só serve para a mensagem de erro

    if not isinstance(nomes, (list, tuple)):
        raise ValueError(
            f"{rotulo} deve ser uma lista de nomes de classes (ex.: [\"Shirt\"]), "
            f"mas recebeu {nomes!r}"
        )
    if len(nomes) == 0:
        raise ValueError(f"{rotulo} não pode ser uma lista vazia")
    if len(set(nomes)) != len(nomes):
        raise ValueError(f"{rotulo} tem nomes duplicados: {list(nomes)}")

    desconhecidas = [n for n in nomes if n not in CLASSES]
    if desconhecidas:
        raise ValueError(
            f"{rotulo} tem nomes que não existem no Fashion MNIST: {desconhecidas}. "
            f"Nomes válidos: {CLASSES}"
        )

    return [CLASSES.index(n) for n in nomes]

# Função para obter os rótulos originais do dataset
def obter_rotulos(dataset):
    # Devolve o tensor com os rótulos ORIGINAIS (0 a 9) de todos os exemplos do dataset

    if isinstance(dataset, Subset):
        return obter_rotulos(dataset.dataset)[torch.as_tensor(dataset.indices, dtype=torch.long)]
    return torch.as_tensor(dataset.targets, dtype=torch.long)

# Classe para criar um dataset binário a partir de um dataset original
class BinaryDataset(Dataset):
    #Visão binária de um dataset: só traz as classes escolhidas, com rótulo 0 ou 1

    def __init__(self, subset, targets):
        self.subset = subset
        self.targets = targets  # tensor de 0/1, alinhado com as posições de self.subset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, i):
        imagem, _rotulo_original = self.subset[i]
        return imagem, int(self.targets[i])

# Função para filtrar classes binárias e remapear rótulos
def filter_binary_classes(dataset, classes_0, classes_1):
    # Filtra `dataset` para as duas classes (ou composições de classes) e remapeia os rótulos.

    #classes_0: lista de nomes de classes que formam a classe negativa (rótulo 0).
    #classes_1: lista de nomes de classes que formam a classe positiva (rótulo 1).

    indices_0 = classes_para_indices(classes_0, "classe_0")
    indices_1 = classes_para_indices(classes_1, "classe_1")

    # Uma classe não pode estar nos dois lados, senão o rótulo ficaria ambíguo.
    repetidas = sorted(set(indices_0) & set(indices_1))
    if repetidas:
        raise ValueError(
            f"classe_0 e classe_1 não podem compartilhar classes: {[CLASSES[i] for i in repetidas]}"
        )

    rotulos = obter_rotulos(dataset)
    eh_classe_0 = torch.isin(rotulos, torch.tensor(indices_0))
    eh_classe_1 = torch.isin(rotulos, torch.tensor(indices_1))

    # Posições (dentro de `dataset`) dos exemplos que pertencem a alguma das duas classes
    posicoes = torch.nonzero(eh_classe_0 | eh_classe_1).squeeze(1)

    # Rótulo binário: True (classe_1) vira 1 e False (classe_0) vira 0
    rotulos_binarios = eh_classe_1[posicoes].long()

    return BinaryDataset(Subset(dataset, posicoes.tolist()), rotulos_binarios)

# Função para criar datasets binários a partir de datasets originais
def create_binary_datasets(config, training_data, validation_data, test_data):
    # Aplica filter_binary_classes nas três partições, com o par definido em config.yaml (bloco `binario`).
    classes_0 = config["binario"]["classe_0"]
    classes_1 = config["binario"]["classe_1"]

    return (
        filter_binary_classes(training_data, classes_0, classes_1),
        filter_binary_classes(validation_data, classes_0, classes_1),
        filter_binary_classes(test_data, classes_0, classes_1),
    )

# Função para criar dataloaders, o foco aqui é em otimização do processamento
def create_dataloaders(config, training_data, validation_data, test_data):

    # Definindo um batch_size a partir do valor que vem de config.yaml.
    # Cada elemento no iterável do dataloader
    # retornará um lote de características e rótulos com o valor especificado
    # em config.yaml
    batch_size = config["dados"]["batch_size"]
    g = torch.Generator().manual_seed(config["seed"])

    # Criando data loaders, passando os datasets como argumento
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, generator=g) # dataloader dos dados de treino
    test_dataloader = DataLoader(test_data, batch_size=batch_size) # dataloader dos dados de teste
    validation_dataloader = DataLoader(validation_data, batch_size=batch_size) # dataloader dos dados de validação

    return train_dataloader, validation_dataloader, test_dataloader
