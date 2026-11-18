"""
Trabalho de IA 2026 - Objetivo 2 (CNN com Fashion MNIST)

Integrantes:
    Lucas Neves Souza, Nro. USP: 15577627
    Willian Jefferson Farias, Nro. USP: 15642665

dataset.py: carga, pré-processamento, divisão treino/validação/teste
e filtro de classes para a tarefa binária.
"""

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
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

    training_data, validation_data = split_train_val(training_data, seed)

    # Baixando dados de teste
    test_data = datasets.FashionMNIST(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=get_transform(),
    )
    return training_data, validation_data, test_data

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
