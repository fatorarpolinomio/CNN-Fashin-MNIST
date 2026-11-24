"""
Trabalho de IA 2026 - Objetivo 2 (CNN com Fashion MNIST)

Integrantes:
    Lucas Neves Souza, Nro. USP: 15577627
    Willian Jefferson Farias, Nro. USP: 15642665

run_binary.py: executa o experimento binário (par de classes definido em config.yaml).
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import torch
from torch import nn
import utils, dataset, model, train

# Carregando e validando hiperparâmetros
config = utils.load_config()
utils.validate_config(config)

# Fixando a seed de aleatoriedade
utils.fix_seed(config["seed"])

# Definindo local de saída
saida = utils.create_output_dir("binary")

# Buscando um dispositivo para realizar o treino
device = train.get_device_for_training()

# Carregando dataset
train_dataset, test_dataset = dataset.load_fashionMNIST(config["seed"])
train_dataset, val_dataset = dataset.split_train_val(train_dataset, config["seed"])

# Criando datasets binários a partir dos originais
train_dataset, val_dataset, test_dataset = dataset.create_binary_datasets(
    config,
    train_dataset,
    val_dataset,
    test_dataset
)

# Criando dataloaders a partir dos datasets
train_dl, val_dl, test_dl = dataset.create_dataloaders(
    config,
    train_dataset,
    val_dataset,
    test_dataset
)

# Criando instância do nosso modelo
model = model.NeuralNetwork(config, num_classes=2).to(device)

# Definindo nossa loss function
loss_function = nn.CrossEntropyLoss()

# Construindo o otimizador
lr = config["treino"]["lr"]
if config["treino"]["otimizador"] == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
else:
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# Pesos iniciais, antes de treinar
utils.save_weights(model, saida / utils.PESOS_INICIAIS_PT)

# Realizando treinamento do modelo e salvando lista de histórico
historico = train.model_training(
    model,
    train_dl,
    val_dl,
    test_dl,
    loss_function,
    optimizer,
    device,
    config
)

# Saídas finais
utils.save_weights(model, saida / utils.PESOS_FINAIS_PT)  # já com os melhores pesos restaurados
utils.save_csv(historico, saida / utils.ERRO_POR_ITERACAO_CSV)
utils.save_json(
    {
        **config,
        "num_classes":2,
        "dispositivo": str(device),
        "inicializacao": "padrão do PyTorch para Conv2d e Linear",
    },
    saida / utils.HIPERPARAMETROS_JSON,
)
