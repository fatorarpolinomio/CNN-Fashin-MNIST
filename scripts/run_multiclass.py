"""
Trabalho de IA 2026 - Objetivo 2 (CNN com Fashion MNIST)

Integrantes:
    Lucas Neves Souza, Nro. USP: 15577627
    Willian Jefferson Farias, Nro. USP: 15642665

run_multiclass.py: executa o experimento multiclasse.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import torch
from torch import nn
import utils, dataset, model, train

# 1) Configuração, seed (antes de criar o modelo) e pasta de saída
config = utils.load_config()
utils.fix_seed(config["seed"])
device = train.get_device_for_training()
saida = utils.create_output_dir("multiclass")

# 2) Dados
train_data, test_data = dataset.load_fashionMNIST(config["seed"])
train_data, val_data = dataset.split_train_val(train_data, config["seed"])
train_dl, val_dl, test_dl = dataset.create_dataloaders(config, train_data, val_data, test_data)

# 3) Modelo no dispositivo, função de perda e otimizador
modelo = model.NeuralNetwork(config, num_classes=10).to(device)
loss_function = nn.CrossEntropyLoss()

lr = config["treino"]["lr"]
if config["treino"]["otimizador"] == "adam":
    optimizer = torch.optim.Adam(modelo.parameters(), lr=lr)
else:
    optimizer = torch.optim.SGD(modelo.parameters(), lr=lr)

# 4) Pesos iniciais, antes de treinar
utils.save_weights(modelo, saida / utils.PESOS_INICIAIS_PT)

# 5) Treino
historico = train.model_training(
    modelo, train_dl, val_dl, test_dl,
    loss_function, optimizer, device, config, config["treino"]["epocas"]
)

# 6) Saídas finais
utils.save_weights(modelo, saida / utils.PESOS_FINAIS_PT)  # já com os melhores pesos restaurados
utils.save_csv(historico, saida / utils.ERRO_POR_ITERACAO_CSV)
utils.save_json(
    {
        **config,
        "num_classes": 10,
        "dispositivo": str(device),
        "inicializacao": "padrão do PyTorch para Conv2d e Linear",
    },
    saida / utils.HIPERPARAMETROS_JSON,
)
