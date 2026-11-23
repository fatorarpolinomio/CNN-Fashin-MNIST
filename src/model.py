"""
Trabalho de IA 2026 - Objetivo 2 (CNN com Fashion MNIST)

Integrantes:
    Lucas Neves Souza, Nro. USP: 15577627
    Willian Jefferson Farias, Nro. USP: 15642665

model.py: arquitetura da CNN (camadas de convolução e pooling, camadas densas e camada de saída).
"""

import torch
from torch import nn

# Definindo nossa rede neural e inicializando as suas camadas em __init__
# Um bloco é um conjunto de camadas aplicadas em sequência
class NeuralNetwork(nn.Module):
    def __init__(self, config, num_classes=10): # <- Parametrizável (padrão 10, mas aceita 2)
        super().__init__()

        # Buscando os hiperparâmetros definidos no config.yaml
        cfg = config["modelo"]
        f1 = cfg["filtros_bloco1"] # Quantidade de filtros do primeiro bloco
        f2 = cfg["filtros_bloco2"] # Quantidade de filtros do segundo bloco
        n_densa = cfg["neuronios_densa"] # Quantidade de neurônios da camada densa

        self.bloco1 = nn.Sequential(

            # Formato de entrada: 1 x 28 x 28
            # Aplicando convolução 2D sobre as entradas. Saída: f1 x 28 x 28
            nn.Conv2d(
                in_channels=1, # As imagens de entrada do Fashion-MNIST estão em escala de cinza (1 único canal de intensidade)
                out_channels=f1, # quantos kernels queremos utilizar
                kernel_size=(3, 3), # tamanho do kernel (3x3)
                stride=1, # o kernel se desloca de um em um pixel
                padding=1 # borda de 1 pixel com zeros ao redor da imagem
            ),

            # Função de ativação
            nn.ReLU(),

            # MaxPool2D, reduzindo o tamanho dos tensores de saída. Saída: f1 x 13 x 13
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.bloco2 = nn.Sequential(

            # Formato de entrada: f1 x 13 x 13
            # Aplicando convolução 2D sobre as saídas do bloco anterior. Saída: f2 x 13 x 13
            nn.Conv2d(
                in_channels=f1, # número de canais de saída do bloco anterior,
                out_channels=f2, # número de kernels desta convolução
                kernel_size=(3, 3), # tamanho do kernel (3x3)
                stride=1,
                padding=1
            ),

            # Função de ativação
            nn.ReLU(),

            # MaxPool2D, reduzindo o tamanho dos tensores de saída. Saída: f2 x 6 x 6
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.classificador = nn.Sequential(

            # Formato de entrada: f2 x 6 x 6
            # Camada para achatar os tensores em uma única dimensão. Saída: f2 x 6 x 6 = número inteiro
            nn.Flatten(),

            # Camada densa para calcular logits
            # 1ª camada densa (oculta). Saída: n_densa
            nn.Linear(in_features=f2 * 6 * 6, out_features=n_densa, bias=True), # 6 x 6 depende do kernel/padding/pooling dos blocos (ver comentários acima)

            # Função de ativação
            nn.ReLU(),

            # 2ª camada densa (saída parametrizável). Saída: num_classes
            nn.Linear(in_features=n_densa, out_features=num_classes, bias=True)
        )


    def forward(self, x):
        saida_bloco1 = self.bloco1(x)
        saida_bloco2 = self.bloco2(saida_bloco1)
        saida = self.classificador(saida_bloco2)
        return saida
