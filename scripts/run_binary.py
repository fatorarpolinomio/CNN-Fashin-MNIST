"""
Trabalho de IA 2026 - Objetivo 2 (CNN com Fashion MNIST)

Integrantes:
    Lucas Neves Souza, Nro. USP: 15577627
    Willian Jefferson Farias, Nro. USP: 15642665

run_binary.py: executa o experimento binário. O par de classes vem do config.yaml
ou, se informado, da linha de comando (ex.: --classe0 Trouser --classe1 Bag).
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from torch import nn
import utils, dataset, model, train, evaluate

def nome_do_par(config):
    """Nome da pasta do par, ex.: 'T-shirt-top_vs_Shirt' (cada par guarda as suas saídas)."""
    b = config["binario"]
    lado = lambda nomes: "+".join(n.replace("/", "-").replace(" ", "") for n in nomes)
    return f"{lado(b['classe_0'])}_vs_{lado(b['classe_1'])}"

# 1) Argumentos opcionais: sobrescrevem o par de classes do config.yaml
parser = argparse.ArgumentParser(description="Experimento binário (par de classes)")
parser.add_argument("--classe0", nargs="+", help="nome(s) da classe com rótulo 0")
parser.add_argument("--classe1", nargs="+", help="nome(s) da classe com rótulo 1")
args = parser.parse_args()

# Carregando e validando hiperparâmetros
config = utils.load_config()
if args.classe0:
    config["binario"]["classe_0"] = args.classe0
if args.classe1:
    config["binario"]["classe_1"] = args.classe1

# Fixando a seed de aleatoriedade
utils.fix_seed(config["seed"])

# Definindo local de saída
saida = utils.create_output_dir("binary") / nome_do_par(config)

# Buscando um dispositivo para realizar o treino
device = train.get_device_for_training()

print(f"Par de classes: {nome_do_par(config)}")

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
print(f"Treino: {len(train_dataset)} | Validação: {len(val_dataset)} | Teste: {len(test_dataset)}")

# Criando instância do nosso modelo
modelo = model.NeuralNetwork(config, num_classes=2).to(device)

# Definindo nossa loss function
loss_function = nn.CrossEntropyLoss()

# Construindo o otimizador
optimizer = train.build_optimizer(modelo, config)

# Pesos iniciais, antes de treinar
utils.save_weights(modelo, saida / utils.PESOS_INICIAIS_PT)

# Realizando treinamento do modelo e salvando lista de histórico
historico = train.model_training(
    modelo,
    train_dl,
    val_dl,
    test_dl,
    loss_function,
    optimizer,
    device,
    config
)

# Saídas finais
utils.save_weights(modelo, saida / utils.PESOS_FINAIS_PT)  # já com os melhores pesos restaurados
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

# Resumo para preencher a tabela de comparação dos pares (a época dos pesos restaurados)
melhor = min(historico, key=lambda h: h["loss_val"])
print(f"Melhor época: {melhor['epoca']} | perda de validação: {melhor['loss_val']:.4f} "
      f"| acurácia de validação: {melhor['acc_val']:.4f}")
print(f"Saídas salvas em: {saida}")

# Avaliação no teste com os pesos finais salvos (saidas_teste.csv e matriz_confusao.png)
evaluate.avaliar_experimento(saida, device)
