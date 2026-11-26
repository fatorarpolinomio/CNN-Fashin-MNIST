"""
Trabalho de IA 2026 - Objetivo 2 (CNN com Fashion MNIST)

Integrantes:
    Lucas Neves Souza, Nro. USP: 15577627
    Willian Jefferson Farias, Nro. USP: 15642665

evaluate.py: avaliação no conjunto de teste a partir dos pesos finais salvos
(saidas_teste.csv, acurácia, precisão, recall, F1 e matriz_confusao.png).

Uso (a partir da raiz do projeto):
    python src/evaluate.py outputs/multiclass
    python src/evaluate.py outputs/binary/Dress_vs_Coat
    python src/evaluate.py --todos          # avalia todo experimento que tenha pesos_finais.pt
"""

import argparse
import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import torch
from matplotlib.figure import Figure
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader

# A linha abaixo permite rodar `python src/evaluate.py` de qualquer pasta
sys.path.append(str(Path(__file__).resolve().parent))

import dataset
import train
import utils
from model import NeuralNetwork


# Nomes das classes

def nomes_das_classes(config, num_classes):
    """Devolve o nome de cada classe de saída, na ordem dos rótulos 0, 1, 2...

    Multiclasse (10 saídas): os 10 nomes do Fashion MNIST.
    Binária (2 saídas): um nome por lado do par; uma composição vira "A + B + C".
    """
    if num_classes == len(dataset.CLASSES):
        return list(dataset.CLASSES)

    if num_classes == 2:
        b = config["binario"]
        return [" + ".join(b["classe_0"]), " + ".join(b["classe_1"])]

    raise ValueError(f"num_classes deve ser 2 ou {len(dataset.CLASSES)}, mas recebeu {num_classes!r}")


# Carregamento do modelo e dos dados de teste

def carregar_modelo(pasta, device):
    """Reconstrói a CNN a partir de hiperparametros.json e carrega pesos_finais.pt """
    pasta = Path(pasta)
    caminho_hiper = pasta / utils.HIPERPARAMETROS_JSON
    caminho_pesos = pasta / utils.PESOS_FINAIS_PT

    for caminho in (caminho_hiper, caminho_pesos):
        if not caminho.is_file():
            raise FileNotFoundError(f"não encontrei {caminho}. Rode o experimento antes de avaliar.")

    with open(caminho_hiper, "r", encoding="utf-8") as arquivo:
        config = json.load(arquivo)

    if "num_classes" not in config:
        raise ValueError(f"{caminho_hiper} não tem a chave 'num_classes'")

    modelo = NeuralNetwork(config, num_classes=config["num_classes"]).to(device)

    # map_location leva os pesos para o dispositivo atual (um .pt salvo na GPU abre na CPU);
    # weights_only=True carrega só tensores, sem executar código arbitrário do arquivo
    pesos = torch.load(caminho_pesos, map_location=device, weights_only=True)
    modelo.load_state_dict(pesos)

    modelo.eval()  # desliga comportamentos de treino (a rede atual não tem dropout/batchnorm, mas é o correto)
    return modelo, config

def indices_originais(test_dataset):
    """Índice de cada exemplo no conjunto de teste original do Fashion MNIST (0 a 9999).

    No multiclasse é a própria posição. No binário o dataset é um filtro do original,
    então pegamos os índices guardados no Subset: assim cada linha do CSV
    aponta para a imagem certa mesmo depois do filtro.
    """
    if isinstance(test_dataset, dataset.BinaryDataset):
        return [int(i) for i in test_dataset.subset.indices]
    return list(range(len(test_dataset)))

def carregar_teste(config, num_classes):
    """Monta o DataLoader do teste do experimento. Devolve (dataloader, índices originais).

    O teste é o conjunto oficial de 10.000 imagens, que não passa pelo random_split,
    então não depende da seed. Na tarefa binária aplicamos o mesmo filtro do treino
    (par de classes lido do hiperparametros.json) e shuffle=False mantém a ordem fixa.
    """
    _, test_data = dataset.load_fashionMNIST(config["seed"])

    if num_classes == 2:
        test_data = dataset.filter_binary_classes(
            test_data, config["binario"]["classe_0"], config["binario"]["classe_1"]
        )

    test_dl = DataLoader(test_data, batch_size=config["dados"]["batch_size"], shuffle=False)
    return test_dl, indices_originais(test_data)


# Predição e métricas

def prever(modelo, dataloader, device):
    """Passa todo o dataloader pela rede e devolve (logits, rótulos reais) como arrays NumPy.

    logits tem formato (N, num_classes): é a saída crua da última camada densa,
    antes do softmax. A classe prevista é a de maior logit (argmax).
    """
    modelo.eval()
    todos_logits, todos_rotulos = [], []

    with torch.no_grad():  # sem gradientes: só queremos a saída, não treinar
        for X, y in dataloader:
            logits = modelo(X.to(device))
            todos_logits.append(logits.cpu())
            todos_rotulos.append(y)

    return torch.cat(todos_logits).numpy(), torch.cat(todos_rotulos).numpy()

def calcular_metricas(rotulos, previsoes, nomes):
    """Calcula acurácia, precisão, recall e F1 (por classe e média macro) e a matriz de confusão.

    - Acurácia: fração de acertos.
    - Precisão da classe c: dos exemplos previstos como c, quantos eram c.
    - Recall da classe c: dos exemplos que são c, quantos a rede achou.
    - F1: média harmônica entre precisão e recall.
    - Macro: média simples das classes (todas pesam igual; no teste são 1000 por classe no multiclasse).
    zero_division=0 evita aviso/erro se alguma classe nunca for prevista (precisão indefinida vira 0).
    """
    labels = list(range(len(nomes)))  # fixa todas as classes, mesmo que alguma não apareça

    precisao, recall, f1, suporte = precision_recall_fscore_support(
        rotulos, previsoes, labels=labels, zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        rotulos, previsoes, labels=labels, average="macro", zero_division=0
    )

    por_classe = [
        {
            "classe": nomes[c],
            "precisao": float(precisao[c]),
            "recall": float(recall[c]),
            "f1": float(f1[c]),
            "suporte": int(suporte[c]),
        }
        for c in labels
    ]

    return {
        "nomes": list(nomes),
        "total": int(len(rotulos)),
        "acertos": int((np.asarray(rotulos) == np.asarray(previsoes)).sum()),
        "acuracia": float(accuracy_score(rotulos, previsoes)),
        "por_classe": por_classe,
        "macro": {"precisao": float(macro_p), "recall": float(macro_r), "f1": float(macro_f1)},
        # linha = classe real, coluna = classe prevista
        "matriz": confusion_matrix(rotulos, previsoes, labels=labels),
    }

def maior_confusao(matriz, nomes):
    """Devolve (classe real, classe prevista, quantidade) do erro mais frequente, ou None se não houve erro."""
    erros = np.array(matriz, copy=True)
    np.fill_diagonal(erros, 0)  # a diagonal são os acertos; só interessam os erros

    if erros.max() == 0:
        return None

    real, prevista = np.unravel_index(erros.argmax(), erros.shape)
    return nomes[real], nomes[prevista], int(erros[real, prevista])


# Arquivos de saída

def _softmax(logits):
    # Subtrair o maior logit de cada linha não muda o resultado e evita overflow no exp
    deslocado = logits - logits.max(axis=1, keepdims=True)
    exponenciais = np.exp(deslocado)
    return exponenciais / exponenciais.sum(axis=1, keepdims=True)

def montar_saidas_teste(indices, logits, rotulos, nomes):
    """Monta as linhas do saidas_teste.csv: uma por exemplo de teste (lista de dicionários, como no save_csv).

    Colunas: indice (no teste original), rotulo_real, rotulo_previsto, classe_real,
    classe_prevista, acerto (1/0), logit_0..logit_k e prob_0..prob_k (softmax dos logits).
    """
    logits = np.asarray(logits, dtype=np.float64)
    probabilidades = _softmax(logits)
    previsoes = logits.argmax(axis=1)
    num_classes = len(nomes)

    linhas = []
    for k in range(len(rotulos)):
        real, prevista = int(rotulos[k]), int(previsoes[k])

        linha = {
            "indice": int(indices[k]),
            "rotulo_real": real,
            "rotulo_previsto": prevista,
            "classe_real": nomes[real],
            "classe_prevista": nomes[prevista],
            "acerto": int(real == prevista),
        }
        for c in range(num_classes):
            linha[f"logit_{c}"] = round(float(logits[k, c]), 6)
        for c in range(num_classes):
            linha[f"prob_{c}"] = round(float(probabilidades[k, c]), 6)

        linhas.append(linha)

    return linhas

def salvar_matriz_confusao(matriz, nomes, caminho, titulo="Matriz de confusão (teste)"):
    """Desenha a matriz de confusão (contagens em cada célula) e salva em PNG. Devolve o caminho.

    Usa a classe Figure do matplotlib direto, sem pyplot.
    """
    matriz = np.asarray(matriz)
    n = len(nomes)

    lado = max(5.0, 0.8 * n + 1.5)   # 2 classes: 5x5 polegadas; 10 classes: ~9,5
    fig = Figure(figsize=(lado + 1, lado), layout="constrained")
    ax = fig.subplots()

    imagem = ax.imshow(matriz, cmap="Blues")
    fig.colorbar(imagem, ax=ax, label="Nº de exemplos")

    # Nomes longos (composições de classes) quebram em várias linhas
    rotulos_eixo = [textwrap.fill(nome, 14) for nome in nomes]
    inclinar = n > 2
    ax.set_xticks(range(n), labels=rotulos_eixo,
                  rotation=45 if inclinar else 0,
                  ha="right" if inclinar else "center",
                  rotation_mode="anchor")
    ax.set_yticks(range(n), labels=rotulos_eixo)
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe real")
    ax.set_title(titulo)

    # Número em cada célula; branco nas células escuras para continuar legível
    limite = matriz.max() / 2
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(int(matriz[i, j])), ha="center", va="center",
                    fontsize=14 if n <= 2 else 9,
                    color="white" if matriz[i, j] > limite else "black")

    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho, dpi=150)
    return caminho


# Resumo no console

def imprimir_resumo(pasta, device, resultado):
    """Imprime o resumo da avaliação: acurácia, tabela por classe, média macro e maior confusão."""
    nomes = resultado["nomes"]
    largura = min(max(len(n) for n in nomes + ["Média macro"]), 32)

    def curto(nome):
        return nome if len(nome) <= largura else nome[: largura - 1] + "…"

    linha = "=" * (largura + 44)
    print(f"\n{linha}")
    print(f"AVALIAÇÃO NO TESTE: {pasta}")
    print(linha)
    print(f"Dispositivo: {device} | Exemplos de teste: {resultado['total']}")
    print(f"Acurácia: {resultado['acuracia']:.4f} ({100 * resultado['acuracia']:.2f}%) "
          f"- {resultado['acertos']} de {resultado['total']} acertos\n")

    print(f"{'Classe':<{largura}}  {'Precisão':>9}  {'Recall':>9}  {'F1':>9}  {'Suporte':>8}")
    print("-" * (largura + 44))
    for c in resultado["por_classe"]:
        print(f"{curto(c['classe']):<{largura}}  {c['precisao']:>9.4f}  {c['recall']:>9.4f}  "
              f"{c['f1']:>9.4f}  {c['suporte']:>8d}")
    print("-" * (largura + 44))
    m = resultado["macro"]
    print(f"{'Média macro':<{largura}}  {m['precisao']:>9.4f}  {m['recall']:>9.4f}  "
          f"{m['f1']:>9.4f}  {resultado['total']:>8d}")

    confusao = maior_confusao(resultado["matriz"], nomes)
    if confusao is None:
        print("\nNenhum erro de classificação no teste.")
    else:
        real, prevista, quantidade = confusao
        print(f"\nErro mais frequente: '{real}' previsto como '{prevista}' ({quantidade} exemplos)")


# Fluxo completo de um experimento

def avaliar_experimento(pasta, device=None):
    """Avalia o experimento da pasta (ex.: outputs/multiclass) no teste e salva as saídas nela.

    Carrega hiperparametros.json + pesos_finais.pt, gera saidas_teste.csv e
    matriz_confusao.png, imprime o resumo e devolve o dicionário de métricas.
    """
    pasta = Path(pasta)
    if device is None:
        device = train.get_device_for_training()

    modelo, config = carregar_modelo(pasta, device)
    num_classes = config["num_classes"]
    nomes = nomes_das_classes(config, num_classes)

    test_dl, indices = carregar_teste(config, num_classes)
    logits, rotulos = prever(modelo, test_dl, device)
    previsoes = logits.argmax(axis=1)

    resultado = calcular_metricas(rotulos, previsoes, nomes)

    utils.save_csv(montar_saidas_teste(indices, logits, rotulos, nomes), pasta / utils.SAIDAS_TESTE_CSV)

    titulo = "Matriz de confusão (teste): " + ("multiclasse" if num_classes > 2 else "binária")
    salvar_matriz_confusao(resultado["matriz"], nomes, pasta / utils.MATRIZ_CONFUSAO_PNG, titulo)

    imprimir_resumo(pasta, device, resultado)
    print(f"\nSaídas salvas em: {pasta} ({utils.SAIDAS_TESTE_CSV}, {utils.MATRIZ_CONFUSAO_PNG})")

    return resultado


def main():
    parser = argparse.ArgumentParser(description="Avalia no teste um experimento já treinado")
    parser.add_argument("pastas", nargs="*", help="pasta(s) do experimento, ex.: outputs/binary/Dress_vs_Coat")
    parser.add_argument("--todos", action="store_true",
                        help=f"avalia todas as pastas de outputs/ que tenham {utils.PESOS_FINAIS_PT}")
    args = parser.parse_args()

    pastas = [Path(p) for p in args.pastas]
    if args.todos:
        pastas += sorted(p.parent for p in utils.OUTPUTS_DIR.rglob(utils.PESOS_FINAIS_PT))
    if not pastas:
        parser.error("informe uma pasta de experimento ou use --todos")

    device = train.get_device_for_training()
    for pasta in pastas:
        avaliar_experimento(pasta, device)


if __name__ == "__main__":
    main()