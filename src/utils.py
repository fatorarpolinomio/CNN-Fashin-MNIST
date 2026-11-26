"""
Trabalho de IA 2026 - Objetivo 2 (CNN com Fashion MNIST)

Integrantes:
    Lucas Neves Souza, Nro. USP: 15577627
    Willian Jefferson Farias, Nro. USP: 15642665

utils.py: semente aleatória, leitura e validação do config.yaml, criação das
pastas de saída e salvamento de arquivos JSON e CSV
"""

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml

# Caminhos a partir da raiz do projeto (não dependem de onde o script é executado)
RAIZ = Path(__file__).resolve().parent.parent
CONFIG_PATH = RAIZ / "config.yaml"
OUTPUTS_DIR = RAIZ / "outputs"

# Experimentos: cada um tem a sua pasta em outputs/<experimento>/
EXPERIMENTOS = ("multiclass", "binary", "hog")

# Otimizadores aceitos em config.yaml
OTIMIZADORES = ("adam", "sgd")

# Nomes dos arquivos de saída de cada experimento (os mesmos listados no README e na spec do trabalho)
HIPERPARAMETROS_JSON = "hiperparametros.json"      # hiperparâmetros da arquitetura e da inicialização
PESOS_INICIAIS_PT = "pesos_iniciais.pt"            # pesos antes do treinamento
PESOS_FINAIS_PT = "pesos_finais.pt"                # pesos após o treinamento
ERRO_POR_ITERACAO_CSV = "erro_por_iteracao.csv"    # erro a cada iteração/época
SAIDAS_TESTE_CSV = "saidas_teste.csv"              # saída da rede para cada dado de teste
MATRIZ_CONFUSAO_PNG = "matriz_confusao.png"        # matriz de confusão do teste


# Semente

def fix_seed(seed):
    """Fixa a semente de todos os geradores aleatórios usados no projeto.

    Deixa o cuDNN determinístico, o que pode deixar o treino em GPU um pouco mais lento.
    Chame uma vez no início de cada script, antes de criar o modelo (a semente
    define os pesos iniciais).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)  # não faz nada se não houver GPU
    torch.backends.cudnn.deterministic = True  # comente se quiser velocidade
    torch.backends.cudnn.benchmark = False



# Configuração

def _eh_inteiro(valor):
    return isinstance(valor, int) and not isinstance(valor, bool)

def _eh_numero(valor):
    return isinstance(valor, (int, float)) and not isinstance(valor, bool)

def validate_config(config):
    """Confere se o config.yaml tem o que o treino precisa; levanta ValueError com a causa

    Valida seed, dados.batch_size, o bloco treino (otimizador, lr e epocas),
    o bloco modelo (filtros_bloco1, filtros_bloco2 e neuronios_densa) e
    o bloco parada (patience, delta e verbose).
    O bloco `binario` é validado em dataset.py (classes_para_indices).
    """
    if not isinstance(config, dict):
        raise ValueError(
            "config.yaml deve conter chaves e valores (seed, dados, treino, modelo, parada, binario)"
        )

    if not _eh_inteiro(config.get("seed")):
        raise ValueError(f"seed deve ser um inteiro, mas recebeu {config.get('seed')!r}")

    for secao in ("dados", "treino", "modelo", "parada"):
        if not isinstance(config.get(secao), dict):
            raise ValueError(f"config.yaml não tem a seção '{secao}'")

    batch_size = config["dados"].get("batch_size")
    if not _eh_inteiro(batch_size) or batch_size < 1:
        raise ValueError(f"dados.batch_size deve ser um inteiro >= 1, mas recebeu {batch_size!r}")

    treino = config["treino"]

    if treino.get("otimizador") not in OTIMIZADORES:
        raise ValueError(
            f"treino.otimizador deve ser um de {list(OTIMIZADORES)}, mas recebeu {treino.get('otimizador')!r}"
        )

    lr = treino.get("lr")
    if not _eh_numero(lr) or lr <= 0:
        raise ValueError(f"treino.lr deve ser um número > 0 (ex.: 0.001), mas recebeu {lr!r}")

    epocas = treino.get("epocas")
    if not _eh_inteiro(epocas) or epocas < 1:
        raise ValueError(f"treino.epocas deve ser um inteiro >= 1, mas recebeu {epocas!r}")

    modelo = config["modelo"]
    for chave in ("filtros_bloco1", "filtros_bloco2", "neuronios_densa"):
        valor = modelo.get(chave)
        if not _eh_inteiro(valor) or valor < 1:
            raise ValueError(f"modelo.{chave} deve ser um inteiro >= 1, mas recebeu {valor!r}")

    parada = config["parada"]

    patience = parada.get("patience")
    if not _eh_inteiro(patience) or patience < 1:
        raise ValueError(f"parada.patience deve ser um inteiro >= 1, mas recebeu {patience!r}")

    delta = parada.get("delta")
    if not _eh_numero(delta) or delta < 0:
        raise ValueError(f"parada.delta deve ser um número >= 0, mas recebeu {delta!r}")

    verbose = parada.get("verbose")
    if not isinstance(verbose, bool):
        raise ValueError(f"parada.verbose deve ser um booleano (True ou False), mas recebeu {verbose!r}")

def load_config(path=CONFIG_PATH):
    """Lê o config.yaml, valida e devolve um dicionário."""
    with open(path, "r", encoding="utf-8") as arquivo:
        config = yaml.safe_load(arquivo)

    validate_config(config)
    return config



# Pastas e arquivos de saída

def create_output_dir(experimento, base=OUTPUTS_DIR):
    """Cria (se não existir) a pasta outputs/<experimento>/ e devolve o seu caminho """

    if experimento not in EXPERIMENTOS:
        raise ValueError(f"experimento deve ser um de {list(EXPERIMENTOS)}, mas recebeu {experimento!r}")

    pasta = Path(base) / experimento
    pasta.mkdir(parents=True, exist_ok=True)
    return pasta

def _json_padrao(objeto):
    # Converte para tipos do JSON o que o json.dump não sabe salvar sozinho
    if isinstance(objeto, Path):
        return str(objeto)
    if hasattr(objeto, "tolist"):  # numpy (escalar ou array) e tensor do PyTorch
        return objeto.tolist()
    raise TypeError(f"não sei salvar {type(objeto).__name__} em JSON")

def save_json(dados, caminho):
    """Salva `dados` (dicionário ou lista) em JSON legível e devolve o caminho. """

    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)

    with open(caminho, "w", encoding="utf-8") as arquivo:
        json.dump(dados, arquivo, indent=2, ensure_ascii=False, default=_json_padrao)
        arquivo.write("\n")

    return caminho

def save_csv(linhas, caminho, colunas=None):
    """Salva `linhas` em CSV (com cabeçalho) e devolve o caminho

    linhas: lista de dicionários (uma por linha do CSV) ou lista de listas/tuplas.
    colunas: nomes das colunas. Obrigatório para listas de listas; para
        dicionários, o padrão são as chaves da primeira linha.
    Cria as pastas do caminho se não existirem.

    Exemplo (erro_por_iteracao.csv):
        save_csv([{"epoca": 1, "erro_treino": 0.52}, ...], pasta / ERRO_POR_ITERACAO_CSV)
    """
    linhas = list(linhas)
    com_dicionarios = len(linhas) > 0 and isinstance(linhas[0], dict)

    if colunas is None:
        if not com_dicionarios:
            raise ValueError("informe `colunas` quando as linhas não forem dicionários (ou se não houver linhas)")
        colunas = list(linhas[0].keys())

    # Confere e monta todas as linhas antes de abrir o arquivo, para um erro não deixar CSV pela metade
    valores = []
    for linha in linhas:
        if com_dicionarios:
            # Chave faltando ou sobrando é erro: o CSV sairia com colunas trocadas
            if not isinstance(linha, dict) or set(linha.keys()) != set(colunas):
                raise ValueError(f"linha com colunas diferentes de {list(colunas)}: {linha}")
            valores.append([linha[c] for c in colunas])
        else:
            if len(linha) != len(colunas):
                raise ValueError(f"linha com {len(linha)} valores, mas há {len(colunas)} colunas: {linha}")
            valores.append(list(linha))

    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)

    with open(caminho, "w", encoding="utf-8", newline="") as arquivo:
        escritor = csv.writer(arquivo)
        escritor.writerow(colunas)
        escritor.writerows(valores)

    return caminho

def save_weights(model, caminho):
    """Salva os pesos (state_dict) do modelo e devolve o caminho."""
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), caminho)
    return caminho
