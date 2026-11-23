"""
Trabalho de IA 2026 - Objetivo 2 (CNN com Fashion MNIST)

Integrantes:
    Lucas Neves Souza, Nro. USP: 15577627
    Willian Jefferson Farias, Nro. USP: 15642665

tests/test_utils.py: testes de src/utils.py (semente, leitura do config.yaml,
pastas de saída e salvamento de JSON e CSV). Pode ser executado, a partir da
raiz do projeto, de duas formas:

    python tests/test_utils.py      # roda tudo e imprime um resumo por teste
    pytest tests/                   # se o pytest estiver instalado

Os testes de escrita usam pastas temporárias: não criam nada em outputs/.
"""

import copy
import csv
import json
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

# Permite rodar `python tests/test_utils.py` sem instalar o projeto como pacote
RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.utils import (  # noqa: E402
    ERRO_POR_ITERACAO_CSV,
    EXPERIMENTOS,
    HIPERPARAMETROS_JSON,
    MATRIZ_CONFUSAO_PNG,
    PESOS_FINAIS_PT,
    PESOS_INICIAIS_PT,
    SAIDAS_TESTE_CSV,
    create_output_dir,
    fix_seed,
    load_config,
    save_csv,
    save_json,
    validate_config,
)



# Utilitários

def deve_falhar(excecao, funcao, *args, **kwargs):
    """Garante que funcao(*args, **kwargs) levanta `excecao` (equivalente simples ao pytest.raises)."""
    try:
        funcao(*args, **kwargs)
    except excecao:
        return
    raise AssertionError(f"{funcao.__name__} deveria levantar {excecao.__name__}")


def amostra_aleatoria():
    """Um número de cada gerador usado no projeto."""
    return (random.random(), float(np.random.rand()), float(torch.rand(1)))


def config_valida():
    """Cópia do config.yaml real, para os testes alterarem sem afetar o arquivo."""
    return copy.deepcopy(load_config())



# Testes: semente

def test_fix_seed_reprodutivel():
    fix_seed(123)
    primeira = amostra_aleatoria()
    fix_seed(123)
    segunda = amostra_aleatoria()
    assert primeira == segunda


def test_fix_seed_sementes_diferentes_dao_resultados_diferentes():
    fix_seed(1)
    a = amostra_aleatoria()
    fix_seed(2)
    b = amostra_aleatoria()
    assert a != b



# Testes: configuração

def test_load_config_do_projeto():
    config = load_config()
    assert config["seed"] == 42
    assert config["dados"]["batch_size"] >= 1
    assert config["treino"]["otimizador"] in ("adam", "sgd")
    assert config["treino"]["lr"] > 0
    assert config["treino"]["epocas"] >= 1
    assert config["binario"]["classe_0"] and config["binario"]["classe_1"]


def test_load_config_de_outro_arquivo_e_arquivo_inexistente():
    with tempfile.TemporaryDirectory() as tmp:
        caminho = Path(tmp) / "outro.yaml"
        caminho.write_text(
            "seed: 7\ndados:\n  batch_size: 32\ntreino:\n  otimizador: sgd\n  lr: 0.01\n  epocas: 3\n",
            encoding="utf-8",
        )
        config = load_config(caminho)
        assert config["seed"] == 7 and config["treino"]["otimizador"] == "sgd"

        deve_falhar(FileNotFoundError, load_config, Path(tmp) / "nao_existe.yaml")


def test_validate_config_rejeita_valores_invalidos():
    # Cada item: (função que estraga a config, descrição do erro)
    estragos = [
        (lambda c: c.pop("seed"), "sem seed"),
        (lambda c: c.update(seed=True), "seed booleana"),
        (lambda c: c.update(seed=1.5), "seed não inteira"),
        (lambda c: c.pop("dados"), "sem seção dados"),
        (lambda c: c.pop("treino"), "sem seção treino"),
        (lambda c: c["dados"].update(batch_size=0), "batch_size 0"),
        (lambda c: c["dados"].update(batch_size="64"), "batch_size texto"),
        (lambda c: c["treino"].update(otimizador="rmsprop"), "otimizador desconhecido"),
        (lambda c: c["treino"].update(otimizador="Adam"), "otimizador fora de minúsculas"),
        (lambda c: c["treino"].update(lr=0), "lr zero"),
        (lambda c: c["treino"].update(lr=-0.1), "lr negativa"),
        (lambda c: c["treino"].update(lr="1e-3"), "lr texto (o YAML lê 1e-3 como texto)"),
        (lambda c: c["treino"].update(epocas=0), "epocas 0"),
        (lambda c: c["treino"].update(epocas=2.5), "epocas não inteira"),
    ]
    for estragar, descricao in estragos:
        config = config_valida()
        estragar(config)
        try:
            validate_config(config)
        except ValueError:
            continue
        raise AssertionError(f"validate_config deveria rejeitar: {descricao}")

    deve_falhar(ValueError, validate_config, None)       # arquivo YAML vazio
    deve_falhar(ValueError, validate_config, ["a", "b"])  # YAML que é uma lista


def test_validate_config_aceita_lr_inteira_e_float():
    config = config_valida()
    config["treino"]["lr"] = 1
    validate_config(config)
    config["treino"]["lr"] = 0.001
    validate_config(config)



# Testes: pastas de saída

def test_create_output_dir():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "outputs"  # a própria base ainda não existe
        for experimento in EXPERIMENTOS:
            pasta = create_output_dir(experimento, base=base)
            assert pasta == base / experimento and pasta.is_dir()

        # Chamar de novo não pode dar erro nem apagar o que já existe
        (base / "binary" / "arquivo.txt").write_text("x")
        create_output_dir("binary", base=base)
        assert (base / "binary" / "arquivo.txt").exists()

        deve_falhar(ValueError, create_output_dir, "multiclasse", base=base)
        assert not (base / "multiclasse").exists()


def test_experimentos_esperados():
    assert set(EXPERIMENTOS) == {"multiclass", "binary", "hog"}



# Testes: JSON
 
def test_save_json_ida_e_volta():
    with tempfile.TemporaryDirectory() as tmp:
        dados = {"nome": "classificação", "lr": 0.001, "camadas": [32, 64], "otimizador": "adam"}
        caminho = save_json(dados, Path(tmp) / "pasta" / "nova" / HIPERPARAMETROS_JSON)

        assert caminho.exists()  # criou as pastas que faltavam
        assert json.loads(caminho.read_text(encoding="utf-8")) == dados
        assert "classificação" in caminho.read_text(encoding="utf-8")  # acento mantido, sem \u00e7


def test_save_json_tipos_extras():
    with tempfile.TemporaryDirectory() as tmp:
        dados = {
            "pasta": Path("outputs") / "binary",
            "escalar_numpy": np.float32(0.5),
            "inteiro_numpy": np.int64(3),
            "vetor_numpy": np.array([1, 2, 3]),
            "tensor": torch.tensor([0.25, 0.75]),
        }
        caminho = save_json(dados, Path(tmp) / "extras.json")
        lido = json.loads(caminho.read_text(encoding="utf-8"))

        assert lido["pasta"] == str(Path("outputs") / "binary")
        assert lido["escalar_numpy"] == 0.5 and lido["inteiro_numpy"] == 3
        assert lido["vetor_numpy"] == [1, 2, 3] and lido["tensor"] == [0.25, 0.75]

        deve_falhar(TypeError, save_json, {"objeto": object()}, Path(tmp) / "invalido.json")



# Testes: CSV

def ler_csv(caminho):
    with open(caminho, "r", encoding="utf-8", newline="") as arquivo:
        return list(csv.reader(arquivo))


def test_save_csv_com_dicionarios():
    with tempfile.TemporaryDirectory() as tmp:
        linhas = [
            {"epoca": 1, "erro_treino": 0.5, "erro_validacao": 0.6},
            {"epoca": 2, "erro_treino": 0.25, "erro_validacao": 0.3},
        ]
        caminho = save_csv(linhas, Path(tmp) / "novo" / ERRO_POR_ITERACAO_CSV)

        assert ler_csv(caminho) == [
            ["epoca", "erro_treino", "erro_validacao"],
            ["1", "0.5", "0.6"],
            ["2", "0.25", "0.3"],
        ]


def test_save_csv_com_listas_e_colunas():
    with tempfile.TemporaryDirectory() as tmp:
        linhas = [(0, 1, 1), (1, 0, 0)]
        caminho = save_csv(linhas, Path(tmp) / SAIDAS_TESTE_CSV, colunas=["indice", "rotulo", "predicao"])

        assert ler_csv(caminho) == [["indice", "rotulo", "predicao"], ["0", "1", "1"], ["1", "0", "0"]]


def test_save_csv_so_cabecalho_quando_nao_ha_linhas():
    with tempfile.TemporaryDirectory() as tmp:
        caminho = save_csv([], Path(tmp) / "vazio.csv", colunas=["a", "b"])
        assert ler_csv(caminho) == [["a", "b"]]


def test_save_csv_entradas_invalidas_nao_criam_arquivo():
    with tempfile.TemporaryDirectory() as tmp:
        destino = Path(tmp) / "invalido.csv"

        deve_falhar(ValueError, save_csv, [[1, 2]], destino)                      # lista sem `colunas`
        deve_falhar(ValueError, save_csv, [], destino)                            # sem linhas e sem `colunas`
        deve_falhar(ValueError, save_csv, [[1, 2, 3]], destino, colunas=["a", "b"])   # tamanho errado
        deve_falhar(ValueError, save_csv, [{"a": 1}, {"b": 2}], destino)          # chaves diferentes
        deve_falhar(ValueError, save_csv, [{"a": 1, "b": 2}], destino, colunas=["a"])  # chave sobrando

        assert not destino.exists()

