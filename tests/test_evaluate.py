"""
Trabalho de IA 2026 - Objetivo 2 (CNN com Fashion MNIST)

Integrantes:
    Lucas Neves Souza, Nro. USP: 15577627
    Willian Jefferson Farias, Nro. USP: 15642665

test_evaluate.py: testes do src/evaluate.py com dados sintéticos (não precisam do Fashion MNIST).
Rodar a partir da raiz: python tests/test_evaluate.py  ou  pytest tests/test_evaluate.py
"""

import contextlib
import csv
import io
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import evaluate
import utils

PNG_ASSINATURA = b"\x89PNG\r\n\x1a\n"   # os 8 primeiros bytes de todo arquivo PNG

# Caso pequeno com valores calculados à mão (3 classes, 8 exemplos):
#   matriz [[2,1,0],[1,2,1],[0,0,1]] (linha = real, coluna = prevista)
ROTULOS = np.array([0, 0, 0, 1, 1, 1, 1, 2])
PREVISOES = np.array([0, 0, 1, 1, 1, 0, 2, 2])
NOMES = ["A", "B", "C"]

def aprox(a, b, tol=1e-9):
    return abs(a - b) < tol


# Métricas

def test_metricas_valores_calculados_a_mao():
    r = evaluate.calcular_metricas(ROTULOS, PREVISOES, NOMES)

    assert r["total"] == 8 and r["acertos"] == 5
    assert aprox(r["acuracia"], 5 / 8)
    assert r["matriz"].tolist() == [[2, 1, 0], [1, 2, 1], [0, 0, 1]]

    precisao = [c["precisao"] for c in r["por_classe"]]
    recall = [c["recall"] for c in r["por_classe"]]
    f1 = [c["f1"] for c in r["por_classe"]]
    assert all(aprox(a, b) for a, b in zip(precisao, [2 / 3, 2 / 3, 1 / 2]))
    assert all(aprox(a, b) for a, b in zip(recall, [2 / 3, 1 / 2, 1.0]))
    assert all(aprox(a, b) for a, b in zip(f1, [2 / 3, 4 / 7, 2 / 3]))
    assert [c["suporte"] for c in r["por_classe"]] == [3, 4, 1]

    assert aprox(r["macro"]["precisao"], (2 / 3 + 2 / 3 + 1 / 2) / 3)
    assert aprox(r["macro"]["recall"], (2 / 3 + 1 / 2 + 1.0) / 3)
    assert aprox(r["macro"]["f1"], (2 / 3 + 4 / 7 + 2 / 3) / 3)

def test_metricas_classe_nunca_prevista_nao_quebra():
    # A classe 2 existe nos dados reais, mas a rede nunca a prevê: precisão indefinida vira 0
    with warnings.catch_warnings():
        warnings.simplefilter("error")   # qualquer aviso do scikit-learn vira erro e reprova o teste
        r = evaluate.calcular_metricas(np.array([0, 1, 2]), np.array([0, 1, 1]), NOMES)
    assert r["por_classe"][2]["precisao"] == 0.0
    assert r["por_classe"][2]["recall"] == 0.0
    assert r["matriz"].shape == (3, 3)

def test_metricas_acerto_total():
    r = evaluate.calcular_metricas(ROTULOS, ROTULOS, NOMES)
    assert r["acuracia"] == 1.0
    assert r["macro"]["f1"] == 1.0
    assert evaluate.maior_confusao(r["matriz"], NOMES) is None

def test_maior_confusao():
    r = evaluate.calcular_metricas(ROTULOS, PREVISOES, NOMES)
    # Empate em 1 entre vários erros; o que importa é devolver um erro real (fora da diagonal)
    real, prevista, quantidade = evaluate.maior_confusao(r["matriz"], NOMES)
    assert real != prevista and quantidade == 1

    matriz = np.array([[5, 0], [7, 3]])
    assert evaluate.maior_confusao(matriz, ["x", "y"]) == ("y", "x", 7)


# Nomes das classes

def test_nomes_das_classes():
    assert len(evaluate.nomes_das_classes({}, 10)) == 10
    config = {"binario": {"classe_0": ["Dress"], "classe_1": ["Coat", "Shirt"]}}
    assert evaluate.nomes_das_classes(config, 2) == ["Dress", "Coat + Shirt"]
    try:
        evaluate.nomes_das_classes(config, 5)
    except ValueError:
        return
    raise AssertionError("num_classes inválido deveria levantar ValueError")


# saidas_teste.csv

def test_saidas_teste_colunas_e_valores():
    logits = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, 1.5]])
    rotulos = np.array([0, 0, 1])
    linhas = evaluate.montar_saidas_teste([10, 25, 40], logits, rotulos, ["X", "Y"])

    assert len(linhas) == 3
    assert list(linhas[0].keys()) == [
        "indice", "rotulo_real", "rotulo_previsto", "classe_real", "classe_prevista",
        "acerto", "logit_0", "logit_1", "prob_0", "prob_1",
    ]
    # linha 0: acerto; linha 1: a rede previu 1 mas o real é 0; linha 2: previu 1 e acertou
    assert [l["rotulo_previsto"] for l in linhas] == [0, 1, 1]
    assert [l["acerto"] for l in linhas] == [1, 0, 1]
    assert [l["indice"] for l in linhas] == [10, 25, 40]
    assert linhas[1]["classe_real"] == "X" and linhas[1]["classe_prevista"] == "Y"
    for l in linhas:
        assert abs(l["prob_0"] + l["prob_1"] - 1.0) < 1e-5
    assert linhas[0]["prob_0"] > 0.5 > linhas[0]["prob_1"]

def test_saidas_teste_softmax_estavel_com_logits_grandes():
    # Sem subtrair o máximo, exp(1000) estouraria e geraria nan
    linhas = evaluate.montar_saidas_teste([0], np.array([[1000.0, 0.0, -1000.0]]), np.array([0]), NOMES)
    assert linhas[0]["prob_0"] == 1.0 and linhas[0]["prob_2"] == 0.0

def test_saidas_teste_salva_csv_com_uma_linha_por_exemplo():
    logits = np.random.RandomState(0).randn(50, 10)
    rotulos = np.random.RandomState(1).randint(0, 10, size=50)
    nomes = [f"c{i}" for i in range(10)]
    linhas = evaluate.montar_saidas_teste(list(range(50)), logits, rotulos, nomes)

    with tempfile.TemporaryDirectory() as tmp:
        caminho = utils.save_csv(linhas, Path(tmp) / utils.SAIDAS_TESTE_CSV)
        with open(caminho, newline="", encoding="utf-8") as f:
            lidas = list(csv.DictReader(f))

    assert len(lidas) == 50
    assert len(lidas[0]) == 6 + 10 + 10   # 6 colunas fixas + 10 logits + 10 probabilidades
    assert int(lidas[7]["rotulo_real"]) == int(rotulos[7])


# matriz_confusao.png

def test_matriz_confusao_gera_png():
    casos = {
        "binaria": (np.array([[90, 10], [20, 80]]), ["Dress", "Coat"]),
        "composicao": (np.array([[400, 5], [3, 300]]),
                       ["T-shirt/top + Pullover + Coat + Shirt", "Sandal + Sneaker + Ankle boot"]),
        "multiclasse": (np.diag([1000] * 10) + 1, [f"classe {i}" for i in range(10)]),
    }
    with tempfile.TemporaryDirectory() as tmp:
        for nome, (matriz, nomes) in casos.items():
            caminho = evaluate.salvar_matriz_confusao(matriz, nomes, Path(tmp) / "sub" / f"{nome}.png")
            conteudo = caminho.read_bytes()
            assert conteudo.startswith(PNG_ASSINATURA), nome
            assert len(conteudo) > 5000, nome   # uma figura de verdade, não um arquivo vazio

def test_matriz_confusao_todos_zeros_nao_quebra():
    with tempfile.TemporaryDirectory() as tmp:
        evaluate.salvar_matriz_confusao(np.zeros((2, 2), dtype=int), ["a", "b"], Path(tmp) / "m.png")


# Resumo no console

def test_resumo_mostra_metricas_e_classes():
    r = evaluate.calcular_metricas(ROTULOS, PREVISOES, NOMES)
    saida = io.StringIO()
    with contextlib.redirect_stdout(saida):
        evaluate.imprimir_resumo("outputs/teste", "cpu", r)
    texto = saida.getvalue()

    for esperado in ("Acurácia", "0.6250", "Precisão", "Recall", "F1", "Média macro", "0.6349", "Erro mais frequente"):
        assert esperado in texto, esperado
    for nome in NOMES:
        assert nome in texto

def test_resumo_nome_longo_e_truncado():
    nomes = ["T-shirt/top + Pullover + Coat + Shirt + Dress", "Sandal"]
    r = evaluate.calcular_metricas(np.array([0, 1]), np.array([0, 1]), nomes)
    saida = io.StringIO()
    with contextlib.redirect_stdout(saida):
        evaluate.imprimir_resumo("x", "cpu", r)
    assert "…" in saida.getvalue()
    assert "Nenhum erro" in saida.getvalue()


# Partes que usam o PyTorch (modelo, pesos e dataloader)

def _config_pequeno():
    return {"seed": 42, "dados": {"batch_size": 16},
            "modelo": {"filtros_bloco1": 4, "filtros_bloco2": 4, "neuronios_densa": 8}}

def test_prever_devolve_logits_e_rotulos_na_ordem():
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(0)
    X = torch.randn(37, 1, 28, 28)             # 37 não é múltiplo do batch: testa o último batch menor
    y = torch.randint(0, 10, (37,))
    dl = DataLoader(TensorDataset(X, y), batch_size=16, shuffle=False)

    modelo = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))
    logits, rotulos = evaluate.prever(modelo, dl, "cpu")

    assert logits.shape == (37, 10) and rotulos.shape == (37,)
    assert rotulos.tolist() == y.tolist()                          # mesma ordem
    esperado = modelo(X).detach().numpy()
    assert np.allclose(logits, esperado, atol=1e-5)                # igual a passar tudo de uma vez

def test_carregar_modelo_recupera_os_pesos_salvos():
    import torch
    from model import NeuralNetwork

    config = _config_pequeno()
    with tempfile.TemporaryDirectory() as tmp:
        pasta = Path(tmp)
        torch.manual_seed(3)
        original = NeuralNetwork(config, num_classes=2)
        utils.save_weights(original, pasta / utils.PESOS_FINAIS_PT)
        utils.save_json({**config, "num_classes": 2}, pasta / utils.HIPERPARAMETROS_JSON)

        carregado, config_lido = evaluate.carregar_modelo(pasta, "cpu")

    assert config_lido["num_classes"] == 2
    assert not carregado.training                                  # modo de avaliação
    for nome, tensor in original.state_dict().items():
        assert torch.equal(tensor, carregado.state_dict()[nome]), nome

    x = torch.randn(4, 1, 28, 28)
    with torch.no_grad():
        assert torch.allclose(original.eval()(x), carregado(x))    # mesma saída

def test_carregar_modelo_sem_arquivos_da_erro_claro():
    with tempfile.TemporaryDirectory() as tmp:
        try:
            evaluate.carregar_modelo(tmp, "cpu")
        except FileNotFoundError as erro:
            assert "Rode o experimento" in str(erro)
            return
    raise AssertionError("pasta sem pesos deveria levantar FileNotFoundError")


if __name__ == "__main__":
    testes = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    falhas = 0
    for nome, funcao in testes:
        try:
            funcao()
            print(f"OK    {nome}")
        except Exception as erro:
            falhas += 1
            print(f"FALHA {nome}: {type(erro).__name__}: {erro}")
    print(f"\n{len(testes) - falhas}/{len(testes)} testes passaram")
    sys.exit(1 if falhas else 0)