"""
Trabalho de IA 2026 - Objetivo 2 (CNN com Fashion MNIST)

Integrantes:
    Lucas Neves Souza, Nro. USP: 15577627
    Willian Jefferson Farias, Nro. USP: 15642665

tests/test_binario.py: testes do filtro de classes e do remapeamento de rótulos
(tarefa binária). Pode ser executado de duas formas, a partir da raiz do projeto:

    python tests/test_binario.py      # roda tudo e imprime um resumo por teste
    pytest tests/                     # com o pytest estiver instalado

Os testes usam o Fashion MNIST real (baixado em data/ na primeira execução) e
conferem as contagens efetivamente obtidas em cada partição, porque o
random_split não é estratificado
"""

import sys
from functools import lru_cache
from pathlib import Path

import torch
import yaml

# Permite rodar `python tests/test_binario.py` sem instalar o projeto como pacote
RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.dataset import ( 
    CLASSES,
    BinaryDataset,
    classes_para_indices,
    create_binary_datasets,
    create_dataloaders,
    filter_binary_classes,
    load_fashionMNIST,
    obter_rotulos,
)

# Pares testados: (nome, classe_0, classe_1)
PARES = [
    ("fácil: Trouser x Bag", ["Trouser"], ["Bag"]),
    ("difícil: T-shirt/top x Shirt", ["T-shirt/top"], ["Shirt"]),
    (
        "composição: roupas de cima x calçados",
        ["T-shirt/top", "Pullover", "Coat", "Shirt"],
        ["Sandal", "Sneaker", "Ankle boot"],
    ),
]



# Utilitários

def ler_config():
    return load_config()


@lru_cache(maxsize=1)
def particoes():
    #Carrega o Fashion MNIST uma única vez e reaproveita nos testes (treino, validação, teste)
    return load_fashionMNIST(ler_config()["seed"])


def deve_falhar(funcao, *args):
    #Garante que funcao(*args) levanta ValueError (equivalente simples ao pytest.raises)
    try:
        funcao(*args)
    except ValueError:
        return
    raise AssertionError(f"{funcao.__name__}{args} deveria levantar ValueError")


def posicoes_base(particao, binario):
    #Índices, no dataset-base de 60.000 (ou 10.000) imagens, dos exemplos do BinaryDataset
    #Serve para checar que treino e validação não compartilham imagens depois do filtro
    
    if hasattr(particao, "indices"):  # Subset do random_split
        return [particao.indices[p] for p in binario.subset.indices]
    return list(binario.subset.indices)



# Testes

def test_split_documentado():
    # O split tem os tamanhos documentados (50.000 / 10.000 / 10.000)
    treino, validacao, teste = particoes()
    assert (len(treino), len(validacao), len(teste)) == (50000, 10000, 10000)


def test_classes_para_indices():
    assert classes_para_indices(["T-shirt/top"]) == [0]
    assert classes_para_indices(["Shirt", "Bag"]) == [6, 8]
    assert classes_para_indices(("Trouser",)) == [1]  # tupla também é aceita


def test_validacoes_de_entrada():
    # Lista vazia, string solta em vez de lista, duplicatas e nome inexistente
    deve_falhar(classes_para_indices, [])
    deve_falhar(classes_para_indices, "Shirt")
    deve_falhar(classes_para_indices, ["Shirt", "Shirt"])
    deve_falhar(classes_para_indices, ["Camisa"])

    # Classe repetida nos dois lados
    treino = particoes()[0]
    deve_falhar(filter_binary_classes, treino, ["Shirt", "Bag"], ["Bag"])
    # Os mesmos erros via filtro (lista vazia nos dois lados)
    deve_falhar(filter_binary_classes, treino, [], ["Bag"])
    deve_falhar(filter_binary_classes, treino, ["Shirt"], [])


def test_rotulos_so_0_e_1_e_classes_corretas():
    # Para cada par e cada partição: rótulos só 0/1 e todas as imagens vêm das classes pedidas
    for nome, c0, c1 in PARES:
        ids0, ids1 = classes_para_indices(c0), classes_para_indices(c1)
        for particao in particoes():
            binario = filter_binary_classes(particao, c0, c1)
            originais = obter_rotulos(particao)

            # 1) só rótulos 0 e 1
            assert set(binario.targets.tolist()) <= {0, 1}, nome

            # 2) cada exemplo mantido vem de uma das classes pedidas, e o 0/1 bate com o lado
            for i, pos in enumerate(binario.subset.indices):
                original = int(originais[pos])
                if binario.targets[i] == 0:
                    assert original in ids0, (nome, original)
                else:
                    assert original in ids1, (nome, original)

            # 3) nenhuma imagem das classes pedidas ficou de fora
            esperado = int(torch.isin(originais, torch.tensor(ids0 + ids1)).sum())
            assert len(binario) == esperado, (nome, len(binario), esperado)


def test_contagens_efetivas_por_particao():
    # Confere as contagens reais por classe (o random_split não é estratificado)
    for nome, c0, c1 in PARES:
        ids0, ids1 = classes_para_indices(c0), classes_para_indices(c1)
        for particao in particoes():
            binario = filter_binary_classes(particao, c0, c1)
            originais = obter_rotulos(particao)

            n0_esperado = int(torch.isin(originais, torch.tensor(ids0)).sum())
            n1_esperado = int(torch.isin(originais, torch.tensor(ids1)).sum())
            contagem = torch.bincount(binario.targets, minlength=2).tolist()

            assert contagem == [n0_esperado, n1_esperado], (nome, contagem)
            assert sum(contagem) == len(binario)


def test_dataset_original_nao_e_modificado():
    # O remapeamento fica no BinaryDataset
    treino, validacao, teste = particoes()
    antes = [obter_rotulos(p).clone() for p in (treino, validacao, teste)]

    for _, c0, c1 in PARES:
        for particao in (treino, validacao, teste):
            filter_binary_classes(particao, c0, c1)

    depois = [obter_rotulos(p) for p in (treino, validacao, teste)]
    for a, d in zip(antes, depois):
        assert torch.equal(a, d)
    assert int(obter_rotulos(treino).max()) == 9  # ainda existem rótulos de 0 a 9


def test_item_devolve_imagem_correta_e_rotulo_binario():
    #__getitem__ devolve a mesma imagem da partição original, com rótulo 0/1 (int)
    c0, c1 = ["T-shirt/top"], ["Shirt"]
    treino = particoes()[0]
    binario = filter_binary_classes(treino, c0, c1)

    for i in (0, 1, len(binario) // 2, len(binario) - 1):
        imagem, rotulo = binario[i]
        imagem_original, rotulo_original = treino[binario.subset.indices[i]]
        assert torch.equal(imagem, imagem_original)
        assert isinstance(rotulo, int) and rotulo in (0, 1)
        assert rotulo == (1 if rotulo_original == 6 else 0)
        assert imagem.shape == (1, 28, 28)


def test_treino_e_validacao_continuam_disjuntos():
    # Filtrar não pode misturar treino e validação (nenhuma imagem nas duas)
    treino, validacao, _ = particoes()
    for nome, c0, c1 in PARES:
        base_treino = set(posicoes_base(treino, filter_binary_classes(treino, c0, c1)))
        base_val = set(posicoes_base(validacao, filter_binary_classes(validacao, c0, c1)))
        assert base_treino.isdisjoint(base_val), nome


def test_par_do_config_yaml():
    # O par definido em config.yaml é lido e aplicado nas três partições.
    config = ler_config()
    treino, validacao, teste = create_binary_datasets(config, *particoes())

    for binario in (treino, validacao, teste):
        assert isinstance(binario, BinaryDataset)
        assert len(binario) > 0
        assert set(binario.targets.tolist()) == {0, 1}


def test_dataloader_com_dataset_binario():
    # Os DataLoaders existentes funcionam com os datasets binários (batch_size vem do config)
    config = ler_config()
    treino, validacao, teste = create_binary_datasets(config, *particoes())
    loaders = create_dataloaders(config, treino, validacao, teste)

    for loader in loaders:
        imagens, rotulos = next(iter(loader))
        assert imagens.shape[1:] == (1, 28, 28)
        assert imagens.shape[0] <= config["dados"]["batch_size"]
        assert rotulos.dtype == torch.int64
        assert set(rotulos.tolist()) <= {0, 1}


def test_ordem_do_treino_reprodutivel():
    # Com a mesma seed, dois DataLoaders de treino entregam os mesmos batches
    config = ler_config()
    treino, validacao, teste = create_binary_datasets(config, *particoes())

    primeiro = next(iter(create_dataloaders(config, treino, validacao, teste)[0]))
    segundo = next(iter(create_dataloaders(config, treino, validacao, teste)[0]))
    assert torch.equal(primeiro[0], segundo[0]) and torch.equal(primeiro[1], segundo[1])


def test_relatorio_dos_pares():
    # Imprime as contagens por partição (aparece com `python tests/test_binario.py` ou `pytest -s`)
    print()
    for nome, c0, c1 in PARES:
        print(f"  {nome}")
        for rotulo, particao in zip(("treino", "validação", "teste"), particoes()):
            b = filter_binary_classes(particao, c0, c1)
            n0, n1 = torch.bincount(b.targets, minlength=2).tolist()
            print(f"    {rotulo:<10} total={len(b):>6}  classe_0={n0:>6}  classe_1={n1:>6}")


if __name__ == "__main__":
    testes = [(n, f) for n, f in list(globals().items()) if n.startswith("test_") and callable(f)]
    falhas = 0
    for nome, funcao in testes:
        try:
            funcao()
            print(f"OK    {nome}")
        except Exception as erro:  # noqa: BLE001
            falhas += 1
            print(f"FALHA {nome}: {type(erro).__name__}: {erro}")
    print(f"\n{len(testes) - falhas}/{len(testes)} testes passaram")
    sys.exit(1 if falhas else 0)