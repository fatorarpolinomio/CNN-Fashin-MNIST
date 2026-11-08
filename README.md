# Trabalho de IA 2026: Objetivo 2 (CNN com Fashion MNIST)

## Descrição

Implementação de uma Convolutional Neural Network (CNN) em PyTorch para classificação do conjunto Fashion MNIST, com duas tarefas:

- **Multiclasse:** reconhecimento das 10 classes.
- **Binária:** reconhecimento de duas classes (par configurável em `config.yaml`).

Também há uma variação com descritor HOG (extraído com scikit-image) como entrada da rede.

## Requisitos

- Python 3.10 ou 3.11
- Dependências listadas em `requirements.txt`

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Como executar

```bash
python scripts/run_multiclass.py   # tarefa multiclasse
python scripts/run_binary.py       # tarefa binária
python scripts/run_hog.py          # variação com HOG
```

Os hiperparâmetros (taxa de aprendizado, épocas, batch size, otimizador, classes do teste binário) ficam em `config.yaml`. O Fashion MNIST é baixado automaticamente em `data/` na primeira execução.

## Estrutura do projeto

```text
src/        código principal (dataset, features, model, train, evaluate, utils)
scripts/    scripts de execução de cada experimento
outputs/    arquivos de saída de cada experimento
docs/       checklists e justificativas das escolhas
video/      roteiro do vídeo 2
```

## Arquivos de saída

Cada experimento gera, em `outputs/<experimento>/`:

- `hiperparametros.json`: hiperparâmetros da arquitetura e da inicialização
- `pesos_iniciais.pt`: pesos antes do treinamento
- `pesos_finais.pt`: pesos após o treinamento
- `erro_por_iteracao.csv`: erro a cada iteração/época
- `saidas_teste.csv`: saídas da rede para cada dado de teste
- `matriz_confusao.png`: matriz de confusão do teste

## Referências

- scikit-image: van der Walt et al. (2014). https://doi.org/10.7717/peerj.453
- PyTorch: https://pytorch.org
- Fashion MNIST: Xiao et al. (2017). https://github.com/zalandoresearch/fashion-mnist
