# Justificativas

Trabalho de IA 2026 - Objetivo 2 (CNN com Fashion MNIST)

## Integrantes:
- Lucas Neves Souza, Nro. USP: 15577627
- Willian Jefferson Farias, Nro. USP: 15642665

## Hiperparâmetros de treino e tentativas (tarefa multiclasse)

 Aqui ficam as escolhas de otimizador, taxa de aprendizado, número de épocas, parada antecipada e tamanho do lote, junto com o registro das tentativas que levaram à configuração final. Todos esses valores estão no `config.yaml`.

<!-- As seções 1, 2 e 3 abaixo seguem, na mesma ordem, as três perguntas do slide "Determinação de parâmetros" do vídeo 2. -->

### 1. Como os parâmetros são determinados

- **Em arquivo de configuração, não no código dos scripts:** os hiperparâmetros ficam no `config.yaml`, que cada script lê com `utils.load_config()` e valida com `utils.validate_config()` (tipo e faixa de cada valor) antes de treinar. Mudar um valor não exige alterar nenhum `.py`.
- **Parâmetros da arquitetura:** o número de filtros dos dois blocos e o de neurônios da camada densa vêm do bloco `modelo` do `config.yaml`, lidos em `NeuralNetwork.__init__`. O número de saídas vem do script (`num_classes=10` no multiclasse, 2 no binário). Kernel (3x3), padding (1) e pooling (janela 3x3, stride 2) são constantes escritas em `src/model.py`.
- **Linha de comando:** apenas o par de classes da tarefa binária (`--classe0` e `--classe1` em `run_binary.py`). Não há interface gráfica.
- **Registro:** cada execução grava a configuração usada em `outputs/<experimento>/hiperparametros.json`, junto com o número de classes, o dispositivo e a inicialização dos pesos, o que permite reproduzir o resultado.

### 2. Quais parâmetros podem ser variados

| Parâmetro | Onde alterar | Valor base | Valores testados |
|-----------|--------------|------------|------------------|
| Otimizador | `config.yaml`: `treino.otimizador` | adam | adam, sgd (momentum 0,9 fixo em `train.build_optimizer`) |
| Taxa de aprendizado | `config.yaml`: `treino.lr` | 0.001 | 0.0003, 0.001, 0.003 (adam); 0.01 (sgd) |
| Épocas (teto) | `config.yaml`: `treino.epocas` | 10 | 10, 20 |
| Tamanho do lote | `config.yaml`: `dados.batch_size` | 64 | 64, 128 |
| Filtros (bloco 1 / bloco 2) | `config.yaml`: `modelo.filtros_bloco1` e `modelo.filtros_bloco2` | 16 / 32 | 16 / 32, 32 / 64 |
| Neurônios da camada densa | `config.yaml`: `modelo.neuronios_densa` | 128 | 64, 128, 256 |
| Pooling (janela / stride) | `src/model.py` (código) | 3x3 / 2 | 3x3 / 2, 2x2 / 2 (opcional) |
| Parada antecipada | `config.yaml`: `parada.patience` e `parada.delta` | 3 / 0 | não variados |
| Seed | `config.yaml`: `seed` | 42 | fixa, para reprodutibilidade |
| Kernel e padding da convolução | `src/model.py` (código) | 3x3 / 1 | não variados |

### 3. Procedimento de exploração dos valores

#### Metodologia

- **Dados:** treino 50.000, validação 10.000 e teste 10.000 (o teste é o conjunto oficial; treino e validação vêm do `random_split` com seed 42).
- **Critério de escolha: validação.** Cada tentativa registra a perda e a acurácia de **validação da melhor época**, que é a época cujos pesos a parada antecipada restaura. A acurácia de teste aparece nas tabelas só para informação. Escolher a configuração pelo teste faria dele parte do ajuste e ele deixaria de ser uma estimativa honesta de desempenho.
- **Uma mudança por vez** em relação à base de cada rodada, para saber o que causou cada diferença.
- **Empate técnico:** diferença de acurácia de validação menor que 0,3 ponto percentual (aproximadamente o erro-padrão da acurácia com 10.000 exemplos e acurácia em torno de 91%). Nesse caso, preferimos a configuração mais simples ou mais rápida.
- **Uma execução por tentativa, seed 42.** Diferenças pequenas podem ser efeito da inicialização e da ordem dos lotes, não da mudança testada.
- **Parada antecipada:** `patience = 3` e `delta = 0`. O treino para quando a perda de validação não melhora por 3 épocas seguidas, e os pesos da melhor época são restaurados. Por isso `epocas` funciona como um teto, não como um número fixo de épocas.

#### Ordem das rodadas

A melhor configuração de cada rodada vira a base da rodada seguinte.

| Rodada | O que varia | Base |
|--------|-------------|------|
| A | épocas, taxa de aprendizado e otimizador | T0 (configuração inicial) |
| B | capacidade da rede (neurônios da camada densa e filtros) | melhor da Rodada A |
| C | tamanho do lote (opcional) | melhor da Rodada B |
| D | pooling 2x2 em vez de 3x3 (opcional) | melhor da Rodada C |

#### Como ler as tabelas

- **Épocas (rodadas / melhor):** quantas épocas o treino executou (menos que o teto se a parada antecipada atuou) e qual época teve a menor perda de validação, cujos pesos foram mantidos.
- **Acur. val. e Perda val.:** valores de validação da melhor época. A **Acur. teste** é só informativa.
- **Δ val. (pp):** acurácia de validação da tentativa menos a da base da rodada, em pontos percentuais. Valor entre -0,3 e +0,3 é empate técnico.
- **Decisão:** `Base`, `Escolhida`, `Descartada` ou `Empate` (nesse caso, vale a mais simples ou mais rápida).

#### Rodada A: épocas, taxa de aprendizado e otimizador

T1 muda só o teto de épocas. **A partir de T1, a base passa a ter 20 épocas**, para que o teto não limite as comparações seguintes. Por isso T2, T3 e T4 são comparadas com T1.

| ID | Alteração | Épocas (rodadas / melhor) | Acur. val. | Perda val. | Δ val. (pp) | Acur. teste | Decisão |
|----|-----------|---------------------------|------------|------------|-------------|-------------|---------|
| T0 | **Base inicial:** adam, lr 0.001, épocas 10, filtros 16/32, densa 128, lote 64 | 10 / 9 | 91,52% | 0,2344 | (base de T1) | 90,79% | Base |
| T1 | **épocas** 10 → 20 | 14 / 11 | 91,84% | 0,2259 | +0,32% | 91,25% | |
| T2 | **lr** 0.001 → 0.003 (épocas 20) | 7 / 4 | 90,85% | 0,2488| -0,99% (vs T1) | 90,56% | |
| T3 | **lr** 0.001 → 0.0003 (épocas 20) | 20 / 18 | 91,15% | 0,2406 | -0,69% (vs T1) | 90,63% | |
| T4 | **otimizador** adam → sgd, com **lr** 0.01 (épocas 20) | 15 / 12 | 91,65% | 0,2365 | -0,29% (vs T1) | 90,91% | |

**Escolha da rodada (base da Rodada B):** T1

#### Rodada B: capacidade da rede

| ID | Alteração | Épocas (rodadas / melhor) | Acur. val. | Perda val. | Δ val. (pp) | Acur. teste | Decisão |
|----|-----------|---------------------------|------------|------------|-------------|-------------|---------|
| T1 | **Base:** melhor da Rodada A (T1) | 14 / 11 | 91,84% | 0,2259 | - | 91,25% | Base |
| T5 | **densa** 128 → 64 | 15 / 12 | 91,79% | 0,2359 | -0,05% | 91,04% | |
| T6 | **densa** 128 → 256 | 11 / 8 | 91,80% | 0,2267 | -0,04% | 91,51% | |
| T7 | **filtros** 16/32 → 32/64 | 13 / 10 | 92,22% | 0,221 | +0,38% | 91,72%  | |

**Escolha da rodada (base da Rodada C):** T7

> Se `filtros_bloco1`, `filtros_bloco2` ou `neuronios_densa` mudarem em relação a 16, 32 e 128, os números de `justificativas - rede neural.md` (1.152 valores no Flatten, cerca de 147 mil pesos, trecho "o valor ainda deve ser confirmado na seção de experimentos") precisam ser atualizados.

#### Rodada C: tamanho do lote (opcional)

| ID | Alteração | Épocas (rodadas / melhor) | Acur. val. | Perda val. | Δ val. (pp) | Acur. teste | Decisão |
|----|-----------|---------------------------|------------|------------|-------------|-------------|---------|
| T7 | **Base:** melhor da Rodada B (T7) | 13 / 10 | 92,22% | 0,221 | - | 91,72% | Base |
| T8 | **lote** 64 → 128 | 12 / 9 | 91,95% | 0,2232 | -0,27 | 91,17% | |

**Escolha da rodada (base da Rodada D):** T7

#### Rodada D: pooling (opcional)

`justificativas - rede neural.md` adota MaxPool 3x3 com stride 2 e cita 2x2 com stride 2 como alternativa comum (resoluções 14x14 e 7x7; Flatten de 32 x 7 x 7 = 1.568). T9 compara as duas.

| ID | Alteração | Épocas (rodadas / melhor) | Acur. val. | Perda val. | Δ val. (pp) | Acur. teste | Decisão |
|----|-----------|---------------------------|------------|------------|-------------|-------------|---------|
| T7 | **Base:** melhor da Rodada C (PREENCHER o ID) | 13 / 10 | 92,22% | 0,221 | - | 91,72% | Base |
| T9 | **pooling** 3x3 / 2 → 2x2 / 2 | 8 / 5 | 91,70% | 0,2328 | -0,52% | 90,91% | |

**Escolha da rodada:** T7

#### Gráficos

- **Acurácia de validação por tentativa** (inserir `acuracia_validacao_por_tentativa.png`): mostra de uma vez qual configuração foi melhor e quais ficaram na faixa de empate técnico.
- **Perda de validação por época** (inserir `curvas_perda_validacao.png`): mostra a velocidade de convergência de cada tentativa e a época em que cada uma atingiu o mínimo.

#### Conclusão do procedimento

PREENCHER (em 2 ou 3 frases: o que mais influenciou o resultado, o que quase não mudou e o que ficou fora da exploração).

### Configuração final

| Hiperparâmetro | Valor final | 
|----------------|-------------|
| Otimizador | adam |
| Taxa de aprendizado | 0.001 |
| Épocas (teto) e parada antecipada | 13 / 10 |
| Filtros (bloco 1 / bloco 2) | 32/64 |
| Neurônios da camada densa | 128 | 
| Tamanho do lote | 64 |
| Pooling | 3x3 / 2 | 

### Resultado final no conjunto de teste

Medido uma única vez, com a configuração final, depois de todas as escolhas feitas pela validação:

| Métrica | Valor |
|---------|-------|
| Acurácia | 91,72% |
| Precisão (média macro) | 0,9172 |
| Recall (média macro) | 0,9172 |
| F1 (média macro) | 0,9168 |

Os mesmos valores constam no `README.md`.

### Limitações

- Cada tentativa é uma única execução com seed 42, então diferenças menores que 0,3 ponto percentual não são conclusivas.
- A busca varre um fator por vez e não testa combinações entre eles.
- Os hiperparâmetros de treino do `config.yaml` valem também para a tarefa binária e para a variação com HOG: PREENCHER (se os experimentos binários foram refeitos com a configuração final).