# Justificativas

Trabalho de IA 2026 - Objetivo 2 (CNN com Fashion MNIST)

## Integrantes:
- Lucas Neves Souza, Nro. USP: 15577627
- Willian Jefferson Farias, Nro. USP: 15642665

## Arquitetura da Rede Neural
 
Os valores de filtros e neurônios abaixo (16, 32 e 128) correspondem à configuração atual do `config.yaml` (`modelo.filtros_bloco1`, `modelo.filtros_bloco2` e `modelo.neuronios_densa`). Se esses valores mudarem, os números deste documento devem ser atualizados.
 
### Fórmulas de dimensão
 
Os tamanhos citados nas justificativas seguem estas fórmulas (N: tamanho da entrada, F: tamanho do kernel ou da janela, P: padding, S: stride, divisão inteira):
 
- Convolução: `(N - F + 2P) / S + 1`
- Pooling: `(N - F) / S + 1`
Percurso das dimensões para uma imagem de entrada: 1x28x28 -> (conv) 16x28x28 -> (pool) 16x13x13 -> (conv) 32x13x13 -> (pool) 32x6x6 -> (flatten) 1.152 -> (densa) 128 -> (saída) num_classes.
 
### Primeiro Bloco de Camadas
 
1) **Convolução 2D sobre as entradas**:
- **in_channels**: 1 -> Como as imagens de entrada do Fashion-MNIST estão em escala de cinza, temos 1 único canal de intensidade;
- **out_channels**: 16 -> quantos kernels queremos utilizar. Vamos criar 16 mapas de características. 16 canais na primeira camada oferece capacidade suficiente para capturar padrões básicos de baixo nível em imagens simples sem inflar o custo computacional logo no início;
- **kernel_size**: 3x3 -> Padrão comum para extração de características. É o menor tamanho ímpar, o que dá ao kernel um pixel central e permite um padding simétrico que preserva o tamanho da imagem (kernels pares, como 2x2, não conseguem isso). Ainda assim, é capaz de capturar relações espaciais locais (esquerda/direita, cima/baixo, diagonais) com baixo custo computacional;
- **stride**: 1 -> o kernel se desloca de um em um pixel;
- **padding**: 1 -> Adiciona uma borda de 1 pixel com zeros ao redor da imagem para preservar a resolução espacial (28x28) e evitar perda de informação nas extremidades. Conferência: `(28 - 3 + 2) / 1 + 1 = 28`.
2) **ReLU**
- Função de ativação para introduzir não linearidade ao modelo, substituindo valores negativos por zero (`max(0, x)`). É barata de calcular e sofre menos com o desaparecimento do gradiente do que funções saturantes como sigmoide e tanh;
- Sem uma não linearidade entre as camadas, empilhar múltiplos blocos convolucionais colapsaria matematicamente em uma única transformação linear equivalente — impedindo a rede de aprender padrões visuais complexos e hierárquicos.
3) **MaxPool2D**
- Operação para reduzir o tamanho dos tensores de saída a partir da seleção dos valores máximos dentre **janelas de pooling** de tamanho **3x3** `kernel_size=3`, que se deslocam de dois em dois valores de cada tensor `stride=2`;
- Como o passo (2) é menor que a janela (3), janelas vizinhas se sobrepõem levemente (*overlapping pooling*, técnica usada, por exemplo, na AlexNet), de modo que nenhum pixel fica de fora do pooling. Uma alternativa comum seria janela 2x2 com stride 2, que levaria a 14x14 e depois 7x7 (Flatten de 32x7x7 = 1.568 valores). A comparação entre as duas configurações pode ser feita na seção de experimentos;
- Reduz a resolução espacial (28x28 -> 13x13, pois `(28 - 3) / 2 + 1 = 13`), reduzindo custo computacional e conferindo invariância a pequenas translações.
### Segundo Bloco de Camadas
 
1) **Convolução 2D sobre as saídas do bloco anterior**:
- **in_channels**: 16 -> Como os `out_channels` da convolução anterior são iguais a 16, definimos esse número como canais de entrada para receber corretamente a saída;
- **out_channels**: 32 -> quantos kernels queremos utilizar. Vamos criar 32 mapas de características. Dobramos o número de mapas para um maior detalhamento da extração de características. O bloco 2 vai combinar os 16 mapas básicos para detectar partes mais complexas e abstratas das peças de roupa;
- **kernel_size**, **stride**, **padding** continuam os mesmos do bloco anterior para que não tenhamos perda da dimensionalidade espacial.
2) **ReLU**
- Função de ativação para reintroduzir a não linearidade ao modelo, substituindo por zero os valores negativos que possam ter surgido.
- Mesmo que a saída do bloco1 já tenha passado por uma ReLU, a Conv2d do bloco2 realiza novas combinações lineares ponderadas por pesos treináveis (que podem ser negativos) e soma um viés (bias). Portanto, valores negativos voltam a surgir naturalmente e a nova ReLU garante que a composição entre os dois blocos continue sendo não linear.
3) **MaxPool2D**
- Reduz a resolução espacial (13x13 -> 6x6, pois `(13 - 3) / 2 + 1 = 6`), mais uma vez reduzindo custo computacional e conferindo invariância a pequenas translações.
### Classificador (Camadas Densas)
 
1) **Flatten**
- Camada para achatar os tensores em uma única dimensão;
- Assim, **f2 x 6 x 6 features** (1.152 com f2 = 32, ou seja, 32 x 6 x 6 = 1.152).
2) **Linear**
- Primeira camada densa oculta. Isso permite que a rede combine as 1.152 características locais em representações intermediárias de alto nível;
- **out_features**: 128 -> valor intermediário entre as 1.152 entradas e as saídas finais. Oferece capacidade para combinar as características extraídas sem inflar demais a quantidade de parâmetros (esta camada concentra a maior parte deles, cerca de 147 mil). O valor ainda deve ser confirmado na seção de experimentos.
3) **ReLU**
- Função de ativação para reintroduzir a não linearidade ao modelo, substituindo por zero os valores negativos que possam ter surgido.
4) **Linear**
- Segunda camada densa (camada de saída), que calcula os **logits**, um valor por classe. Não há softmax na rede, pois a função de perda `CrossEntropyLoss` já o aplica internamente;
- Saída parametrizável (`num_classes`) para deixar a arquitetura mais flexível;
- Classificação completa do Fashion-MNIST (num_classes=10): todas as 10 peças de roupa;
- Classificação binária (num_classes=2): por exemplo, separar apenas duas categorias (como tênis vs. bota ou roupas vs. calçados).
