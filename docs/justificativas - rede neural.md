# Justificativas

Trabalho de IA 2026 - Objetivo 2 (CNN com Fashion MNIST)

## Integrantes:
- Lucas Neves Souza, Nro. USP: 15577627
- Willian Jefferson Farias, Nro. USP: 15642665

## Arquitetura da Rede Neural

### Primeiro Bloco de Camadas

1) **Convolução 2D sobre as entradas**: 

- **in_channels**: 1 -> Como as imagens de entrada do Fashion-MNIST estão em escala de cinza, temos 1 único canal de intensidade;

- **out_channels**: 16 ->  quantos kernels queremos utilizar. Vamos criar 16 mapas de características. 16 canais na primeira camada oferece capacidade suficiente para capturar padrões básicos de baixo nível em imagens simples sem inflar o custo computacional logo no início;

- **kernel_size**: 3x3 -> Padrão comum para extração de características. É o menor tamanho capaz de capturar relações espaciais locais (esquerda/direita, cima/baixo, diagonais) com baixo custo computacional;

- **stride**: 1 -> o kernel se desloca de um em um pixel;

- **padding**: 1 -> Adiciona uma borda de 1 pixel com zeros ao redor da imagem para preservar a resolução espacial (28x28) e evitar perda de informação nas extremidades.

2) **ReLU**

- Função de ativação para introduzir não linearidade ao modelo, retirando valores negativos;

- Sem uma não linearidade entre as camadas, empilhar múltiplos blocos convolucionais colapsaria matematicamente em uma única transformação linear equivalente — impedindo a rede de aprender padrões visuais complexos e hierárquicos.

3) **MaxPool2D**

- Operação para reduzir o tamanho dos tensores de saída a partir da seleção dos valores máximos dentre **janelas de pooling** de tamanho **3x3** `kernel_size=3`, que se deslocam de dois em dois valores de cada tensor;
 
- Reduz a resolução espacial (28x28 -> 13x13), reduzindo custo computacional e conferindo invariância a pequenas translações.

### Segundo Bloco de Camadas

1) **Convolução 2D sobre as saídas do bloco anterior**: 

- **in_channels**: 16 -> Como os `out_channels` da convolução anterior são iguais a 16, definimos esse número como canais de entrada para receber corretamente a saída;

- **out_channels**: 32 ->  quantos kernels queremos utilizar. Vamos criar 32 mapas de características. Dobramos o número de mapas para um maior detalhamento da extração de características. O bloco 2 vai combinar os 16 mapas básicos para detectar partes mais complexas e abstratas das peças de roupa;

- **kernel_size**, **stride**, **padding** continuam os mesmos do bloco anterior para que não tenhamos perda da dimensionalidade espacial.

2) **ReLU**

- Função de ativação para reintroduzir a não linearidade ao modelo, retirando valores negativos que possam ter surgido.
- Mesmo que a saída do bloco1 já tenha passado por uma ReLU, a Conv2d do bloco2 realiza novas combinações lineares ponderadas por pesos treináveis (que podem ser negativos) e soma um viés (bias). Portanto, valores negativos voltam a surgir naturalmente e a nova ReLU garante que a composição entre os dois blocos continue sendo não linear.

3) **MaxPool2D**
 
- Reduz a resolução espacial (13x13 -> 6x6), mais uma vez reduzindo custo computacional e conferindo invariância a pequenas translações.

### Terceiro Bloco de Camadas

1) **Flatten**

- Camada para achatar os tensores em uma única dimensão;

- Assim, **32 x 6 x 6 = 1.152 features**.

2) **Linear**

- Primeira camada densa oculta para calcular logits. Isso permite com que a rede combine as 1.152 características locais em representações intermediárias de alto nível.

3) **ReLU**

- Função de ativação para reintroduzir a não linearidade ao modelo, retirando valores negativos que possam ter surgido.

4) **Linear**

- Segunda camada densa, agora com a saída parametrizável para deixar a arquitetura mais flexível. 

- Classificação completa do Fashion-MNIST (num_classes=10): todas as 10 peças de roupa.

- Classificação binária (num_classes=2): por exemplo, separar apenas duas categorias (como tênis vs. bota ou roupas vs. calçados).
