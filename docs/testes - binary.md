# Testes entre diferentes pares de classes para run_binary

Salvando saídas do terminal para organizar melhor as informações

## T-shirt/top vs Shirt

1) Parada antecipada

```
Parando antecipadamente na época: 15
```

2) Erro realizado no teste

```
Erro de Teste:
Acurácia: 87.6%, Perda Média: 0.278457
```

3) Resumo ao final

```
Melhor época: 12 | 
perda de validação: 0.2762 | 
acurácia de validação: 0.8757
```

4) Precisão, Recall, F1 e Suporte

```
Classe        Precisão     Recall         F1   Suporte
-------------------------------------------------------
T-shirt/top     0.8902     0.8590     0.8743      1000
Shirt           0.8638     0.8940     0.8786      1000
-------------------------------------------------------
Média macro     0.8770     0.8765     0.8765      2000

Erro mais frequente: 'T-shirt/top' previsto como 'Shirt' (141 exemplos)
```

## Sneaker vs Sandal

1) Parada antecipada

```
Parando antecipadamente na época: 9
```

2) Erro realizado no teste

```
Erro de Teste:
Acurácia: 98.4%, Perda Média: 0.044965
```

3) Resumo ao final

```
Melhor época: 6 | 
perda de validação: 0.0409 | 
acurácia de validação: 0.9862
```

4) Precisão, Recall, F1 e Suporte

```
Classe        Precisão     Recall         F1   Suporte
-------------------------------------------------------
Sneaker         0.9736     0.9950     0.9842      1000
Sandal          0.9949     0.9730     0.9838      1000
-------------------------------------------------------
Média macro     0.9842     0.9840     0.9840      2000

Erro mais frequente: 'Sandal' previsto como 'Sneaker' (27 exemplos)
```


## Dress vs Coat

1) Parada antecipada

```
Parando antecipadamente na época: 13
```

2) Erro realizado no teste

```
Erro de Teste:
Acurácia: 96.5%, Perda Média: 0.079003
```

3) Resumo ao final

```
Melhor época: 10 | 
perda de validação: 0.0823 | 
acurácia de validação: 0.9588
```

## Pullover vs Shirt

1) Parada antecipada

```
Parando antecipadamente na época: 11
```

2) Erro realizado no teste

```
Erro de Teste:
Acurácia: 91.5%, Perda Média: 0.227070
```

3) Resumo ao final

```
Melhor época: 8 | 
perda de validação: 0.1721 | 
acurácia de validação: 0.9286
```
