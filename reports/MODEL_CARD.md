# Model Card — Gelato Demand Forecast

## Visão geral

| Campo | Valor |
|---|---|
| tarefa | regressão supervisionada |
| versão do schema | 1 |
| algoritmo selecionado | `polynomial_ridge` |
| variável de entrada | `temperature_c` |
| alvo | `ice_creams_sold` |
| semente | 42 |

## Finalidade

Estimar a quantidade diária de sorvetes vendidos a partir da temperatura em graus Celsius.
O modelo é educacional e não deve ser usado para decisões financeiras reais sem validação
com dados da operação.

### Uso pretendido

- demonstração de um pipeline reproduzível de regressão;
- previsões locais por CLI ou API dentro do limite operacional de -20 °C a 60 °C;
- estudo de comparação de modelos, rastreamento e disponibilização de inferência.

### Uso fora de escopo

- decisões autônomas de compra, contratação ou descarte;
- previsão para uma loja real sem retreinamento e validação com seus próprios dados;
- uso como modelo meteorológico ou como fonte de verdade financeira.

## Dados

- origem: dados sintéticos e determinísticos;
- registros usados no ajuste: 584;
- registros reservados para teste: 146;
- proporção de teste: 0.20;
- faixa de temperatura observada no treino: 11.86 °C a 40.90 °C;
- o gerador contém sazonalidade, ruído e bônus de fim de semana não informado ao modelo.

Não há dados pessoais nem dados comerciais reais na base incluída.

## Seleção do modelo

A seleção usa o menor RMSE médio em validação cruzada de cinco folds no conjunto de treino.

| Candidato | CV RMSE médio | Desvio padrão |
|---|---:|---:|
| `polynomial_ridge` | 7.9670 | 0.6873 |
| `linear_regression` | 8.3394 | 0.6517 |
| `random_forest` | 8.5059 | 0.8166 |

## Avaliação do campeão

| Métrica no teste | Valor |
|---|---:|
| MAE | 5.9051 |
| RMSE | 7.6666 |
| R² | 0.932370 |

O conjunto de teste é usado depois da seleção. As métricas descrevem somente o dataset
sintético e o split registrados nos metadados.

## Inferência e incerteza

As previsões são arredondadas para unidades inteiras e limitadas a valores não negativos. O
intervalo apresentado é `previsão ± 1,96 vezes o desvio padrão dos resíduos de teste`, também
limitado a zero. Ele é uma aproximação global e não um intervalo condicional formalmente
calibrado. Temperaturas fora da faixa observada geram um aviso de extrapolação.

## Limitações e riscos

- vendas reais dependem de chuva, preço, promoções, feriados, estoque e concorrência;
- o gerador sintético contém uma relação quadrática, o que pode favorecer o candidato
  polinomial;
- o split aleatório não mede degradação temporal nem mudanças de regime;
- os resíduos podem não ter variância constante em todas as temperaturas;
- previsões fora da faixa de treinamento têm incerteza maior;
- o formato `joblib` só deve ser carregado de uma origem confiável.

## Considerações de uso responsável

O modelo não utiliza atributos pessoais ou protegidos. Ainda assim, decisões de estoque ou
trabalho baseadas em previsões imprecisas podem causar desperdício ou sobrecarga. Um uso real
exigiria supervisão humana, limites de decisão, dados governados, avaliação por loja e
monitoramento contínuo de drift e erro.

## Monitoramento recomendado

- distribuição e faixa de `temperature_c`;
- MAE e RMSE quando a venda observada ficar disponível;
- proporção de previsões com aviso de extrapolação;
- latência, erros HTTP e disponibilidade;
- mudança de preços, cardápio ou operação que invalide o relacionamento aprendido.
