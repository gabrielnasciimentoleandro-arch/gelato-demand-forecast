# Entrega do desafio DIO

## Título

Prevendo Vendas de Sorvete com Machine Learning

## Descrição para a entrega

Desenvolvi um pipeline completo e reproduzível de Machine Learning para estimar a demanda
diária de sorvetes a partir da temperatura. O projeto gera e valida dados sintéticos de forma
transparente, compara Regressão Linear, Regressão Polinomial com Ridge e Random Forest por
validação cruzada, seleciona o melhor modelo e avalia o campeão em um conjunto de teste.

A solução registra parâmetros, métricas, artefatos, assinatura e modelo com MLflow, incluindo
Model Registry local. Também disponibiliza previsões pela CLI e por uma API FastAPI com
validação estrita, intervalo aproximado de 95% e aviso de extrapolação. O repositório contém
51 testes automatizados com 100% de cobertura do código medido, lint com Ruff, CI em três
versões do Python, auditoria de dependências, Docker, Compose, Model Card, gráficos, exemplos
de entrada e templates para uma implantação futura no Azure.

Tudo foi validado localmente. Como não utilizei uma assinatura Azure, não alego deploy,
endpoint público ou execução em nuvem; a pasta de infraestrutura contém somente preparação
documentada para evolução futura.

## Resultado principal

- modelo selecionado: Regressão Polinomial de grau 2 com Ridge;
- MAE no teste: 5,9051 sorvetes;
- RMSE no teste: 7,6666 sorvetes;
- R² no teste: 0,932370;
- origem dos dados: sintética e determinística, sem alegação de dados comerciais reais.

## Evidências no repositório

- processo e execução: `README.md`;
- métricas: `reports/metrics.json`;
- gráficos: `reports/images/`;
- Model Card: `reports/MODEL_CARD.md`;
- exemplos: `inputs/`;
- testes: `tests/`;
- rastreamento: `docs/MLFLOW.md`;
- arquitetura: `docs/ARCHITECTURE.md`;
- preparação Azure: `docs/AZURE_DEPLOYMENT.md`.

## Links a preencher ao publicar

- repositório GitHub: `https://github.com/gabrielnasciimentoleandro-arch/gelato-demand-forecast`;
- execução da CI: `https://github.com/gabrielnasciimentoleandro-arch/gelato-demand-forecast/actions`.
