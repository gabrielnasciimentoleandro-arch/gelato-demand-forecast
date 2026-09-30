# Rastreamento com MLflow

## O que é registrado

Cada execução de `gelato-forecast train` registra:

- **parâmetros:** modelo selecionado, semente, proporção de teste, quantidade de linhas e hash
  SHA-256 do dataset;
- **métricas:** MAE, RMSE e R² no teste, além do RMSE de validação de cada candidato;
- **tags:** projeto, origem sintética dos dados e tipo da tarefa;
- **artefatos:** CSV, metadados, gráficos e modelo scikit-learn;
- **assinatura:** uma coluna `temperature_c` e saída numérica inferidas pelo MLflow;
- **registro:** uma versão em `gelato-demand-model`, salvo quando `--no-register` é usado.

## Treinar e registrar

```bash
gelato-forecast train
```

O backend padrão é o SQLite local `mlflow.db`. O banco e os artefatos internos do MLflow não
são versionados no Git; eles são recriados pela execução.

Para rastrear sem criar uma versão no Registry:

```bash
gelato-forecast train --no-register
```

Para treinar sem MLflow:

```bash
gelato-forecast train --no-mlflow
```

## Abrir a interface local

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Depois, acesse `http://127.0.0.1:5000` no mesmo computador.

## Resultado validado neste projeto

A validação local criou o experimento `gelato-demand-forecast`, registrou os parâmetros e as
métricas do campeão e criou a versão 1 de `gelato-demand-model` no Registry local. Essa
validação foi feita somente no ambiente local; não existe workspace, endpoint ou registro de
modelo no Azure.

## Configuração por ambiente

| Variável | Padrão |
|---|---|
| `GELATO_PROJECT_ROOT` | diretório de trabalho atual |
| `MLFLOW_TRACKING_URI` | `sqlite:///CAMINHO_DO_PROJETO/mlflow.db` |
| `MLFLOW_EXPERIMENT_NAME` | `gelato-demand-forecast` |
| `MLFLOW_MODEL_NAME` | `gelato-demand-model` |

É possível apontar `MLFLOW_TRACKING_URI` para outro backend no futuro, mas credenciais e
segredos nunca devem ser adicionados ao repositório.
