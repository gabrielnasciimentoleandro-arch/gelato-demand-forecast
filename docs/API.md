# API HTTP

## Executar localmente

```bash
gelato-forecast serve --host 0.0.0.0 --port 8000
```

A documentação interativa fica em `http://127.0.0.1:8000/docs` e o schema OpenAPI em
`http://127.0.0.1:8000/openapi.json`.

## Endpoints

### `GET /`

Identifica o serviço e informa a versão e o caminho da documentação.

### `GET /health`

Verifica se os dois artefatos necessários existem.

Resposta pronta:

```json
{
  "status": "healthy",
  "model_ready": true
}
```

O endpoint continua respondendo HTTP 200 no estado `degraded`; orquestradores devem avaliar
`model_ready`, como faz o `HEALTHCHECK` do contêiner.

### `POST /predict`

Requisição:

```json
{
  "temperature_c": 30.0
}
```

Resposta do modelo versionado:

```json
{
  "temperature_c": 30.0,
  "predicted_sales": 132,
  "prediction_interval_95": [117, 147],
  "model_name": "polynomial_ridge",
  "warning": null
}
```

Regras de entrada:

- o corpo aceita somente `temperature_c`;
- o valor deve ser um número JSON, sem coerção de texto;
- o limite operacional é de -20 °C a 60 °C;
- temperatura fora da faixa observada no treino ainda pode ser aceita, mas gera um aviso de
  extrapolação.

## Erros

Erros de validação usam HTTP 422 no formato padrão do FastAPI. Modelo ausente ou metadados
inválidos retornam HTTP 503 com `application/problem+json`:

```json
{
  "type": "https://httpstatuses.com/503",
  "title": "Model not ready",
  "status": 503,
  "detail": "Model artifacts were not found. Run the training pipeline first.",
  "instance": "/predict"
}
```

## Exemplo com cURL

```bash
curl --fail --request POST http://127.0.0.1:8000/predict \
  --header 'Content-Type: application/json' \
  --data '{"temperature_c":30.0}'
```

A API é educacional e não implementa autenticação nem rate limiting. Não a exponha diretamente
à internet sem um gateway ou proxy com esses controles.
