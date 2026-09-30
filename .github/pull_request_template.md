## O que mudou?

Descreva a alteração e a motivação.

## Como foi validado?

- [ ] `ruff format --check src tests scripts`
- [ ] `ruff check src tests scripts`
- [ ] `pytest --cov=gelato_forecast`
- [ ] `pip-audit --skip-editable`
- [ ] documentação atualizada, quando aplicável

## Checklist de ML

- [ ] não houve vazamento de dados;
- [ ] sementes e parâmetros permanecem reproduzíveis;
- [ ] métricas foram comparadas com a referência;
- [ ] a origem sintética dos dados continua explícita.
