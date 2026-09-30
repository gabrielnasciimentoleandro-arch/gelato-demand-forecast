# Como contribuir

## Preparação

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

No PowerShell, ative com `.venv\Scripts\Activate.ps1`.

## Antes de enviar uma alteração

```bash
ruff format src tests scripts
ruff check src tests scripts
pytest -q --cov=gelato_forecast --cov-report=term-missing
pip-audit --skip-editable
pip check
```

## Diretrizes

- mantenha o dado sintético claramente identificado;
- não ajuste o modelo olhando para o conjunto de teste;
- adicione testes para novos comportamentos;
- preserve sementes e parâmetros necessários à reprodução;
- atualize métricas, Model Card e README quando o experimento mudar;
- não adicione chaves, tokens, bancos locais ou dados pessoais;
- use commits pequenos e mensagens objetivas.

Mudanças de métrica devem explicar a metodologia e comparar o mesmo split ou documentar por
que ele mudou.
