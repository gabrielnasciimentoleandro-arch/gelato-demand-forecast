# Política de segurança

## Versões suportadas

Este projeto educacional mantém apenas a versão mais recente da branch `main`.

## Como relatar

Não publique credenciais, dados pessoais ou detalhes exploráveis em uma issue pública.
Ao publicar o repositório, use o canal privado de security advisories do GitHub para relatar
uma vulnerabilidade.

## Premissas importantes

- os dados incluídos são sintéticos e não contêm informações de clientes;
- `.env`, bancos MLflow e artefatos temporários são ignorados pelo Git;
- a API não possui autenticação nem rate limiting e não deve ser exposta diretamente;
- arquivos `joblib` usam mecanismo compatível com pickle: nunca carregue um modelo de origem
  não confiável;
- templates Azure não foram implantados e não contêm credenciais;
- dependências são auditadas com `pip-audit`, mas uma auditoria sem alertas não garante ausência
  de vulnerabilidades.

## Verificações locais

```bash
pip-audit --skip-editable
pip check
ruff check src tests scripts
pytest --cov=gelato_forecast
```
