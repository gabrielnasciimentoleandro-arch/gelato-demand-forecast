# Preparação para Azure — não implantado

> **Estado real:** estes arquivos são modelos para uma evolução futura. Nenhum recurso Azure
> foi criado, nenhum endpoint público existe e nenhuma captura de tela em nuvem foi produzida,
> pois o projeto foi desenvolvido sem assinatura Azure.

## Validação local dos templates

Em 29/09/2026, `main.bicep` e `main.example.bicepparam` foram compilados localmente sem erros
com Bicep CLI 0.47.16. Os dois YAMLs de Azure ML também passaram pelos schemas JSON oficiais
indicados em `$schema`. Isso valida a sintaxe e a estrutura declarativa, mas não substitui um
`what-if` nem uma implantação em uma assinatura real.

## Opção preparada: Azure Container Apps

A API já é empacotada em um contêiner sem privilégios. O template
`infra/azure/main.bicep` descreve:

- um Container Apps Environment;
- um Container App com ingress HTTPS externo na porta 8000;
- probes de prontidão e vida em `/health`;
- escala de zero a três réplicas;
- identidade gerenciada atribuída pelo sistema;
- parâmetros de imagem, região, CPU e memória.

### Pré-requisitos futuros

- assinatura e grupo de recursos Azure;
- Azure CLI e Bicep instalados;
- imagem publicada no ACR, GHCR ou outro registro acessível;
- permissões de pull configuradas quando o registro for privado.

### Sequência futura de referência

Os comandos abaixo **não foram executados** neste projeto. Substitua todos os valores em
maiúsculas antes de um eventual uso.

```bash
az login
az account set --subscription SUBSCRIPTION_ID
az group create --name RG_GELATO --location brazilsouth
az deployment group what-if \
  --resource-group RG_GELATO \
  --template-file infra/azure/main.bicep \
  --parameters nameSuffix=SUFIXO_UNICO \
               containerImage=REGISTRY/gelato-demand-forecast:1.0.0
az deployment group create \
  --resource-group RG_GELATO \
  --template-file infra/azure/main.bicep \
  --parameters nameSuffix=SUFIXO_UNICO \
               containerImage=REGISTRY/gelato-demand-forecast:1.0.0
```

Executar `what-if` antes do deploy reduz surpresas, mas não substitui revisão de segurança,
custos e políticas da organização.

## Treinamento futuro no Azure Machine Learning

`deploy/azureml/environment.yml` e `deploy/azureml/job.yml` são esqueletos declarativos para
uma command job. Antes de usar, seria necessário:

1. criar workspace e compute;
2. publicar/criar o ambiente;
3. substituir `SUBSTITUA_PELO_COMPUTE`;
4. mapear saídas para um datastore, em vez de depender apenas do disco efêmero;
5. configurar uma URI de tracking e política de registro apropriadas;
6. validar schema e custos no tenant real.

Esses arquivos não afirmam compatibilidade validada com um workspace específico.

## Controles recomendados antes de produção

- imagem em registry privado, com digest imutável e varredura;
- autenticação e autorização na borda;
- Key Vault para segredos;
- logs sem dados sensíveis e Application Insights;
- limites de custo, alertas e política de retenção;
- armazenamento de modelo versionado e verificação de integridade;
- monitoramento de qualidade, latência, drift e faixa das features;
- estratégia de rollback e implantação gradual;
- dados reais governados, documentados e aprovados.

## Como remover recursos futuramente

Se uma implantação for realizada em um grupo exclusivo, a remoção completa pode ser feita
com:

```bash
az group delete --name RG_GELATO --yes --no-wait
```

Revise o conteúdo do grupo antes: o comando remove todos os recursos nele.
