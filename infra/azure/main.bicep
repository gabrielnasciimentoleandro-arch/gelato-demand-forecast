targetScope = 'resourceGroup'

@description('Azure region for all resources.')
param location string = resourceGroup().location

@description('Globally unique suffix used in resource names.')
@minLength(3)
@maxLength(20)
param nameSuffix string

@description('OCI image already published in a registry accessible by Azure Container Apps.')
param containerImage string

@description('Container CPU allocation.')
@allowed([
  '0.25'
  '0.5'
  '0.75'
  '1.0'
])
param cpu string = '0.5'

@description('Container memory allocation.')
@allowed([
  '0.5Gi'
  '1Gi'
  '1.5Gi'
  '2Gi'
])
param memory string = '1Gi'

var environmentName = 'cae-gelato-${nameSuffix}'
var appName = 'ca-gelato-api-${nameSuffix}'

resource managedEnvironment 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: environmentName
  location: location
}

resource containerApp 'Microsoft.App/containerApps@2024-03-01' = {
  name: appName
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    managedEnvironmentId: managedEnvironment.id
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: true
        targetPort: 8000
        transport: 'auto'
        allowInsecure: false
      }
    }
    template: {
      containers: [
        {
          name: 'gelato-api'
          image: containerImage
          env: [
            {
              name: 'GELATO_MODEL_PATH'
              value: '/app/models/model.joblib'
            }
            {
              name: 'GELATO_METADATA_PATH'
              value: '/app/models/metadata.json'
            }
          ]
          resources: {
            cpu: json(cpu)
            memory: memory
          }
          probes: [
            {
              type: 'Liveness'
              httpGet: {
                path: '/health'
                port: 8000
                scheme: 'HTTP'
              }
              initialDelaySeconds: 10
              periodSeconds: 30
            }
            {
              type: 'Readiness'
              httpGet: {
                path: '/health'
                port: 8000
                scheme: 'HTTP'
              }
              initialDelaySeconds: 5
              periodSeconds: 10
            }
          ]
        }
      ]
      scale: {
        minReplicas: 0
        maxReplicas: 3
        rules: [
          {
            name: 'http-scaling'
            http: {
              metadata: {
                concurrentRequests: '50'
              }
            }
          }
        ]
      }
    }
  }
}

output containerAppName string = containerApp.name
output applicationUrl string = 'https://${containerApp.properties.configuration.ingress.fqdn}'
output principalId string = containerApp.identity.principalId
