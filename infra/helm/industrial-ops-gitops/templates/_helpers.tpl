{{- define "industrial-ops-gitops.validate" -}}
{{- $repo := required "repository.url must point at the reviewed project Git repository" .Values.repository.url -}}
{{- if not (or (hasPrefix "https://" $repo) (hasPrefix "ssh://" $repo) (hasPrefix "git@" $repo)) -}}
{{- fail "repository.url must use HTTPS or SSH Git transport" -}}
{{- end -}}
{{- if not (regexMatch "^[0-9a-f]{40}$" .Values.repository.revision) -}}
{{- fail "repository.revision must be an immutable full 40-character Git commit" -}}
{{- end -}}
{{- if not (regexMatch "^[a-z0-9]([-a-z0-9]*[a-z0-9])?$" .Values.environment) -}}
{{- fail "environment must be a DNS-label-safe value" -}}
{{- end -}}
{{- $_ := required "platform.gateway.name must be configured" .Values.platform.gateway.name -}}
{{- $_ := required "platform.gateway.hostname must be configured" .Values.platform.gateway.hostname -}}
{{- $_ := required "platform.gateway.tlsSecretName must reference an external TLS Secret" .Values.platform.gateway.tlsSecretName -}}
{{- if and (eq .Values.environment "production") .Values.platform.gateway.allowHttp -}}
{{- fail "production cannot enable the plaintext HTTP Gateway listener" -}}
{{- end -}}
{{- if .Values.releaseController.enabled -}}
{{- $_ := required "releaseController.image.repository must be configured" .Values.releaseController.image.repository -}}
{{- if not (regexMatch "^sha256:[0-9a-f]{64}$" .Values.releaseController.image.digest) -}}
{{- fail "releaseController.image.digest must be an immutable sha256 digest" -}}
{{- end -}}
{{- $_ := required "releaseController.controller.tenantId must be configured" .Values.releaseController.controller.tenantId -}}
{{- end -}}
{{- if .Values.supplyChainPolicy.enabled -}}
{{- if not (regexMatch "^[0-9]+\\.[0-9]+\\.[0-9]+$" .Values.supplyChainPolicy.version) -}}
{{- fail "supplyChainPolicy.version must be an exact MAJOR.MINOR.PATCH chart release" -}}
{{- end -}}
{{- $_ := required "supplyChainPolicy.imageRepositoryGlob must be configured" .Values.supplyChainPolicy.imageRepositoryGlob -}}
{{- $_ := required "supplyChainPolicy.certificateIdentityRegexp must be configured" .Values.supplyChainPolicy.certificateIdentityRegexp -}}
{{- if not (hasPrefix "https://" .Values.supplyChainPolicy.certificateOidcIssuer) -}}
{{- fail "supplyChainPolicy.certificateOidcIssuer must use HTTPS" -}}
{{- end -}}
{{- if not (has .Values.supplyChainPolicy.mode (list "enforce" "warn")) -}}
{{- fail "supplyChainPolicy.mode must be enforce or warn" -}}
{{- end -}}
{{- if and (eq .Values.environment "production") (ne .Values.supplyChainPolicy.mode "enforce") -}}
{{- fail "production supply-chain admission must use enforce mode" -}}
{{- end -}}
{{- end -}}
{{- if .Values.observability.enabled -}}
{{- if not (regexMatch "^[0-9]+\\.[0-9]+\\.[0-9]+$" .Values.observability.version) -}}
{{- fail "observability.version must be an exact MAJOR.MINOR.PATCH chart release" -}}
{{- end -}}
{{- $_ := required "observability.grafanaAdminSecretName must reference an external Secret" .Values.observability.grafanaAdminSecretName -}}
{{- if not (regexMatch "^[0-9]+[smhdwy]$" .Values.observability.retention) -}}
{{- fail "observability.retention must be a simple Prometheus duration such as 30d" -}}
{{- end -}}
{{- if not (regexMatch "^[1-9][0-9]*(Mi|Gi|Ti)$" .Values.observability.storageSize) -}}
{{- fail "observability.storageSize must be a positive Kubernetes storage quantity" -}}
{{- end -}}
{{- end -}}
{{- if and (eq .Values.environment "production") (not .Values.observability.enabled) -}}
{{- fail "production must enable the reviewed observability stack" -}}
{{- end -}}
{{- if eq (toString .Values.externalSecrets.enabled) "true" -}}
{{- if ne .Values.externalSecrets.repositoryUrl "https://charts.external-secrets.io" -}}
{{- fail "externalSecrets.repositoryUrl must use the official Helm repository" -}}
{{- end -}}
{{- if ne .Values.externalSecrets.chart "external-secrets" -}}
{{- fail "externalSecrets.chart must be external-secrets" -}}
{{- end -}}
{{- if ne .Values.externalSecrets.version "2.8.0" -}}
{{- fail "externalSecrets.version must remain 2.8.0" -}}
{{- end -}}
{{- $_ := required "externalSecrets.namespace is required" .Values.externalSecrets.namespace -}}
{{- $_ := required "externalSecrets.controllerClass is required" .Values.externalSecrets.controllerClass -}}
{{- $_ := required "externalSecrets.runtime.chartPath is required" .Values.externalSecrets.runtime.chartPath -}}
{{- if ne .Values.externalSecrets.scopedNamespace .Values.application.namespace -}}
{{- fail "externalSecrets.scopedNamespace must equal application.namespace" -}}
{{- end -}}
{{- if ne .Values.externalSecrets.runtime.targetSecretName .Values.application.runtime.secretName -}}
{{- fail "externalSecrets.runtime.targetSecretName must equal application.runtime.secretName" -}}
{{- end -}}
{{- if lt (int .Values.externalSecrets.controllerReplicaCount) 2 -}}
{{- fail "externalSecrets.controllerReplicaCount must be at least two" -}}
{{- end -}}
{{- if lt (int .Values.externalSecrets.webhookReplicaCount) 2 -}}
{{- fail "externalSecrets.webhookReplicaCount must be at least two" -}}
{{- end -}}
{{- if lt (int .Values.externalSecrets.certControllerReplicaCount) 2 -}}
{{- fail "externalSecrets.certControllerReplicaCount must be at least two" -}}
{{- end -}}
{{- if ne .Values.externalSecrets.topologySpread.topologyKey "kubernetes.io/hostname" -}}
{{- fail "externalSecrets.topologySpread.topologyKey must be kubernetes.io/hostname" -}}
{{- end -}}
{{- if lt (int .Values.externalSecrets.topologySpread.minDomains) 2 -}}
{{- fail "externalSecrets.topologySpread.minDomains must be at least two" -}}
{{- end -}}
{{- if lt (int .Values.externalSecrets.topologySpread.maxSkew) 1 -}}
{{- fail "externalSecrets.topologySpread.maxSkew must be positive" -}}
{{- end -}}
{{- if not (hasPrefix "https://" .Values.externalSecrets.runtime.vault.server) -}}
{{- fail "externalSecrets.runtime.vault.server must use HTTPS" -}}
{{- end -}}
{{- if ne .Values.externalSecrets.runtime.vault.version "v2" -}}
{{- fail "externalSecrets.runtime.vault.version must be v2" -}}
{{- end -}}
{{- $_ := required "externalSecrets.runtime.vault.caProvider.name is required" .Values.externalSecrets.runtime.vault.caProvider.name -}}
{{- $_ := required "externalSecrets.runtime.vault.caProvider.key is required" .Values.externalSecrets.runtime.vault.caProvider.key -}}
{{- $_ := required "externalSecrets.runtime.vault.auth.mountPath is required" .Values.externalSecrets.runtime.vault.auth.mountPath -}}
{{- $_ := required "externalSecrets.runtime.vault.auth.role is required" .Values.externalSecrets.runtime.vault.auth.role -}}
{{- if or (ne (len .Values.externalSecrets.runtime.vault.auth.audiences) 1) (ne (index .Values.externalSecrets.runtime.vault.auth.audiences 0) "vault") -}}
{{- fail "externalSecrets.runtime.vault.auth.audiences must contain only vault" -}}
{{- end -}}
{{- $nextAuthBound := false -}}
{{- range .Values.externalSecrets.runtime.mappings -}}
{{- if and (eq .secretKey $.Values.application.runtime.nextAuthSecretKey) (eq .remoteProperty "nextauth_secret") -}}
{{- $nextAuthBound = true -}}
{{- end -}}
{{- end -}}
{{- if not $nextAuthBound -}}
{{- fail "externalSecrets.runtime.mappings must bind application.runtime.nextAuthSecretKey" -}}
{{- end -}}
{{- end -}}
{{- if .Values.recovery.enabled -}}
{{- $_ := required "recovery.reporter.image.repository must be configured" .Values.recovery.reporter.image.repository -}}
{{- if not (regexMatch "^sha256:[0-9a-f]{64}$" .Values.recovery.reporter.image.digest) -}}
{{- fail "recovery.reporter.image.digest must be an immutable sha256 digest" -}}
{{- end -}}
{{- $_ := required "recovery.reporter.clientId must be configured" .Values.recovery.reporter.clientId -}}
{{- $_ := required "recovery.reporter.clientSecret.name must reference an external Secret" .Values.recovery.reporter.clientSecret.name -}}
{{- $_ := required "recovery.reporter.caSecret.name must reference an external CA Secret" .Values.recovery.reporter.caSecret.name -}}
{{- if and (eq .Values.environment "production") (eq (len .Values.recovery.verificationJobs) 0) -}}
{{- fail "production recovery must define at least one reviewed verification job" -}}
{{- end -}}
{{- range .Values.recovery.verificationJobs -}}
{{- if not (regexMatch "@sha256:[0-9a-f]{64}$" .verifier.image) -}}
{{- fail (printf "recovery verifier %s image must be pinned by sha256 digest" .name) -}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- range $name, $component := dict "certManager" .Values.certManager "envoyGateway" .Values.envoyGateway "gpuOperator" .Values.gpuOperator "kserve" .Values.kserve }}
{{- if and $component.enabled (not (regexMatch "^v[0-9]+\\.[0-9]+\\.[0-9]+$" $component.version)) -}}
{{- fail (printf "%s.version must be an exact vMAJOR.MINOR.PATCH release" $name) -}}
{{- end -}}
{{- end -}}
{{- end }}

{{- define "industrial-ops-gitops.applicationMetadata" -}}
finalizers:
  - resources-finalizer.argocd.argoproj.io
labels:
  app.kubernetes.io/part-of: industrial-ops-agent-platform
  industrial-ops.ai/environment: {{ .root.Values.environment | quote }}
  industrial-ops.ai/gitops-tier: {{ .tier | quote }}
annotations:
  argocd.argoproj.io/sync-wave: {{ .wave | quote }}
{{- end }}

{{- define "industrial-ops-gitops.syncPolicy" -}}
automated:
  prune: {{ .Values.syncPolicy.prune }}
  selfHeal: {{ .Values.syncPolicy.selfHeal }}
syncOptions:
  - CreateNamespace=true
  - ServerSideApply=true
  - PruneLast=true
retry:
  limit: {{ .Values.syncPolicy.retryLimit }}
  backoff:
    duration: 10s
    factor: 2
    maxDuration: 3m
{{- end }}
