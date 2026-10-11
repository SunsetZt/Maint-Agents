{{- define "industrial-ops-application.labels" -}}
app.kubernetes.io/part-of: industrial-ops-agent-platform
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | quote }}
{{- end }}

{{- define "industrial-ops-application.selectorLabels" -}}
app.kubernetes.io/name: {{ printf "industrial-ops-%s" .component }}
app.kubernetes.io/instance: {{ .root.Release.Name }}
{{- end }}

{{- define "industrial-ops-application.workloadLabels" -}}
{{ include "industrial-ops-application.selectorLabels" . }}
app.kubernetes.io/component: {{ .component }}
{{- end }}

{{- define "industrial-ops-application.validateWorkload" -}}
{{- $root := .root -}}
{{- $haEnabled := eq (toString $root.Values.highAvailability.enabled) "true" -}}
{{- $workload := .workload -}}
{{- $image := .image -}}
{{- $name := .name -}}
{{- if $workload.enabled -}}
  {{- if lt (int $workload.replicaCount) 1 -}}
    {{- fail (printf "workloads.%s.replicaCount must be at least one" $name) -}}
  {{- end -}}
  {{- if and $haEnabled (lt (int $workload.replicaCount) 2) -}}
    {{- fail (printf "workloads.%s: high availability requires at least two replicas" $name) -}}
  {{- end -}}
  {{- if and $haEnabled (lt (int $workload.podDisruptionBudget.minAvailable) 1) -}}
    {{- fail (printf "workloads.%s.podDisruptionBudget.minAvailable must be at least one" $name) -}}
  {{- end -}}
  {{- if and $haEnabled (gt (int $workload.podDisruptionBudget.minAvailable) (int $workload.replicaCount)) -}}
    {{- fail (printf "workloads.%s.podDisruptionBudget.minAvailable cannot exceed replicaCount" $name) -}}
  {{- end -}}
  {{- if empty $image.repository -}}
    {{- fail (printf "images.%s.repository is required" $name) -}}
  {{- end -}}
  {{- if not (regexMatch "^sha256:[0-9a-f]{64}$" $image.digest) -}}
    {{- fail (printf "images.%s.digest must match sha256:<64 lowercase hex characters>" $name) -}}
  {{- end -}}
{{- end -}}
{{- end }}

{{- define "industrial-ops-application.validateHpa" -}}
{{- $root := .root -}}
{{- $name := .name -}}
{{- $config := .config -}}
{{- if lt (int $config.minReplicas) 1 -}}
  {{- fail (printf "autoscaling.hpa.%s.minReplicas must be at least one" $name) -}}
{{- end -}}
{{- if and (eq (toString $root.Values.highAvailability.enabled) "true") (lt (int $config.minReplicas) 2) -}}
  {{- fail (printf "autoscaling.hpa.%s high availability requires minReplicas at least two" $name) -}}
{{- end -}}
{{- if lt (int $config.maxReplicas) (int $config.minReplicas) -}}
  {{- fail (printf "autoscaling.hpa.%s.maxReplicas cannot be less than minReplicas" $name) -}}
{{- end -}}
{{- if or (lt (int $config.targetCPUUtilizationPercentage) 1) (gt (int $config.targetCPUUtilizationPercentage) 100) -}}
  {{- fail (printf "autoscaling.hpa.%s.targetCPUUtilizationPercentage must be between 1 and 100" $name) -}}
{{- end -}}
{{- if or (lt (int $config.targetMemoryUtilizationPercentage) 1) (gt (int $config.targetMemoryUtilizationPercentage) 100) -}}
  {{- fail (printf "autoscaling.hpa.%s.targetMemoryUtilizationPercentage must be between 1 and 100" $name) -}}
{{- end -}}
{{- end }}

{{- define "industrial-ops-application.validateKeda" -}}
{{- $root := .root -}}
{{- $name := .name -}}
{{- $config := .config -}}
{{- if lt (int $config.minReplicaCount) 1 -}}
  {{- fail (printf "autoscaling.keda.%s.minReplicaCount must be at least one" $name) -}}
{{- end -}}
{{- if and (eq (toString $root.Values.highAvailability.enabled) "true") (lt (int $config.minReplicaCount) 2) -}}
  {{- fail (printf "autoscaling.keda.%s high availability requires minReplicaCount at least two" $name) -}}
{{- end -}}
{{- if lt (int $config.maxReplicaCount) (int $config.minReplicaCount) -}}
  {{- fail (printf "autoscaling.keda.%s.maxReplicaCount cannot be less than minReplicaCount" $name) -}}
{{- end -}}
{{- if lt (int $config.fallback.failureThreshold) 1 -}}
  {{- fail (printf "autoscaling.keda.%s.fallback.failureThreshold must be positive" $name) -}}
{{- end -}}
{{- if lt (int $config.fallback.replicas) (int $config.minReplicaCount) -}}
  {{- fail (printf "autoscaling.keda.%s fallback cannot be below minReplicaCount" $name) -}}
{{- end -}}
{{- end }}

{{- define "industrial-ops-application.validate" -}}
{{- $haEnabled := eq (toString .Values.highAvailability.enabled) "true" -}}
{{- if empty .Values.runtime.configMapName -}}
  {{- fail "runtime.configMapName is required" -}}
{{- end -}}
{{- if empty .Values.runtime.secretName -}}
  {{- fail "runtime.secretName is required" -}}
{{- end -}}
{{- if empty .Values.runtime.nextAuthSecretKey -}}
  {{- fail "runtime.nextAuthSecretKey is required" -}}
{{- end -}}
{{- if $haEnabled -}}
  {{- if lt (int .Values.topologySpread.minDomains) 2 -}}
    {{- fail "topologySpread.minDomains must be at least two" -}}
  {{- end -}}
  {{- if ne .Values.topologySpread.topologyKey "kubernetes.io/hostname" -}}
    {{- fail "topologySpread.topologyKey must be kubernetes.io/hostname" -}}
  {{- end -}}
  {{- if lt (int .Values.topologySpread.maxSkew) 1 -}}
    {{- fail "topologySpread.maxSkew must be at least one" -}}
  {{- end -}}
{{- end -}}
{{- include "industrial-ops-application.validateWorkload" (dict "root" . "name" "api" "workload" .Values.workloads.api "image" .Values.images.api) -}}
{{- include "industrial-ops-application.validateWorkload" (dict "root" . "name" "web" "workload" .Values.workloads.web "image" .Values.images.web) -}}
{{- include "industrial-ops-application.validateWorkload" (dict "root" . "name" "mediaWorker" "workload" .Values.workloads.mediaWorker "image" .Values.images.mediaWorker) -}}
{{- include "industrial-ops-application.validateWorkload" (dict "root" . "name" "workflowWorker" "workload" .Values.workloads.workflowWorker "image" .Values.images.workflowWorker) -}}
{{- include "industrial-ops-application.validateWorkload" (dict "root" . "name" "eventWorker" "workload" .Values.workloads.eventWorker "image" .Values.images.eventWorker) -}}
{{- if eq (toString .Values.autoscaling.enabled) "true" -}}
  {{- range $name, $workload := .Values.workloads -}}
    {{- if not (eq (toString $workload.enabled) "true") -}}
      {{- fail (printf "autoscaling requires workloads.%s to be enabled" $name) -}}
    {{- end -}}
  {{- end -}}
  {{- include "industrial-ops-application.validateHpa" (dict "root" . "name" "api" "config" .Values.autoscaling.hpa.api) -}}
  {{- include "industrial-ops-application.validateHpa" (dict "root" . "name" "web" "config" .Values.autoscaling.hpa.web) -}}
  {{- include "industrial-ops-application.validateHpa" (dict "root" . "name" "mediaWorker" "config" .Values.autoscaling.hpa.mediaWorker) -}}
  {{- include "industrial-ops-application.validateKeda" (dict "root" . "name" "workflowWorker" "config" .Values.autoscaling.keda.workflowWorker) -}}
  {{- include "industrial-ops-application.validateKeda" (dict "root" . "name" "eventWorker" "config" .Values.autoscaling.keda.eventWorker) -}}
  {{- if lt (int .Values.autoscaling.behavior.scaleDown.stabilizationWindowSeconds) 300 -}}
    {{- fail "autoscaling.behavior.scaleDown.stabilizationWindowSeconds must be at least 300" -}}
  {{- end -}}
  {{- if lt (int .Values.autoscaling.keda.pollingInterval) 5 -}}
    {{- fail "autoscaling.keda.pollingInterval must be at least five seconds" -}}
  {{- end -}}
  {{- if lt (int .Values.autoscaling.keda.cooldownPeriod) 60 -}}
    {{- fail "autoscaling.keda.cooldownPeriod must be at least sixty seconds" -}}
  {{- end -}}
  {{- $temporal := .Values.autoscaling.keda.workflowWorker.temporal -}}
  {{- if empty $temporal.endpoint -}}{{- fail "autoscaling.keda.workflowWorker.temporal.endpoint is required" -}}{{- end -}}
  {{- if empty $temporal.namespace -}}{{- fail "autoscaling.keda.workflowWorker.temporal.namespace is required" -}}{{- end -}}
  {{- if empty $temporal.taskQueue -}}{{- fail "autoscaling.keda.workflowWorker.temporal.taskQueue is required" -}}{{- end -}}
  {{- if not (has $temporal.queueTypes (list "workflow" "activity" "workflow,activity" "activity,workflow")) -}}
    {{- fail "autoscaling.keda.workflowWorker.temporal.queueTypes is invalid" -}}
  {{- end -}}
  {{- if lt (int $temporal.targetQueueSize) 1 -}}
    {{- fail "autoscaling.keda.workflowWorker.temporal.targetQueueSize must be positive" -}}
  {{- end -}}
  {{- if lt (int $temporal.activationTargetQueueSize) 0 -}}
    {{- fail "autoscaling.keda.workflowWorker.temporal.activationTargetQueueSize cannot be negative" -}}
  {{- end -}}
  {{- $kafka := .Values.autoscaling.keda.eventWorker.kafka -}}
  {{- if empty $kafka.bootstrapServers -}}{{- fail "autoscaling.keda.eventWorker.kafka.bootstrapServers is required" -}}{{- end -}}
  {{- if empty $kafka.topic -}}{{- fail "autoscaling.keda.eventWorker.kafka.topic is required" -}}{{- end -}}
  {{- if empty $kafka.consumerGroup -}}{{- fail "autoscaling.keda.eventWorker.kafka.consumerGroup is required" -}}{{- end -}}
  {{- if lt (int $kafka.lagThreshold) 1 -}}
    {{- fail "autoscaling.keda.eventWorker.kafka.lagThreshold must be positive" -}}
  {{- end -}}
  {{- if lt (int $kafka.activationLagThreshold) 0 -}}
    {{- fail "autoscaling.keda.eventWorker.kafka.activationLagThreshold cannot be negative" -}}
  {{- end -}}
  {{- if not (has $kafka.offsetResetPolicy (list "latest" "earliest")) -}}
    {{- fail "autoscaling.keda.eventWorker.kafka.offsetResetPolicy is invalid" -}}
  {{- end -}}
{{- end -}}
{{- if .Values.migration.enabled -}}
  {{- if empty .Values.images.api.repository -}}
    {{- fail "images.api.repository is required for migration" -}}
  {{- end -}}
  {{- if not (regexMatch "^sha256:[0-9a-f]{64}$" .Values.images.api.digest) -}}
    {{- fail "images.api.digest must match sha256:<64 lowercase hex characters>" -}}
  {{- end -}}
{{- end -}}
{{- if .Values.gateway.enabled -}}
  {{- if empty .Values.gateway.className -}}
    {{- fail "gateway.className is required when gateway is enabled" -}}
  {{- end -}}
  {{- if empty .Values.gateway.hostname -}}
    {{- fail "gateway.hostname is required when gateway is enabled" -}}
  {{- end -}}
  {{- if empty .Values.gateway.tlsSecretName -}}
    {{- fail "gateway.tlsSecretName is required when gateway is enabled" -}}
  {{- end -}}
{{- end -}}
{{- end }}
