{{- define "industrial-ops-ai-platform.labels" -}}
app.kubernetes.io/name: industrial-ops-ai-platform
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/instance: {{ .Release.Name }}
industrial-ops.ai/environment: {{ .Values.environment | quote }}
{{- end }}

{{- define "industrial-ops-ai-platform.validate" -}}
{{- if not (regexMatch "^[a-z0-9]([-a-z0-9]*[a-z0-9])?$" .Values.environment) -}}
{{- fail "environment must be a DNS-label-safe value" -}}
{{- end -}}
{{- if .Values.gateway.enabled -}}
{{- $_ := required "gateway.hostname must be configured for the environment" .Values.gateway.hostname -}}
{{- $_ := required "gateway.tlsSecretName must reference an externally managed TLS Secret" .Values.gateway.tlsSecretName -}}
{{- if not (regexMatch (printf "(^|\\.)%s(\\.|$)" .Values.environment) .Values.gateway.hostname) -}}
{{- fail "gateway.hostname must contain the environment as a complete DNS label" -}}
{{- end -}}
{{- end -}}
{{- end }}
