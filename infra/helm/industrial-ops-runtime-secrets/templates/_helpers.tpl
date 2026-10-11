{{- define "industrial-ops-runtime-secrets.labels" -}}
app.kubernetes.io/name: industrial-ops-runtime-secrets
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/part-of: industrial-ops-agent-platform
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | quote }}
{{- end }}

{{- define "industrial-ops-runtime-secrets.validateDnsName" -}}
{{- if or (gt (len .value) 253) (not (regexMatch "^[a-z0-9]([-a-z0-9.]*[a-z0-9])?$" .value)) -}}
{{- fail (printf "%s must be a valid Kubernetes DNS name" .field) -}}
{{- end -}}
{{- end }}

{{- define "industrial-ops-runtime-secrets.validate" -}}
{{- $root := . -}}
{{- range $field, $value := dict
      "controllerClass" .Values.controllerClass
      "serviceAccount.name" .Values.serviceAccount.name
      "secretStore.name" .Values.secretStore.name
      "externalSecret.name" .Values.externalSecret.name
      "target.name" .Values.target.name -}}
  {{- if empty $value -}}{{- fail (printf "%s is required" $field) -}}{{- end -}}
  {{- include "industrial-ops-runtime-secrets.validateDnsName" (dict "field" $field "value" $value) -}}
{{- end -}}

{{- if not (regexMatch "^https://[A-Za-z0-9.-]+(:[0-9]+)?$" .Values.vault.server) -}}
{{- fail "vault.server must use https and contain only a reviewed host and optional port" -}}
{{- end -}}
{{- if ne .Values.vault.version "v2" -}}{{- fail "vault.version must be v2" -}}{{- end -}}
{{- if not (regexMatch "^[a-z0-9][a-z0-9_-]*$" .Values.vault.path) -}}
{{- fail "vault.path must be a single KV mount name" -}}
{{- end -}}
{{- if not (has .Values.vault.caProvider.type (list "ConfigMap" "Secret")) -}}
{{- fail "vault.caProvider.type must be ConfigMap or Secret" -}}
{{- end -}}
{{- range $field, $value := dict
      "vault.caProvider.name" .Values.vault.caProvider.name
      "vault.caProvider.key" .Values.vault.caProvider.key
      "vault.auth.mountPath" .Values.vault.auth.mountPath
      "vault.auth.role" .Values.vault.auth.role -}}
  {{- if empty $value -}}{{- fail (printf "%s is required" $field) -}}{{- end -}}
{{- end -}}
{{- include "industrial-ops-runtime-secrets.validateDnsName" (dict "field" "vault.caProvider.name" "value" .Values.vault.caProvider.name) -}}
{{- if not (regexMatch "^[a-z0-9][a-z0-9_-]*$" .Values.vault.auth.mountPath) -}}
{{- fail "vault.auth.mountPath must be a single auth mount name" -}}
{{- end -}}
{{- if or (ne (len .Values.vault.auth.audiences) 1) (ne (index .Values.vault.auth.audiences 0) "vault") -}}
{{- fail "vault.auth.audiences must contain only vault" -}}
{{- end -}}

{{- if or (empty .Values.remote.key) (hasPrefix "/" .Values.remote.key) (contains ".." .Values.remote.key) (not (regexMatch "^[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)*$" .Values.remote.key)) -}}
{{- fail "remote.key must be a relative path without traversal" -}}
{{- end -}}
{{- if ne .Values.refreshPolicy "Periodic" -}}{{- fail "refreshPolicy must be Periodic" -}}{{- end -}}
{{- if not (regexMatch "^[1-9][0-9]*(s|m|h)$" .Values.refreshInterval) -}}
{{- fail "refreshInterval must be a positive duration using s, m or h" -}}
{{- end -}}
{{- if ne .Values.lifecycle.creationPolicy "Orphan" -}}{{- fail "lifecycle.creationPolicy must be Orphan" -}}{{- end -}}
{{- if ne .Values.lifecycle.deletionPolicy "Retain" -}}{{- fail "lifecycle.deletionPolicy must be Retain" -}}{{- end -}}

{{- $expected := dict
      "IOAP_DATABASE_URL" "database_url"
      "IOAP_OIDC_CLIENT_SECRET" "oidc_client_secret"
      "IOAP_MINIO_ACCESS_KEY" "minio_access_key"
      "IOAP_MINIO_SECRET_KEY" "minio_secret_key"
      "IOAP_NEXTAUTH_SECRET" "nextauth_secret"
      "nextauth-secret" "nextauth_secret" -}}
{{- if gt (len .Values.mappings) (len $expected) -}}
{{- fail "mappings must contain exactly the reviewed six entries" -}}
{{- end -}}
{{- if lt (len .Values.mappings) (len $expected) -}}
{{- fail "mappings must match the reviewed runtime key contract" -}}
{{- end -}}
{{- $actual := dict -}}
{{- range $mapping := .Values.mappings -}}
  {{- $secretKey := required "mappings[].secretKey is required" $mapping.secretKey -}}
  {{- $remoteProperty := required "mappings[].remoteProperty is required" $mapping.remoteProperty -}}
  {{- if not (regexMatch "^[A-Za-z0-9._-]+$" $secretKey) -}}{{- fail "mappings[].secretKey is invalid" -}}{{- end -}}
  {{- if not (regexMatch "^[a-z][a-z0-9_]*$" $remoteProperty) -}}{{- fail "mappings[].remoteProperty must use lower snake case" -}}{{- end -}}
  {{- if hasKey $actual $secretKey -}}{{- fail "mappings[].secretKey must be unique" -}}{{- end -}}
  {{- $_ := set $actual $secretKey $remoteProperty -}}
{{- end -}}
{{- range $secretKey, $remoteProperty := $expected -}}
  {{- if or (not (hasKey $actual $secretKey)) (ne (get $actual $secretKey) $remoteProperty) -}}
  {{- fail "mappings must match the reviewed runtime key contract" -}}
  {{- end -}}
{{- end -}}
{{- end }}
