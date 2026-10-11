{{- define "industrial-ops-release-controller.name" -}}
industrial-ops-release-controller
{{- end -}}

{{- define "industrial-ops-release-controller.labels" -}}
app.kubernetes.io/name: {{ include "industrial-ops-release-controller.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}
