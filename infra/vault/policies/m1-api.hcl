path "secret/data/industrial-ops/m1" {
  capabilities = ["read"]
}

path "secret/metadata/industrial-ops/m1" {
  capabilities = ["read"]
}

path "auth/token/renew-self" {
  capabilities = ["update"]
}
