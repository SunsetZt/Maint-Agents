output "gitops_bootstrap" {
  description = "Non-sensitive infrastructure identifiers consumed by the separately owned GitOps bootstrap workflow."
  # credential_contract = "ARNs, IDs and endpoints only; resolve secret values at runtime through authorized workload identity."
  value = {
    cluster_name        = module.eks.cluster_name
    cluster_arn         = module.eks.cluster_arn
    cluster_endpoint    = module.eks.cluster_endpoint
    oidc_provider_arn   = module.eks.oidc_provider_arn
    oidc_provider_url   = module.eks.cluster_oidc_issuer_url
    vpc_id              = module.vpc.vpc_id
    private_subnet_ids  = module.vpc.private_subnets
    database_endpoint   = module.database.db_instance_endpoint
    database_secret_arn = module.database.db_instance_master_user_secret_arn

    data_kms_key_arn    = module.data_kms.key_arn
    artifact_bucket_arn = module.artifact_bucket.s3_bucket_arn
    audit_bucket_arn    = module.audit_bucket.s3_bucket_arn
    ecr_repository_urls = {
      for name, repository in aws_ecr_repository.platform : name => repository.repository_url
    }
    hosted_zone_id                 = var.hosted_zone_id
    gateway_dns_fqdns              = [for record in aws_route53_record.gateway : record.fqdn]
    platform_pod_identity_role_arn = aws_iam_role.platform_pod_identity.arn
    credential_contract            = "ARNs, IDs and endpoints only; resolve secret values at runtime through authorized workload identity."
    infrastructure_owner           = "Terraform owns AWS infrastructure; Helm and Argo CD own Kubernetes applications."
  }
}
