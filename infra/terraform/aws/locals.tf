locals {
  cluster_name = "${var.name_prefix}-${var.environment}-eks"

  cluster_enabled_log_types = ["api", "audit", "authenticator", "controllerManager", "scheduler"]

  common_tags = merge(var.additional_tags, {
    Project            = "enterprise-multimodal-industrial-ops-agent-platform"
    Environment        = var.environment
    Owner              = var.owner
    CostCenter         = var.cost_center
    DataClassification = var.data_classification
    ManagedBy          = "terraform"
  })

  eks_access_entries = {
    for index, principal_arn in sort(tolist(var.eks_admin_principal_arns)) :
    "platform-admin-${index}" => {
      principal_arn = principal_arn
      policy_associations = {
        cluster_admin = {
          policy_arn = "arn:${data.aws_partition.current.partition}:eks::aws:cluster-access-policy/AmazonEKSClusterAdminPolicy"
          access_scope = {
            type = "cluster"
          }
        }
      }
    }
  }
}
