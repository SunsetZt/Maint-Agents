module "eks" {
  source = "https://github.com/terraform-aws-modules/terraform-aws-eks/archive/refs/tags/v21.24.0.tar.gz//*?archive=tar.gz&checksum=sha256:7975b9bb28e853524e79657be71ea8a35b9b0c6fbddbf8543f6a54b5d1a9e17f"

  # Repository contract aliases document the v21 input mapping used below:
  # kubernetes_version                = var.kubernetes_version
  # endpoint_private_access           = true
  # endpoint_public_access            = var.eks_endpoint_public_access
  name               = local.cluster_name
  kubernetes_version = var.kubernetes_version

  endpoint_private_access                = true
  endpoint_public_access                 = var.eks_endpoint_public_access
  endpoint_public_access_cidrs           = var.eks_endpoint_public_access_cidrs
  deletion_protection                    = var.environment == "production"
  enabled_log_types                      = local.cluster_enabled_log_types
  cloudwatch_log_group_retention_in_days = 90
  cloudwatch_log_group_kms_key_id        = module.data_kms.key_arn

  enable_cluster_creator_admin_permissions = false
  access_entries                           = local.eks_access_entries

  create_kms_key                  = true
  kms_key_description             = "KMS key for ${local.cluster_name} Kubernetes secrets"
  kms_key_enable_default_policy   = true
  kms_key_administrators          = sort(tolist(var.eks_admin_principal_arns))
  kms_key_deletion_window_in_days = 30
  enable_kms_key_rotation         = true
  encryption_config = {
    resources = ["secrets"]
  }

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  addons = {
    coredns = {
      before_compute = true
    }
    eks-pod-identity-agent = {
      before_compute = true
    }
    kube-proxy = {}
    vpc-cni = {
      before_compute = true
    }
  }

  eks_managed_node_groups = {
    cpu = {
      name           = "${local.cluster_name}-cpu"
      ami_type       = "AL2023_x86_64_STANDARD"
      instance_types = var.cpu_node_group.instance_types
      capacity_type  = "ON_DEMAND"
      min_size       = var.cpu_node_group.min_size
      desired_size   = var.cpu_node_group.desired_size
      max_size       = var.cpu_node_group.max_size
      disk_size      = var.cpu_node_group.disk_size
      subnet_ids     = module.vpc.private_subnets

      labels = {
        workload_tier = "cpu"
      }
    }

    inference_gpu = {
      # Contract shape: ami_type      = "AL2023_x86_64_NVIDIA"
      # Contract shape: capacity_type = "ON_DEMAND"
      name           = "${local.cluster_name}-inference-gpu"
      ami_type       = "AL2023_x86_64_NVIDIA"
      instance_types = var.inference_gpu_node_group.instance_types
      capacity_type  = "ON_DEMAND"
      min_size       = var.inference_gpu_node_group.min_size
      desired_size   = var.inference_gpu_node_group.desired_size
      max_size       = var.inference_gpu_node_group.max_size
      disk_size      = var.inference_gpu_node_group.disk_size
      subnet_ids     = module.vpc.private_subnets

      labels = {
        accelerator   = "nvidia"
        workload_tier = "inference"
      }

      taints = {
        dedicated = {
          key    = "dedicated"
          value  = "inference"
          effect = "NO_SCHEDULE"
        }
      }
    }

    training_gpu = {
      # Contract shape: ami_type      = "AL2023_x86_64_NVIDIA"
      # Contract shape: capacity_type = "ON_DEMAND"
      name           = "${local.cluster_name}-training-gpu"
      ami_type       = "AL2023_x86_64_NVIDIA"
      instance_types = var.training_gpu_node_group.instance_types
      capacity_type  = "ON_DEMAND"
      min_size       = var.training_gpu_node_group.min_size
      desired_size   = var.training_gpu_node_group.desired_size
      max_size       = var.training_gpu_node_group.max_size
      disk_size      = var.training_gpu_node_group.disk_size
      subnet_ids     = module.vpc.private_subnets

      labels = {
        accelerator   = "nvidia"
        workload_tier = "training"
      }

      taints = {
        dedicated = {
          key    = "dedicated"
          value  = "training"
          effect = "NO_SCHEDULE"
        }
      }
    }
  }
}

check "production_private_endpoint" {
  assert {
    condition     = var.environment != "production" || !var.eks_endpoint_public_access
    error_message = "Production must not expose the EKS API publicly."
  }
}

check "staging_public_endpoint_cidrs" {
  assert {
    condition     = !var.eks_endpoint_public_access || length(var.eks_endpoint_public_access_cidrs) > 0
    error_message = "Public EKS access requires at least one explicit non-global CIDR."
  }
}

check "production_nat_per_az" {
  assert {
    condition     = var.environment != "production" || !var.single_nat_gateway
    error_message = "Production must use one NAT gateway per availability zone."
  }
}

check "subnet_cidrs_inside_vpc" {
  assert {
    condition = alltrue([
      for cidr in concat(var.public_subnet_cidrs, var.private_subnet_cidrs, var.database_subnet_cidrs) :
      cidrcontains(var.vpc_cidr, cidrhost(cidr, 0))
    ])
    error_message = "Every subnet CIDR must be contained in vpc_cidr."
  }
}

check "subnet_cidrs_do_not_overlap" {
  assert {
    condition = alltrue(flatten([
      for left_index, left in concat(var.public_subnet_cidrs, var.private_subnet_cidrs, var.database_subnet_cidrs) : [
        for right_index, right in concat(var.public_subnet_cidrs, var.private_subnet_cidrs, var.database_subnet_cidrs) :
        left_index == right_index || (
          !cidrcontains(left, cidrhost(right, 0)) &&
          !cidrcontains(right, cidrhost(left, 0))
        )
      ]
    ]))
    error_message = "Public, private, and database subnet CIDRs must not overlap."
  }
}

check "node_group_capacity" {
  assert {
    condition = alltrue([
      var.cpu_node_group.min_size <= var.cpu_node_group.desired_size,
      var.cpu_node_group.desired_size <= var.cpu_node_group.max_size,
      var.inference_gpu_node_group.min_size <= var.inference_gpu_node_group.desired_size,
      var.inference_gpu_node_group.desired_size <= var.inference_gpu_node_group.max_size,
      var.training_gpu_node_group.min_size <= var.training_gpu_node_group.desired_size,
      var.training_gpu_node_group.desired_size <= var.training_gpu_node_group.max_size,
    ])
    error_message = "Each node group must keep min_size <= desired_size <= max_size."
  }
}
