variable "environment" {
  description = "Closed deployment environment. Each environment must use a separate AWS account and state."
  type        = string

  validation {
    condition     = contains(["staging", "production"], var.environment)
    error_message = "environment must be staging or production."
  }
}

variable "aws_region" {
  description = "AWS Region selected by the platform owner."
  type        = string

  validation {
    condition     = can(regex("^[a-z]{2}(?:-gov)?-[a-z]+-[0-9]$", var.aws_region))
    error_message = "aws_region must be a valid AWS Region name."
  }
}

variable "allowed_account_ids" {
  description = "Explicit allowlist of 12-digit AWS account IDs permitted to use this root module."
  type        = list(string)

  validation {
    condition = (
      length(var.allowed_account_ids) > 0 &&
      length(var.allowed_account_ids) == length(distinct(var.allowed_account_ids)) &&
      alltrue([for account_id in var.allowed_account_ids : can(regex("^[0-9]{12}$", account_id))])
    )
    error_message = "allowed_account_ids must contain unique 12-digit AWS account IDs."
  }
}

variable "name_prefix" {
  description = "Short enterprise-owned resource prefix."
  type        = string

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{2,20}[a-z0-9]$", var.name_prefix))
    error_message = "name_prefix must be 4-22 lowercase letters, digits, or hyphens."
  }
}

variable "owner" {
  description = "Accountable platform team written into default AWS tags."
  type        = string

  validation {
    condition     = length(trimspace(var.owner)) >= 3
    error_message = "owner must identify the accountable platform team."
  }
}

variable "cost_center" {
  description = "Enterprise cost allocation identifier written into default AWS tags."
  type        = string

  validation {
    condition     = can(regex("^[A-Za-z0-9_-]{3,32}$", var.cost_center))
    error_message = "cost_center must be a 3-32 character enterprise identifier."
  }
}

variable "data_classification" {
  description = "Enterprise data classification written into default AWS tags."
  type        = string

  validation {
    condition     = contains(["internal", "confidential", "restricted"], var.data_classification)
    error_message = "data_classification must be internal, confidential, or restricted."
  }
}

variable "additional_tags" {
  description = "Additional non-authoritative enterprise tags. Reserved contract tags cannot be overridden."
  type        = map(string)
  default     = {}

  validation {
    condition = length(setintersection(
      toset(keys(var.additional_tags)),
      toset(["Project", "Environment", "Owner", "CostCenter", "DataClassification", "ManagedBy"])
    )) == 0
    error_message = "additional_tags cannot override the governed contract tags."
  }
}

variable "vpc_cidr" {
  description = "RFC1918 CIDR for the environment VPC."
  type        = string

  validation {
    condition = can(cidrhost(var.vpc_cidr, 0)) && (
      startswith(var.vpc_cidr, "10.") ||
      can(regex("^172\\.(?:1[6-9]|2[0-9]|3[01])\\.", var.vpc_cidr)) ||
      startswith(var.vpc_cidr, "192.168.")
    )
    error_message = "vpc_cidr must be a valid RFC1918 IPv4 CIDR."
  }
}

variable "availability_zones" {
  description = "Exactly three distinct availability zones in aws_region."
  type        = list(string)

  validation {
    condition = (
      length(var.availability_zones) == 3 &&
      length(distinct(var.availability_zones)) == 3 &&
      alltrue([for zone in var.availability_zones : startswith(zone, var.aws_region)])
    )
    error_message = "availability_zones must contain three distinct zones from aws_region."
  }
}

variable "public_subnet_cidrs" {
  description = "Three distinct public subnet CIDRs used only by per-AZ NAT gateways."
  type        = list(string)

  validation {
    condition = (
      length(var.public_subnet_cidrs) == 3 &&
      length(distinct(var.public_subnet_cidrs)) == 3 &&
      alltrue([for cidr in var.public_subnet_cidrs : can(cidrhost(cidr, 0))])
    )
    error_message = "public_subnet_cidrs must contain three distinct IPv4 CIDRs."
  }
}

variable "private_subnet_cidrs" {
  description = "Three distinct private subnet CIDRs for EKS nodes and platform workloads."
  type        = list(string)

  validation {
    condition = (
      length(var.private_subnet_cidrs) == 3 &&
      length(distinct(var.private_subnet_cidrs)) == 3 &&
      alltrue([for cidr in var.private_subnet_cidrs : can(cidrhost(cidr, 0))])
    )
    error_message = "private_subnet_cidrs must contain three distinct IPv4 CIDRs."
  }
}

variable "database_subnet_cidrs" {
  description = "Three distinct isolated subnet CIDRs reserved for managed databases."
  type        = list(string)

  validation {
    condition = (
      length(var.database_subnet_cidrs) == 3 &&
      length(distinct(var.database_subnet_cidrs)) == 3 &&
      alltrue([for cidr in var.database_subnet_cidrs : can(cidrhost(cidr, 0))])
    )
    error_message = "database_subnet_cidrs must contain three distinct IPv4 CIDRs."
  }
}

variable "single_nat_gateway" {
  description = "Cost-reduction option for staging only. Production always requires one NAT gateway per AZ."
  type        = bool
  default     = false
}

variable "vpc_flow_log_retention_days" {
  description = "CloudWatch retention for VPC flow logs."
  type        = number
  default     = 90

  validation {
    condition     = contains([30, 60, 90, 120, 180, 365, 400, 545, 731, 1096, 1827, 2192, 2557, 2922, 3288, 3653], var.vpc_flow_log_retention_days)
    error_message = "vpc_flow_log_retention_days must be an AWS-supported governed retention value of at least 30 days."
  }
}

variable "kubernetes_version" {
  description = "Reviewed EKS Kubernetes version. Changing this value requires an upgrade review."
  type        = string
  default     = "1.35"

  validation {
    condition     = var.kubernetes_version == "1.35"
    error_message = "kubernetes_version must remain on the reviewed 1.35 baseline."
  }
}

variable "eks_endpoint_public_access" {
  description = "Whether to expose the EKS API publicly. Production is always rejected when true."
  type        = bool
  default     = false
}

variable "eks_endpoint_public_access_cidrs" {
  description = "Explicit non-global CIDRs permitted only when staging public EKS access is enabled."
  type        = list(string)
  default     = []

  validation {
    condition = (
      !contains(var.eks_endpoint_public_access_cidrs, "0.0.0.0/0") &&
      length(var.eks_endpoint_public_access_cidrs) == length(distinct(var.eks_endpoint_public_access_cidrs)) &&
      alltrue([for cidr in var.eks_endpoint_public_access_cidrs : can(cidrhost(cidr, 0))])
    )
    error_message = "eks_endpoint_public_access_cidrs must be unique valid CIDRs and cannot include 0.0.0.0/0."
  }
}

variable "eks_admin_principal_arns" {
  description = "Explicit IAM role ARNs granted cluster administrator access through EKS Access Entries."
  type        = set(string)

  validation {
    condition = (
      length(var.eks_admin_principal_arns) > 0 &&
      alltrue([for arn in var.eks_admin_principal_arns : can(regex("^arn:[a-z0-9-]+:iam::[0-9]{12}:role/.+$", arn))])
    )
    error_message = "eks_admin_principal_arns must contain at least one IAM role ARN."
  }
}

variable "cpu_node_group" {
  description = "Capacity contract for CPU platform and Agent workloads."
  type = object({
    instance_types = list(string)
    min_size       = number
    desired_size   = number
    max_size       = number
    disk_size      = number
  })

  validation {
    condition = (
      length(var.cpu_node_group.instance_types) > 0 &&
      var.cpu_node_group.min_size >= 1 &&
      var.cpu_node_group.desired_size >= var.cpu_node_group.min_size &&
      var.cpu_node_group.max_size >= var.cpu_node_group.desired_size &&
      var.cpu_node_group.disk_size >= 50
    )
    error_message = "cpu_node_group must have instance types, ordered capacity, and at least 50 GiB disk."
  }
}

variable "inference_gpu_node_group" {
  description = "Capacity contract reserved for online GPU inference."
  type = object({
    instance_types = list(string)
    min_size       = number
    desired_size   = number
    max_size       = number
    disk_size      = number
  })

  validation {
    condition = (
      length(var.inference_gpu_node_group.instance_types) > 0 &&
      var.inference_gpu_node_group.min_size >= 1 &&
      var.inference_gpu_node_group.desired_size >= var.inference_gpu_node_group.min_size &&
      var.inference_gpu_node_group.max_size >= var.inference_gpu_node_group.desired_size &&
      var.inference_gpu_node_group.disk_size >= 100
    )
    error_message = "inference_gpu_node_group must have instance types, ordered capacity, and at least 100 GiB disk."
  }
}

variable "training_gpu_node_group" {
  description = "Capacity contract reserved for governed model training and post-training jobs."
  type = object({
    instance_types = list(string)
    min_size       = number
    desired_size   = number
    max_size       = number
    disk_size      = number
  })

  validation {
    condition = (
      length(var.training_gpu_node_group.instance_types) > 0 &&
      var.training_gpu_node_group.min_size >= 0 &&
      var.training_gpu_node_group.desired_size >= var.training_gpu_node_group.min_size &&
      var.training_gpu_node_group.max_size >= max(1, var.training_gpu_node_group.desired_size) &&
      var.training_gpu_node_group.disk_size >= 200
    )
    error_message = "training_gpu_node_group must have instance types, ordered capacity, and at least 200 GiB disk."
  }
}

variable "kms_admin_principal_arns" {
  description = "Explicit IAM role ARNs permitted to administer the customer-managed data KMS key."
  type        = set(string)

  validation {
    condition = (
      length(var.kms_admin_principal_arns) > 0 &&
      alltrue([for arn in var.kms_admin_principal_arns : can(regex("^arn:[a-z0-9-]+:iam::[0-9]{12}:role/.+$", arn))])
    )
    error_message = "kms_admin_principal_arns must contain at least one IAM role ARN."
  }
}

variable "database_name" {
  description = "Initial PostgreSQL database name."
  type        = string
  default     = "industrialops"

  validation {
    condition     = can(regex("^[A-Za-z][A-Za-z0-9_]{2,62}$", var.database_name))
    error_message = "database_name must be a valid PostgreSQL identifier of 3-63 characters."
  }
}

variable "database_master_username" {
  description = "RDS master username; the corresponding credential is generated and managed by RDS Secrets Manager."
  type        = string
  default     = "industrialopsadmin"

  validation {
    condition     = can(regex("^[A-Za-z][A-Za-z0-9_]{2,62}$", var.database_master_username))
    error_message = "database_master_username must be a valid PostgreSQL identifier of 3-63 characters."
  }
}

variable "database_instance_class" {
  description = "Reviewed RDS PostgreSQL instance class for the target environment."
  type        = string

  validation {
    condition     = can(regex("^db\\.[a-z0-9]+\\.[a-z0-9]+$", var.database_instance_class))
    error_message = "database_instance_class must be an explicit RDS DB instance class."
  }
}

variable "database_allocated_storage" {
  description = "Initial encrypted gp3 database storage in GiB."
  type        = number
  default     = 100

  validation {
    condition     = var.database_allocated_storage >= 100
    error_message = "database_allocated_storage must be at least 100 GiB."
  }
}

variable "database_max_allocated_storage" {
  description = "Upper storage autoscaling boundary in GiB."
  type        = number
  default     = 500

  validation {
    condition     = var.database_max_allocated_storage >= var.database_allocated_storage
    error_message = "database_max_allocated_storage must not be below the initial allocation."
  }
}

variable "database_multi_az" {
  description = "Whether RDS maintains a synchronous standby in another availability zone; required in production."
  type        = bool
  default     = true
}

variable "database_backup_retention_days" {
  description = "Automated RDS backup retention; production requires at least 30 days."
  type        = number
  default     = 30

  validation {
    condition     = var.database_backup_retention_days >= 7 && var.database_backup_retention_days <= 35
    error_message = "database_backup_retention_days must be between 7 and 35 days."
  }
}

variable "database_deletion_protection" {
  description = "RDS deletion protection; must remain enabled in production."
  type        = bool
  default     = true
}

variable "database_skip_final_snapshot" {
  description = "Whether deletion skips the final RDS snapshot; production must remain false."
  type        = bool
  default     = false
}

variable "artifact_bucket_name" {
  description = "Globally unique S3 bucket name for governed model and dataset artifacts."
  type        = string

  validation {
    condition     = can(regex("^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$", var.artifact_bucket_name))
    error_message = "artifact_bucket_name must be a valid explicit S3 bucket name."
  }
}

variable "audit_bucket_name" {
  description = "Globally unique S3 bucket name for immutable platform audit and recovery evidence."
  type        = string

  validation {
    condition     = var.audit_bucket_name != var.artifact_bucket_name && can(regex("^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$", var.audit_bucket_name))
    error_message = "audit_bucket_name must be a valid S3 bucket name distinct from artifact_bucket_name."
  }
}

variable "ecr_repository_names" {
  description = "Closed set of ECR repository names provisioned for platform workloads."
  type        = set(string)

  validation {
    condition = (
      length(var.ecr_repository_names) > 0 &&
      alltrue([for name in var.ecr_repository_names : can(regex("^[a-z0-9]+(?:[._/-][a-z0-9]+)*$", name))])
    )
    error_message = "ecr_repository_names must contain at least one valid lowercase repository name."
  }
}

variable "hosted_zone_id" {
  description = "Existing enterprise Route 53 hosted zone ID that remains the DNS authority."
  type        = string

  validation {
    condition     = can(regex("^Z[A-Z0-9]{8,32}$", var.hosted_zone_id))
    error_message = "hosted_zone_id must be an explicit Route 53 hosted zone ID."
  }
}

variable "gateway_dns_records" {
  description = "Explicit Gateway DNS records. The platform owner supplies the real load-balancer target after GitOps creates it."
  type = map(object({
    name    = string
    type    = string
    ttl     = number
    records = list(string)
  }))

  validation {
    condition = alltrue([
      for record in values(var.gateway_dns_records) :
      length(trimspace(record.name)) > 0 &&
      contains(["A", "AAAA", "CNAME"], record.type) &&
      record.ttl >= 30 && record.ttl <= 86400 &&
      length(record.records) > 0 &&
      alltrue([for target in record.records : length(trimspace(target)) > 0])
    ])
    error_message = "gateway_dns_records must contain explicit A, AAAA, or CNAME targets with a bounded TTL."
  }
}

variable "platform_service_account_namespace" {
  description = "Existing GitOps-owned Kubernetes namespace bound to the platform Pod Identity role."
  type        = string
  default     = "industrial-ops"

  validation {
    condition     = can(regex("^[a-z0-9](?:[-a-z0-9]*[a-z0-9])?$", var.platform_service_account_namespace))
    error_message = "platform_service_account_namespace must be a valid Kubernetes namespace."
  }
}

variable "platform_service_account_name" {
  description = "Existing GitOps-owned service account bound through EKS Pod Identity."
  type        = string
  default     = "industrial-ops-runtime"

  validation {
    condition     = can(regex("^[a-z0-9](?:[-a-z0-9]*[a-z0-9])?$", var.platform_service_account_name))
    error_message = "platform_service_account_name must be a valid Kubernetes service account name."
  }
}
