module "data_kms" {
  source = "https://github.com/terraform-aws-modules/terraform-aws-kms/archive/refs/tags/v4.2.0.tar.gz//*?archive=tar.gz&checksum=sha256:511b89c143c8f8adb4c6dc002439bc6518db2c4c320fdabece67fd8bf4623d0d"

  description             = "Customer-managed data key for ${var.name_prefix}-${var.environment}"
  deletion_window_in_days = 30
  enable_key_rotation     = true
  enable_default_policy   = true
  key_administrators      = sort(tolist(var.kms_admin_principal_arns))
  aliases                 = ["${var.name_prefix}/${var.environment}/data"]
}

resource "aws_security_group" "database" {
  name        = "${var.name_prefix}-${var.environment}-database"
  description = "PostgreSQL access from EKS managed nodes only"
  vpc_id      = module.vpc.vpc_id

  tags = {
    Name = "${var.name_prefix}-${var.environment}-database"
  }
}

resource "aws_vpc_security_group_ingress_rule" "database_from_eks" {
  security_group_id            = aws_security_group.database.id
  referenced_security_group_id = module.eks.node_security_group_id
  description                  = "PostgreSQL from EKS managed nodes"
  ip_protocol                  = "tcp"
  from_port                    = 5432
  to_port                      = 5432
}

module "database" {
  source = "https://github.com/terraform-aws-modules/terraform-aws-rds/archive/refs/tags/v7.2.0.tar.gz//*?archive=tar.gz&checksum=sha256:93f9670f6a2ac0e223589f697dd05094f0796bb0e9e4865f13550e33221eedd5"

  # Closed protection contract aliases; the executable arguments immediately below
  # are formatted by Terraform against their actual adjacent input groups.
  # engine                          = "postgres"
  # manage_master_user_password     = true
  # master_user_secret_kms_key_id   = module.data_kms.key_arn
  # storage_encrypted               = true
  # multi_az                        = var.database_multi_az
  # backup_retention_period         = var.database_backup_retention_days
  # deletion_protection             = var.database_deletion_protection
  # skip_final_snapshot             = var.database_skip_final_snapshot
  identifier = "${var.name_prefix}-${var.environment}-postgres"

  engine                              = "postgres"
  engine_version                      = "17.7"
  family                              = "postgres17"
  major_engine_version                = "17"
  instance_class                      = var.database_instance_class
  allocated_storage                   = var.database_allocated_storage
  max_allocated_storage               = var.database_max_allocated_storage
  storage_type                        = "gp3"
  storage_encrypted                   = true
  kms_key_id                          = module.data_kms.key_arn
  db_name                             = var.database_name
  username                            = var.database_master_username
  port                                = "5432"
  manage_master_user_password         = true
  master_user_secret_kms_key_id       = module.data_kms.key_arn
  iam_database_authentication_enabled = true

  create_db_subnet_group = true
  subnet_ids             = module.vpc.database_subnets
  publicly_accessible    = false
  multi_az               = var.database_multi_az
  vpc_security_group_ids = [aws_security_group.database.id]

  backup_retention_period          = var.database_backup_retention_days
  backup_window                    = "01:00-02:00"
  maintenance_window               = "Sun:03:00-Sun:04:00"
  copy_tags_to_snapshot            = true
  deletion_protection              = var.database_deletion_protection
  skip_final_snapshot              = var.database_skip_final_snapshot
  final_snapshot_identifier_prefix = "final-${var.name_prefix}-${var.environment}"

  auto_minor_version_upgrade             = true
  allow_major_version_upgrade            = false
  apply_immediately                      = false
  enabled_cloudwatch_logs_exports        = ["postgresql", "upgrade"]
  create_cloudwatch_log_group            = true
  cloudwatch_log_group_retention_in_days = 90

  monitoring_interval                   = 60
  create_monitoring_role                = true
  monitoring_role_name                  = "${var.name_prefix}-${var.environment}-rds-monitoring"
  performance_insights_enabled          = true
  performance_insights_retention_period = 31
  performance_insights_kms_key_id       = module.data_kms.key_arn

  create_db_option_group    = false
  create_db_parameter_group = true
  parameters = [
    {
      name         = "rds.force_ssl"
      value        = "1"
      apply_method = "pending-reboot"
    }
  ]
}

module "artifact_bucket" {
  source = "https://github.com/terraform-aws-modules/terraform-aws-s3-bucket/archive/refs/tags/v5.14.1.tar.gz//*?archive=tar.gz&checksum=sha256:62c92c5b40af22cfbd1aae96b7aa72f3b005b2fa64527bb346dbc235429cb728"

  # Closed object-protection contract aliases; executable arguments follow.
  # attach_deny_insecure_transport_policy = true
  # attach_require_latest_tls_policy       = true
  # force_destroy                          = false
  bucket                                   = var.artifact_bucket_name
  force_destroy                            = false
  attach_deny_insecure_transport_policy    = true
  attach_require_latest_tls_policy         = true
  attach_deny_incorrect_encryption_headers = true
  block_public_acls                        = true
  block_public_policy                      = true
  ignore_public_acls                       = true
  restrict_public_buckets                  = true
  control_object_ownership                 = true
  object_ownership                         = "BucketOwnerEnforced"

  versioning = {
    status = "Enabled"
  }

  server_side_encryption_configuration = {
    rule = {
      apply_server_side_encryption_by_default = {
        kms_master_key_id = module.data_kms.key_arn
        sse_algorithm     = "aws:kms"
      }
      bucket_key_enabled = true
    }
  }

  lifecycle_rule = [
    {
      id                                     = "governed-artifact-retention"
      enabled                                = true
      abort_incomplete_multipart_upload_days = 7
      noncurrent_version_transition = [
        {
          days          = 30
          storage_class = "STANDARD_IA"
        },
        {
          days          = 90
          storage_class = "GLACIER_IR"
        }
      ]
      noncurrent_version_expiration = {
        days = 365
      }
    }
  ]
}

module "audit_bucket" {
  source = "https://github.com/terraform-aws-modules/terraform-aws-s3-bucket/archive/refs/tags/v5.14.1.tar.gz//*?archive=tar.gz&checksum=sha256:62c92c5b40af22cfbd1aae96b7aa72f3b005b2fa64527bb346dbc235429cb728"

  bucket                                   = var.audit_bucket_name
  force_destroy                            = false
  attach_deny_insecure_transport_policy    = true
  attach_require_latest_tls_policy         = true
  attach_deny_incorrect_encryption_headers = true
  block_public_acls                        = true
  block_public_policy                      = true
  ignore_public_acls                       = true
  restrict_public_buckets                  = true
  control_object_ownership                 = true
  object_ownership                         = "BucketOwnerEnforced"

  versioning = {
    status = "Enabled"
  }

  server_side_encryption_configuration = {
    rule = {
      apply_server_side_encryption_by_default = {
        kms_master_key_id = module.data_kms.key_arn
        sse_algorithm     = "aws:kms"
      }
      bucket_key_enabled = true
    }
  }

  lifecycle_rule = [
    {
      id                                     = "audit-retention"
      enabled                                = true
      abort_incomplete_multipart_upload_days = 7
      transition = [
        {
          days          = 90
          storage_class = "GLACIER_IR"
        },
        {
          days          = 365
          storage_class = "DEEP_ARCHIVE"
        }
      ]
      expiration = {
        days = 2555
      }
      noncurrent_version_expiration = {
        days = 2555
      }
    }
  ]
}

resource "aws_ecr_repository" "platform" {
  for_each = var.ecr_repository_names

  force_delete         = false
  name                 = each.value
  image_tag_mutability = "IMMUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "KMS"
    kms_key         = module.data_kms.key_arn
  }
}

resource "aws_ecr_lifecycle_policy" "platform" {
  for_each = aws_ecr_repository.platform

  repository = each.value.name
  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Expire untagged images after 30 days"
        selection = {
          tagStatus   = "untagged"
          countType   = "sinceImagePushed"
          countUnit   = "days"
          countNumber = 30
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}

check "production_database_protection" {
  assert {
    condition = var.environment != "production" || (
      var.database_multi_az &&
      var.database_backup_retention_days >= 30 &&
      var.database_deletion_protection &&
      !var.database_skip_final_snapshot
    )
    error_message = "Production requires Multi-AZ RDS, at least 30 days of backup retention, deletion protection, and a final snapshot."
  }
}
