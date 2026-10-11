resource "aws_iam_role" "platform_pod_identity" {
  name        = "${var.name_prefix}-${var.environment}-platform-pod"
  description = "Least-privilege data access for the GitOps-owned industrial operations runtime"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "EksPodIdentity"
        Effect = "Allow"
        Principal = {
          Service = "pods.eks.amazonaws.com"
        }
        Action = ["sts:AssumeRole", "sts:TagSession"]
      }
    ]
  })
}

data "aws_iam_policy_document" "platform_data_access" {
  statement {
    sid    = "ReadWriteGovernedArtifacts"
    effect = "Allow"
    actions = [
      "s3:AbortMultipartUpload",
      "s3:GetObject",
      "s3:ListBucket",
      "s3:PutObject",
    ]
    resources = [
      module.artifact_bucket.s3_bucket_arn,
      "${module.artifact_bucket.s3_bucket_arn}/*",
    ]
  }

  statement {
    sid    = "WriteAuditEvidence"
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:ListBucket",
      "s3:PutObject",
    ]
    resources = [
      module.audit_bucket.s3_bucket_arn,
      "${module.audit_bucket.s3_bucket_arn}/*",
    ]
  }

  statement {
    sid    = "ResolveDatabaseCredential"
    effect = "Allow"
    actions = [
      "secretsmanager:DescribeSecret",
      "secretsmanager:GetSecretValue",
    ]
    resources = [module.database.db_instance_master_user_secret_arn]
  }

  statement {
    sid    = "UseDataEncryptionKey"
    effect = "Allow"
    actions = [
      "kms:Decrypt",
      "kms:DescribeKey",
      "kms:Encrypt",
      "kms:GenerateDataKey",
      "kms:ReEncryptFrom",
      "kms:ReEncryptTo",
    ]
    resources = [module.data_kms.key_arn]
  }

  statement {
    sid       = "AuthenticateToEcr"
    effect    = "Allow"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }

  statement {
    sid    = "PullGovernedImages"
    effect = "Allow"
    actions = [
      "ecr:BatchCheckLayerAvailability",
      "ecr:BatchGetImage",
      "ecr:GetDownloadUrlForLayer",
    ]
    resources = [for repository in aws_ecr_repository.platform : repository.arn]
  }
}

resource "aws_iam_role_policy" "platform_data_access" {
  name   = "governed-platform-data-access"
  role   = aws_iam_role.platform_pod_identity.id
  policy = data.aws_iam_policy_document.platform_data_access.json
}

resource "aws_eks_pod_identity_association" "platform" {
  cluster_name    = module.eks.cluster_name
  namespace       = var.platform_service_account_namespace
  service_account = var.platform_service_account_name
  role_arn        = aws_iam_role.platform_pod_identity.arn
}
