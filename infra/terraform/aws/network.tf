module "vpc" {
  source = "https://github.com/terraform-aws-modules/terraform-aws-vpc/archive/refs/tags/v6.6.1.tar.gz//*?archive=tar.gz&checksum=sha256:97e1dcabe56e258f31195639c72166ef27a0a063f1526f2b5818e2660d7a13eb"

  name = "${var.name_prefix}-${var.environment}-vpc"
  cidr = var.vpc_cidr

  azs              = var.availability_zones
  public_subnets   = var.public_subnet_cidrs
  private_subnets  = var.private_subnet_cidrs
  database_subnets = var.database_subnet_cidrs

  enable_nat_gateway     = true
  single_nat_gateway     = var.single_nat_gateway
  one_nat_gateway_per_az = !var.single_nat_gateway

  enable_dns_support   = true
  enable_dns_hostnames = true

  create_database_subnet_group           = true
  create_database_subnet_route_table     = true
  create_database_internet_gateway_route = false
  create_database_nat_gateway_route      = false

  enable_flow_log                                 = true
  flow_log_destination_type                       = "cloud-watch-logs"
  create_flow_log_cloudwatch_log_group            = true
  create_flow_log_cloudwatch_iam_role             = true
  flow_log_cloudwatch_log_group_retention_in_days = var.vpc_flow_log_retention_days
  flow_log_cloudwatch_log_group_kms_key_id        = module.data_kms.key_arn

  public_subnet_tags = {
    "kubernetes.io/role/elb" = "1"
    NetworkTier              = "public-nat"
  }

  private_subnet_tags = {
    "kubernetes.io/role/internal-elb" = "1"
    "karpenter.sh/discovery"          = local.cluster_name
    NetworkTier                       = "private-workload"
  }

  database_subnet_tags = {
    NetworkTier = "isolated-database"
  }
}

resource "aws_vpc_endpoint" "s3" {
  vpc_id            = module.vpc.vpc_id
  service_name      = "com.amazonaws.${var.aws_region}.s3"
  vpc_endpoint_type = "Gateway"
  route_table_ids   = module.vpc.private_route_table_ids

  tags = {
    Name = "${var.name_prefix}-${var.environment}-s3-endpoint"
  }
}
