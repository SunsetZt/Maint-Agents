resource "aws_route53_record" "gateway" {
  for_each = var.gateway_dns_records

  zone_id = var.hosted_zone_id
  name    = each.value.name
  type    = each.value.type
  ttl     = each.value.ttl
  records = each.value.records
}
