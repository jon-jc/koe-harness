output "service_url" {
  description = "Public HTTPS endpoint."
  value       = "https://${aws_lb.koe.dns_name}"
}

output "websocket_url" {
  description = "Realtime streaming endpoint."
  value       = "wss://${aws_lb.koe.dns_name}/v1/stream"
}

output "ecr_repository_url" {
  description = "Push images here."
  value       = aws_ecr_repository.koe.repository_url
}

output "log_group" {
  value = aws_cloudwatch_log_group.koe.name
}

output "secret_arn" {
  description = "Populate with anthropic_api_key and openai_api_key."
  value       = aws_secretsmanager_secret.provider_keys.arn
}

output "cluster_name" {
  value = aws_ecs_cluster.koe.name
}
