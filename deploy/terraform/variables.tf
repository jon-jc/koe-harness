variable "region" {
  description = "AWS region. ap-northeast-1 keeps inference and users in Japan."
  type        = string
  default     = "ap-northeast-1"
}

variable "environment" {
  description = "Deployment environment."
  type        = string
  default     = "staging"

  validation {
    condition     = contains(["staging", "production"], var.environment)
    error_message = "environment must be staging or production."
  }
}

variable "image_tag" {
  description = "Image tag to deploy. Must be immutable — never 'latest'."
  type        = string

  validation {
    condition     = var.image_tag != "latest"
    error_message = "Deploy an immutable tag; 'latest' makes a rollback ambiguous."
  }
}

# -- networking -------------------------------------------------------------

variable "vpc_id" {
  description = "VPC to deploy into."
  type        = string
}

variable "public_subnet_ids" {
  description = "Subnets for the load balancer (at least two AZs)."
  type        = list(string)

  validation {
    condition     = length(var.public_subnet_ids) >= 2
    error_message = "An ALB needs subnets in at least two availability zones."
  }
}

variable "private_subnet_ids" {
  description = "Subnets for the tasks. Private: tasks are only reachable via the ALB."
  type        = list(string)
}

variable "certificate_arn" {
  description = "ACM certificate for the HTTPS listener."
  type        = string
}

variable "allowed_cidrs" {
  description = "CIDRs permitted to reach the load balancer."
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "cors_origins" {
  description = "Exact origins allowed to call the API. A wildcard is rejected in production."
  type        = list(string)
  default     = []
}

# -- capacity ---------------------------------------------------------------

variable "task_cpu" {
  description = "Fargate CPU units per task."
  type        = string
  default     = "1024"
}

variable "task_memory" {
  description = "Fargate memory (MiB). Sized for max_sessions_per_task audio buffers."
  type        = string
  default     = "2048"
}

variable "max_sessions_per_task" {
  description = "Concurrent streaming sessions one task accepts before rejecting."
  type        = number
  default     = 32
}

variable "desired_count" {
  description = "Initial task count; autoscaling takes over afterwards."
  type        = number
  default     = 2
}

variable "min_capacity" {
  description = "Minimum tasks. Two, so a single AZ failure is not an outage."
  type        = number
  default     = 2
}

variable "max_capacity" {
  description = "Maximum tasks."
  type        = number
  default     = 20
}

# -- behaviour --------------------------------------------------------------

variable "websocket_idle_timeout" {
  description = <<-EOT
    ALB idle timeout in seconds. A WebSocket carrying a meeting is idle
    whenever nobody is speaking, so the 60s default severs live calls.
  EOT
  type        = number
  default     = 900
}

variable "session_budget_usd" {
  description = "Per-session provider spend ceiling."
  type        = number
  default     = 5.0
}

variable "log_level" {
  type    = string
  default = "INFO"
}

variable "log_retention_days" {
  type    = number
  default = 30
}

# -- alarms -----------------------------------------------------------------

variable "latency_p95_alarm_seconds" {
  description = "p95 request latency that should page."
  type        = number
  default     = 3.0
}

variable "error_alarm_threshold" {
  description = "Application errors per 5 minutes before alarming."
  type        = number
  default     = 10
}

variable "alarm_topic_arns" {
  description = "SNS topics to notify on alarm."
  type        = list(string)
  default     = []
}
