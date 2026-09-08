########################################################################
# koe on ECS Fargate.
#
# The shape is ordinary — ALB in front of a Fargate service — but three
# settings are load-bearing for a *voice* workload and are the reason this
# is not a copy of a generic web-service module:
#
#   1. ALB idle timeout is minutes, not the 60s default. A WebSocket carrying
#      a meeting is idle by the load balancer's definition whenever nobody is
#      speaking, and the default cuts the connection mid-meeting.
#
#   2. Deregistration delay exceeds the app's graceful-shutdown window, so a
#      deploy drains in-flight sessions instead of severing calls.
#
#   3. Autoscaling tracks concurrent sessions, not CPU. A streaming session is
#      mostly waiting on a model, so CPU stays low while memory and socket
#      count are the real constraint — CPU-based scaling would under-provision
#      exactly when the service is full.
########################################################################

terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.40"
    }
  }
}

provider "aws" {
  region = var.region
  default_tags {
    tags = {
      Application = "koe"
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}

locals {
  name = "koe-${var.environment}"
}

data "aws_caller_identity" "current" {}

########################################################################
# container registry
########################################################################

resource "aws_ecr_repository" "koe" {
  name                 = local.name
  image_tag_mutability = "IMMUTABLE" # a tag must always mean one image

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
  }
}

resource "aws_ecr_lifecycle_policy" "koe" {
  repository = aws_ecr_repository.koe.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep the last 20 images"
      selection    = { tagStatus = "any", countType = "imageCountMoreThan", countNumber = 20 }
      action       = { type = "expire" }
    }]
  })
}

########################################################################
# logging
########################################################################

resource "aws_cloudwatch_log_group" "koe" {
  name              = "/ecs/${local.name}"
  retention_in_days = var.log_retention_days
}

# The application emits CloudWatch EMF on stdout, so metrics need no agent and
# no sidecar — this filter is only for the error rate, which is a log concern.
resource "aws_cloudwatch_log_metric_filter" "errors" {
  name           = "${local.name}-errors"
  log_group_name = aws_cloudwatch_log_group.koe.name
  pattern        = "{ $.level = \"ERROR\" }"

  metric_transformation {
    name          = "ApplicationErrors"
    namespace     = "koe/${var.environment}"
    value         = "1"
    default_value = "0"
  }
}

########################################################################
# secrets
########################################################################

resource "aws_secretsmanager_secret" "provider_keys" {
  name                    = "${local.name}/provider-keys"
  recovery_window_in_days = var.environment == "production" ? 30 : 0
}

########################################################################
# iam
########################################################################

data "aws_iam_policy_document" "ecs_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

# Execution role: what ECS itself needs to start the task.
resource "aws_iam_role" "execution" {
  name               = "${local.name}-execution"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume.json
}

resource "aws_iam_role_policy_attachment" "execution_managed" {
  role       = aws_iam_role.execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# Reading the secret happens at task start, so it belongs to the execution
# role — the application never reads Secrets Manager itself and therefore
# never needs that permission at runtime.
data "aws_iam_policy_document" "read_secrets" {
  statement {
    actions   = ["secretsmanager:GetSecretValue"]
    resources = [aws_secretsmanager_secret.provider_keys.arn]
  }
}

resource "aws_iam_role_policy" "execution_secrets" {
  name   = "${local.name}-read-secrets"
  role   = aws_iam_role.execution.id
  policy = data.aws_iam_policy_document.read_secrets.json
}

# Task role: what the application may do. Deliberately minimal — it writes
# metrics and nothing else. Audio is not persisted by default.
resource "aws_iam_role" "task" {
  name               = "${local.name}-task"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume.json
}

data "aws_iam_policy_document" "task_permissions" {
  statement {
    actions   = ["cloudwatch:PutMetricData"]
    resources = ["*"]
    condition {
      test     = "StringEquals"
      variable = "cloudwatch:namespace"
      values   = ["koe/${var.environment}"]
    }
  }
}

resource "aws_iam_role_policy" "task" {
  name   = "${local.name}-task"
  role   = aws_iam_role.task.id
  policy = data.aws_iam_policy_document.task_permissions.json
}

########################################################################
# networking
########################################################################

resource "aws_security_group" "alb" {
  name        = "${local.name}-alb"
  description = "Public ingress to the koe load balancer"
  vpc_id      = var.vpc_id

  ingress {
    description = "HTTPS"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidrs
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "service" {
  name        = "${local.name}-service"
  description = "koe tasks; reachable only from the load balancer"
  vpc_id      = var.vpc_id

  ingress {
    description     = "From the ALB"
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  egress {
    description = "Model provider APIs"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

########################################################################
# load balancer
########################################################################

resource "aws_lb" "koe" {
  name               = local.name
  load_balancer_type = "application"
  subnets            = var.public_subnet_ids
  security_groups    = [aws_security_group.alb.id]

  # A WebSocket carrying a meeting is "idle" whenever nobody is speaking. The
  # 60s default severs those connections mid-conversation.
  idle_timeout = var.websocket_idle_timeout

  enable_deletion_protection = var.environment == "production"
  drop_invalid_header_fields = true
}

resource "aws_lb_target_group" "koe" {
  name        = local.name
  port        = 8000
  protocol    = "HTTP"
  vpc_id      = var.vpc_id
  target_type = "ip"

  # Longer than the app's 20s graceful shutdown, so a deploy drains in-flight
  # sessions rather than cutting callers off.
  deregistration_delay = 30

  health_check {
    path                = "/health"
    healthy_threshold   = 2
    unhealthy_threshold = 3
    timeout             = 5
    interval            = 15
    matcher             = "200"
  }

  # Sticky so a reconnecting client returns to the task holding its session
  # state. Sessions live in process, not in a shared store.
  stickiness {
    type            = "lb_cookie"
    cookie_duration = 3600
    enabled         = true
  }
}

resource "aws_lb_listener" "https" {
  load_balancer_arn = aws_lb.koe.arn
  port              = 443
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS13-1-2-2021-06"
  certificate_arn   = var.certificate_arn

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.koe.arn
  }
}

########################################################################
# service
########################################################################

resource "aws_ecs_cluster" "koe" {
  name = local.name

  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

resource "aws_ecs_task_definition" "koe" {
  family                   = local.name
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.task_cpu
  memory                   = var.task_memory
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task.arn

  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "ARM64" # ~20% cheaper per vCPU-hour than x86
  }

  container_definitions = jsonencode([{
    name      = "koe"
    image     = "${aws_ecr_repository.koe.repository_url}:${var.image_tag}"
    essential = true

    portMappings = [{ containerPort = 8000, protocol = "tcp" }]

    environment = [
      { name = "KOE_ENVIRONMENT", value = var.environment },
      { name = "KOE_LOG_LEVEL", value = var.log_level },
      { name = "KOE_STRUCTURED_LOGS", value = "true" },
      { name = "KOE_MAX_CONCURRENT_SESSIONS", value = tostring(var.max_sessions_per_task) },
      { name = "KOE_SESSION_BUDGET_USD", value = tostring(var.session_budget_usd) },
      { name = "KOE_CORS_ORIGINS", value = jsonencode(var.cors_origins) },
      { name = "KOE_DEFAULT_LANGUAGE", value = "ja" },
    ]

    secrets = [
      {
        name      = "KOE_ANTHROPIC_API_KEY"
        valueFrom = "${aws_secretsmanager_secret.provider_keys.arn}:anthropic_api_key::"
      },
      {
        name      = "KOE_OPENAI_API_KEY"
        valueFrom = "${aws_secretsmanager_secret.provider_keys.arn}:openai_api_key::"
      },
    ]

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.koe.name
        "awslogs-region"        = var.region
        "awslogs-stream-prefix" = "koe"
      }
    }

    healthCheck = {
      command     = ["CMD-SHELL", "curl -fsS http://localhost:8000/health || exit 1"]
      interval    = 15
      timeout     = 5
      retries     = 3
      startPeriod = 20
    }

    readonlyRootFilesystem = true
    linuxParameters        = { initProcessEnabled = true }
  }])
}

resource "aws_ecs_service" "koe" {
  name            = local.name
  cluster         = aws_ecs_cluster.koe.id
  task_definition = aws_ecs_task_definition.koe.arn
  desired_count   = var.desired_count
  launch_type     = "FARGATE"

  # Long enough for a session to drain; a voice deploy that cuts calls is a
  # visible outage even when every request "succeeded".
  health_check_grace_period_seconds  = 60
  deployment_minimum_healthy_percent = 100
  deployment_maximum_percent         = 200

  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }

  network_configuration {
    subnets          = var.private_subnet_ids
    security_groups  = [aws_security_group.service.id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.koe.arn
    container_name   = "koe"
    container_port   = 8000
  }

  lifecycle {
    ignore_changes = [desired_count] # autoscaling owns this
  }
}

########################################################################
# autoscaling
########################################################################

resource "aws_appautoscaling_target" "koe" {
  service_namespace  = "ecs"
  resource_id        = "service/${aws_ecs_cluster.koe.name}/${aws_ecs_service.koe.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  min_capacity       = var.min_capacity
  max_capacity       = var.max_capacity
}

# Scale on connections per task rather than CPU. A streaming session spends
# most of its life awaiting a model, so CPU stays low while sockets and memory
# saturate — CPU-based scaling would under-provision exactly at capacity.
resource "aws_appautoscaling_policy" "sessions" {
  name               = "${local.name}-sessions"
  policy_type        = "TargetTrackingScaling"
  service_namespace  = aws_appautoscaling_target.koe.service_namespace
  resource_id        = aws_appautoscaling_target.koe.resource_id
  scalable_dimension = aws_appautoscaling_target.koe.scalable_dimension

  target_tracking_scaling_policy_configuration {
    target_value = var.max_sessions_per_task * 0.7 # headroom for a burst

    customized_metric_specification {
      metrics {
        id    = "active"
        label = "Active connections across the target group"
        metric_stat {
          metric {
            namespace   = "AWS/ApplicationELB"
            metric_name = "ActiveConnectionCount"
            dimensions {
              name  = "LoadBalancer"
              value = aws_lb.koe.arn_suffix
            }
          }
          stat = "Sum"
        }
        return_data = false
      }

      metrics {
        id    = "tasks"
        label = "Running tasks"
        metric_stat {
          metric {
            namespace   = "ECS/ContainerInsights"
            metric_name = "RunningTaskCount"
            dimensions {
              name  = "ClusterName"
              value = aws_ecs_cluster.koe.name
            }
            dimensions {
              name  = "ServiceName"
              value = aws_ecs_service.koe.name
            }
          }
          stat = "Average"
        }
        return_data = false
      }

      metrics {
        id          = "per_task"
        label       = "Sessions per task"
        expression  = "active / MAX([tasks, 1])"
        return_data = true
      }
    }

    # Scale out quickly, in slowly: adding a task is cheap, while removing one
    # during a lull can strand sessions that were about to resume.
    scale_out_cooldown = 60
    scale_in_cooldown  = 300
  }
}

########################################################################
# alarms
########################################################################

resource "aws_cloudwatch_metric_alarm" "p95_latency" {
  alarm_name          = "${local.name}-p95-latency"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 3
  threshold           = var.latency_p95_alarm_seconds
  alarm_description   = "p95 request latency is above budget"
  treat_missing_data  = "notBreaching"

  metric_name        = "TargetResponseTime"
  namespace          = "AWS/ApplicationELB"
  period             = 60
  extended_statistic = "p95"

  dimensions = {
    LoadBalancer = aws_lb.koe.arn_suffix
  }

  alarm_actions = var.alarm_topic_arns
}

resource "aws_cloudwatch_metric_alarm" "errors" {
  alarm_name          = "${local.name}-application-errors"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  threshold           = var.error_alarm_threshold
  alarm_description   = "Application error log rate is elevated"
  treat_missing_data  = "notBreaching"

  metric_name = "ApplicationErrors"
  namespace   = "koe/${var.environment}"
  period      = 300
  statistic   = "Sum"

  alarm_actions = var.alarm_topic_arns
}

resource "aws_cloudwatch_metric_alarm" "unhealthy_hosts" {
  alarm_name          = "${local.name}-unhealthy-hosts"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  threshold           = 0
  alarm_description   = "Tasks are failing health checks"
  treat_missing_data  = "notBreaching"

  metric_name = "UnHealthyHostCount"
  namespace   = "AWS/ApplicationELB"
  period      = 60
  statistic   = "Maximum"

  dimensions = {
    LoadBalancer = aws_lb.koe.arn_suffix
    TargetGroup  = aws_lb_target_group.koe.arn_suffix
  }

  alarm_actions = var.alarm_topic_arns
}
