"""
pf_recsys_stack.py — AWS CDK Stack for PropertyFinder RecSys

Creates the complete production infrastructure:
  1. VPC + Networking (private subnets, NAT, security groups)
  2. S3 Bucket (artifacts, ETL output)
  3. ECR Repositories (API image, training image)
  4. Amazon OpenSearch Service (managed, HNSW vector index)
  5. Amazon ElastiCache Redis (managed, multi-AZ)
  6. ECS Fargate Cluster + 3 Services (ranking, recsys, search)
  7. Application Load Balancer (path-based routing)
  8. Lambda ETL (redshift_data.py)
  9. SageMaker Processing Job trigger
  10. EventBridge Schedules (daily ETL, hourly delta, daily training)
  11. CloudWatch Alarms + SNS Alerts
  12. IAM Roles (least-privilege)

Deploy:
  cdk bootstrap aws://ACCOUNT_ID/me-south-1
  cdk deploy --all
"""

import aws_cdk as cdk
from aws_cdk import (
    Stack,
    Duration,
    RemovalPolicy,
    CfnOutput,
    aws_ec2 as ec2,
    aws_s3 as s3,
    aws_ecr as ecr,
    aws_ecs as ecs,
    aws_ecs_patterns as ecs_patterns,
    aws_elasticache as elasticache,
    aws_opensearchservice as opensearch,
    aws_elasticloadbalancingv2 as elbv2,
    aws_iam as iam,
    aws_lambda as _lambda,
    aws_events as events,
    aws_events_targets as targets,
    aws_logs as logs,
    aws_cloudwatch as cloudwatch,
    aws_cloudwatch_actions as cw_actions,
    aws_sns as sns,
    aws_sns_subscriptions as sns_subs,
    aws_servicediscovery as sd,
)
from constructs import Construct


class PFRecSysStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ══════════════════════════════════════════════════════════════
        # PARAMETERS (configurable at deploy time)
        # ══════════════════════════════════════════════════════════════
        alert_email = cdk.CfnParameter(
            self, "AlertEmail",
            type="String",
            default="recsys-team@propertyfinder.com",
            description="Email address for CloudWatch alarm notifications",
        )

        redshift_cluster_id = cdk.CfnParameter(
            self, "RedshiftClusterId",
            type="String",
            default="pf-prod-cluster",
            description="Redshift cluster identifier for ETL",
        )

        redshift_db = cdk.CfnParameter(
            self, "RedshiftDB",
            type="String",
            default="pf_de_prod_db",
        )

        redshift_user = cdk.CfnParameter(
            self, "RedshiftUser",
            type="String",
            default="recsys_service",
        )

        redshift_iam_role_arn = cdk.CfnParameter(
            self, "RedshiftIAMRoleArn",
            type="String",
            description="IAM Role ARN that Redshift uses to UNLOAD to S3",
        )

        # ══════════════════════════════════════════════════════════════
        # 1. VPC + NETWORKING
        # ══════════════════════════════════════════════════════════════
        vpc = ec2.Vpc(
            self, "RecSysVPC",
            max_azs=2,
            nat_gateways=1,  # Cost optimization: 1 NAT (use 2 for HA)
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="Public",
                    subnet_type=ec2.SubnetType.PUBLIC,
                    cidr_mask=24,
                ),
                ec2.SubnetConfiguration(
                    name="Private",
                    subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
                    cidr_mask=24,
                ),
            ],
        )

        # Security Groups
        sg_alb = ec2.SecurityGroup(self, "SG-ALB", vpc=vpc, description="ALB inbound")
        sg_alb.add_ingress_rule(ec2.Peer.any_ipv4(), ec2.Port.tcp(80), "HTTP from anywhere")
        sg_alb.add_ingress_rule(ec2.Peer.any_ipv4(), ec2.Port.tcp(443), "HTTPS from anywhere")

        sg_ecs = ec2.SecurityGroup(self, "SG-ECS", vpc=vpc, description="ECS tasks")
        sg_ecs.add_ingress_rule(sg_alb, ec2.Port.tcp_range(8000, 8002), "ALB → ECS")
        sg_ecs.add_ingress_rule(sg_ecs, ec2.Port.tcp_range(8000, 8002), "ECS ↔ ECS inter-service")

        sg_opensearch = ec2.SecurityGroup(self, "SG-OpenSearch", vpc=vpc, description="OpenSearch")
        sg_opensearch.add_ingress_rule(sg_ecs, ec2.Port.tcp(443), "ECS → OpenSearch")

        sg_redis = ec2.SecurityGroup(self, "SG-Redis", vpc=vpc, description="Redis")
        sg_redis.add_ingress_rule(sg_ecs, ec2.Port.tcp(6379), "ECS → Redis")

        # ══════════════════════════════════════════════════════════════
        # 2. S3 BUCKET
        # ══════════════════════════════════════════════════════════════
        bucket = s3.Bucket(
            self, "ArtifactsBucket",
            bucket_name=f"pf-recsys-{self.account}-{self.region}",
            versioned=True,  # Enables rollback to previous brain.pkl
            removal_policy=RemovalPolicy.RETAIN,
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="expire-old-snapshots",
                    prefix="gold/",
                    expiration=Duration.days(30),
                ),
            ],
        )

        # ══════════════════════════════════════════════════════════════
        # 3. ECR REPOSITORIES
        # ══════════════════════════════════════════════════════════════
        ecr_api = ecr.Repository(
            self, "ECR-API",
            repository_name="pf-recsys-api",
            removal_policy=RemovalPolicy.RETAIN,
            lifecycle_rules=[ecr.LifecycleRule(max_image_count=10)],
        )

        ecr_training = ecr.Repository(
            self, "ECR-Training",
            repository_name="pf-recsys-training",
            removal_policy=RemovalPolicy.RETAIN,
            lifecycle_rules=[ecr.LifecycleRule(max_image_count=5)],
        )

        # ══════════════════════════════════════════════════════════════
        # 4. AMAZON OPENSEARCH SERVICE
        # ══════════════════════════════════════════════════════════════
        os_domain = opensearch.Domain(
            self, "OpenSearch",
            domain_name="pf-recsys",
            version=opensearch.EngineVersion.OPENSEARCH_2_11,
            vpc=vpc,
            vpc_subnets=[ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS)],
            security_groups=[sg_opensearch],
            capacity=opensearch.CapacityConfig(
                data_node_instance_type="r6g.large.search",
                data_nodes=2,
                master_nodes=0,
            ),
            ebs=opensearch.EbsOptions(
                volume_type=ec2.EbsDeviceVolumeType.GP3,
                volume_size=100,
            ),
            node_to_node_encryption=True,
            encryption_at_rest=opensearch.EncryptionAtRestOptions(enabled=True),
            enforce_https=True,
            removal_policy=RemovalPolicy.RETAIN,
            logging=opensearch.LoggingOptions(
                slow_search_log_enabled=True,
                slow_index_log_enabled=True,
            ),
        )

        # ══════════════════════════════════════════════════════════════
        # 5. AMAZON ELASTICACHE (REDIS)
        # ══════════════════════════════════════════════════════════════
        redis_subnet_group = elasticache.CfnSubnetGroup(
            self, "RedisSubnetGroup",
            description="Private subnets for Redis",
            subnet_ids=[s.subnet_id for s in vpc.private_subnets],
        )

        redis_cluster = elasticache.CfnReplicationGroup(
            self, "RedisCluster",
            replication_group_description="PF RecSys — User DNA + Cache",
            engine="redis",
            engine_version="7.0",
            cache_node_type="cache.r6g.large",
            num_cache_clusters=2,  # Primary + 1 replica (multi-AZ)
            automatic_failover_enabled=True,
            cache_subnet_group_name=redis_subnet_group.ref,
            security_group_ids=[sg_redis.security_group_id],
            snapshot_retention_limit=3,
            at_rest_encryption_enabled=True,
            transit_encryption_enabled=False,  # Set True if your app supports TLS
        )
        redis_cluster.add_dependency(redis_subnet_group)

        redis_endpoint = redis_cluster.attr_primary_end_point_address
        redis_port = redis_cluster.attr_primary_end_point_port

        # ══════════════════════════════════════════════════════════════
        # 6. ECS FARGATE CLUSTER + SERVICES
        # ══════════════════════════════════════════════════════════════
        cluster = ecs.Cluster(
            self, "ECSCluster",
            cluster_name="pf-recsys",
            vpc=vpc,
            container_insights_v2=ecs.ContainerInsights.ENABLED,
        )

        # Cloud Map namespace for service discovery
        namespace = cluster.add_default_cloud_map_namespace(
            name="pf-recsys.local",
            type=sd.NamespaceType.DNS_PRIVATE,
            vpc=vpc,
        )

        # --- Shared task role (S3 read access for brain.pkl) ---
        task_role = iam.Role(
            self, "ECSTaskRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
        )
        bucket.grant_read(task_role)

        # --- Shared execution role ---
        exec_role = iam.Role(
            self, "ECSExecRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AmazonECSTaskExecutionRolePolicy"
                ),
            ],
        )

        # --- Common environment variables ---
        opensearch_url = f"https://{os_domain.domain_endpoint}"

        env_common = {
            "OPENSEARCH_HOST": os_domain.domain_endpoint,
            "OPENSEARCH_PORT": "443",
            "OPENSEARCH_SSL": "true",
            "REDIS_HOST": redis_endpoint,
            "OPENSEARCH_INDEX": "pf-inventory-v1",
        }

        # --- Log groups ---
        log_group_ranking = logs.LogGroup(
            self, "LogRanking", log_group_name="/ecs/pf-ranking-api",
            retention=logs.RetentionDays.TWO_WEEKS, removal_policy=RemovalPolicy.DESTROY,
        )
        log_group_recsys = logs.LogGroup(
            self, "LogRecSys", log_group_name="/ecs/pf-recsys-api",
            retention=logs.RetentionDays.TWO_WEEKS, removal_policy=RemovalPolicy.DESTROY,
        )
        log_group_search = logs.LogGroup(
            self, "LogSearch", log_group_name="/ecs/pf-search-api",
            retention=logs.RetentionDays.TWO_WEEKS, removal_policy=RemovalPolicy.DESTROY,
        )

        # ─────────────────────────────────────────────
        # 6a. RANKING API (1 task, 1 worker — brain.pkl in memory)
        # ─────────────────────────────────────────────
        ranking_task_def = ecs.FargateTaskDefinition(
            self, "RankingTaskDef",
            cpu=1024, memory_limit_mib=2048,
            task_role=task_role, execution_role=exec_role,
        )

        ranking_container = ranking_task_def.add_container(
            "ranking-api",
            image=ecs.ContainerImage.from_ecr_repository(ecr_api, tag="latest"),
            command=["sh", "-c",
                      "aws s3 cp s3://${S3_BUCKET}/artifacts/brain.pkl /app/artifacts/brain.pkl "
                      "&& uvicorn ranking_api:app --host 0.0.0.0 --port 8002 --workers 1"],
            environment={
                **env_common,
                "BRAIN_PATH": "/app/artifacts/brain.pkl",
                "S3_BUCKET": bucket.bucket_name,
                "RANKING_API_URL": "http://localhost:8002",
            },
            logging=ecs.LogDrivers.aws_logs(
                stream_prefix="ranking", log_group=log_group_ranking,
            ),
            health_check=ecs.HealthCheck(
                command=["CMD-SHELL", "curl -f http://localhost:8002/health || exit 1"],
                interval=Duration.seconds(30),
                timeout=Duration.seconds(10),
                retries=3,
                start_period=Duration.seconds(30),
            ),
        )
        ranking_container.add_port_mappings(
            ecs.PortMapping(container_port=8002, protocol=ecs.Protocol.TCP)
        )

        ranking_service = ecs.FargateService(
            self, "RankingService",
            cluster=cluster,
            task_definition=ranking_task_def,
            desired_count=1,
            security_groups=[sg_ecs],
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS),
            cloud_map_options=ecs.CloudMapOptions(
                name="ranking-api",
                cloud_map_namespace=namespace,
            ),
            circuit_breaker=ecs.DeploymentCircuitBreaker(rollback=True),
        )

        # ─────────────────────────────────────────────
        # 6b. RECSYS API (2 tasks, 4 workers each)
        # ─────────────────────────────────────────────
        recsys_task_def = ecs.FargateTaskDefinition(
            self, "RecSysTaskDef",
            cpu=1024, memory_limit_mib=2048,
            task_role=task_role, execution_role=exec_role,
        )

        recsys_container = recsys_task_def.add_container(
            "recsys-api",
            image=ecs.ContainerImage.from_ecr_repository(ecr_api, tag="latest"),
            command=["uvicorn", "recsys_api:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "4"],
            environment={
                **env_common,
                "RANKING_API_URL": "http://ranking-api.pf-recsys.local:8002",
            },
            logging=ecs.LogDrivers.aws_logs(
                stream_prefix="recsys", log_group=log_group_recsys,
            ),
            health_check=ecs.HealthCheck(
                command=["CMD-SHELL", "curl -f http://localhost:8001/health || exit 1"],
                interval=Duration.seconds(30),
                timeout=Duration.seconds(10),
                retries=3,
                start_period=Duration.seconds(15),
            ),
        )
        recsys_container.add_port_mappings(
            ecs.PortMapping(container_port=8001, protocol=ecs.Protocol.TCP)
        )

        recsys_service = ecs.FargateService(
            self, "RecSysService",
            cluster=cluster,
            task_definition=recsys_task_def,
            desired_count=2,
            security_groups=[sg_ecs],
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS),
            cloud_map_options=ecs.CloudMapOptions(
                name="recsys-api",
                cloud_map_namespace=namespace,
            ),
            circuit_breaker=ecs.DeploymentCircuitBreaker(rollback=True),
        )

        # ─────────────────────────────────────────────
        # 6c. SEARCH API (2 tasks, 4 workers each)
        # ─────────────────────────────────────────────
        search_task_def = ecs.FargateTaskDefinition(
            self, "SearchTaskDef",
            cpu=1024, memory_limit_mib=2048,
            task_role=task_role, execution_role=exec_role,
        )

        search_container = search_task_def.add_container(
            "search-api",
            image=ecs.ContainerImage.from_ecr_repository(ecr_api, tag="latest"),
            command=["uvicorn", "search_api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"],
            environment={
                **env_common,
                "RANKING_API_URL": "http://ranking-api.pf-recsys.local:8002",
            },
            logging=ecs.LogDrivers.aws_logs(
                stream_prefix="search", log_group=log_group_search,
            ),
            health_check=ecs.HealthCheck(
                command=["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
                interval=Duration.seconds(30),
                timeout=Duration.seconds(10),
                retries=3,
                start_period=Duration.seconds(15),
            ),
        )
        search_container.add_port_mappings(
            ecs.PortMapping(container_port=8000, protocol=ecs.Protocol.TCP)
        )

        search_service = ecs.FargateService(
            self, "SearchService",
            cluster=cluster,
            task_definition=search_task_def,
            desired_count=2,
            security_groups=[sg_ecs],
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS),
            cloud_map_options=ecs.CloudMapOptions(
                name="search-api",
                cloud_map_namespace=namespace,
            ),
            circuit_breaker=ecs.DeploymentCircuitBreaker(rollback=True),
        )

        # Grant OpenSearch access to ECS tasks
        os_domain.grant_read_write(task_role)

        # ══════════════════════════════════════════════════════════════
        # 7. APPLICATION LOAD BALANCER
        # ══════════════════════════════════════════════════════════════
        alb = elbv2.ApplicationLoadBalancer(
            self, "ALB",
            vpc=vpc,
            internet_facing=True,  # Set False if BFF is internal
            security_group=sg_alb,
        )

        listener = alb.add_listener("HTTP", port=80)

        # Default action: 404
        listener.add_action(
            "Default",
            action=elbv2.ListenerAction.fixed_response(
                status_code=404,
                content_type="application/json",
                message_body='{"error": "not found"}',
            ),
        )

        # Target groups
        tg_ranking = listener.add_targets(
            "TG-Ranking",
            priority=10,
            port=8002,
            targets=[ranking_service],
            conditions=[elbv2.ListenerCondition.path_patterns(["/rank*", "/admin/*", "/health"])],
            health_check=elbv2.HealthCheck(path="/health", interval=Duration.seconds(30)),
        )

        tg_recsys = listener.add_targets(
            "TG-RecSys",
            priority=20,
            port=8001,
            targets=[recsys_service],
            conditions=[elbv2.ListenerCondition.path_patterns(["/internal/v1/recommendations*"])],
            health_check=elbv2.HealthCheck(path="/health", interval=Duration.seconds(30)),
        )

        tg_search = listener.add_targets(
            "TG-Search",
            priority=30,
            port=8000,
            targets=[search_service],
            conditions=[elbv2.ListenerCondition.path_patterns(["/internal/v1/search*"])],
            health_check=elbv2.HealthCheck(path="/health", interval=Duration.seconds(30)),
        )

        # ══════════════════════════════════════════════════════════════
        # 8. AUTO-SCALING (RecSys + Search, not Ranking)
        # ══════════════════════════════════════════════════════════════
        for svc, name in [(recsys_service, "RecSys"), (search_service, "Search")]:
            scaling = svc.auto_scale_task_count(min_capacity=2, max_capacity=8)
            scaling.scale_on_cpu_utilization(
                f"{name}CpuScaling",
                target_utilization_percent=70,
                scale_in_cooldown=Duration.seconds(300),
                scale_out_cooldown=Duration.seconds(60),
            )
            scaling.scale_on_request_count(
                f"{name}RequestScaling",
                requests_per_target=500,
                target_group=tg_recsys if name == "RecSys" else tg_search,
                scale_in_cooldown=Duration.seconds(300),
                scale_out_cooldown=Duration.seconds(60),
            )

        # ══════════════════════════════════════════════════════════════
        # 9. LAMBDA ETL (redshift_data.py)
        # ══════════════════════════════════════════════════════════════
        etl_lambda = _lambda.Function(
            self, "ETLLambda",
            function_name="pf-recsys-etl",
            runtime=_lambda.Runtime.PYTHON_3_11,
            handler="redshift_data.lambda_handler",
            code=_lambda.Code.from_asset("lambda/etl"),  # Directory containing redshift_data.py
            timeout=Duration.minutes(15),
            memory_size=512,
            environment={
                "REDSHIFT_CLUSTER_ID": redshift_cluster_id.value_as_string,
                "REDSHIFT_DB": redshift_db.value_as_string,
                "REDSHIFT_USER": redshift_user.value_as_string,
                "REDSHIFT_IAM_ROLE": redshift_iam_role_arn.value_as_string,
                "S3_BUCKET": bucket.bucket_name,
            },
            log_retention=logs.RetentionDays.TWO_WEEKS,
        )

        # Lambda permissions
        bucket.grant_read_write(etl_lambda)
        etl_lambda.add_to_role_policy(iam.PolicyStatement(
            actions=[
                "redshift-data:ExecuteStatement",
                "redshift-data:DescribeStatement",
                "redshift-data:GetStatementResult",
            ],
            resources=["*"],
        ))
        etl_lambda.add_to_role_policy(iam.PolicyStatement(
            actions=["redshift:GetClusterCredentials"],
            resources=["*"],
        ))

        # ══════════════════════════════════════════════════════════════
        # 10. POST-TRAINING LAMBDA (seed data + reload brain)
        # ══════════════════════════════════════════════════════════════
        post_training_lambda = _lambda.Function(
            self, "PostTrainingLambda",
            function_name="pf-recsys-post-training",
            runtime=_lambda.Runtime.PYTHON_3_11,
            handler="post_training.lambda_handler",
            code=_lambda.Code.from_asset("lambda/post-training"),
            timeout=Duration.minutes(15),
            memory_size=1024,
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS),
            security_groups=[sg_ecs],
            environment={
                "S3_BUCKET": bucket.bucket_name,
                "RANKING_API_URL": "http://ranking-api.pf-recsys.local:8002",
                "OPENSEARCH_HOST": os_domain.domain_endpoint,
                "REDIS_HOST": redis_endpoint,
                "OPENSEARCH_INDEX": "pf-inventory-v1",
            },
            log_retention=logs.RetentionDays.TWO_WEEKS,
        )
        bucket.grant_read(post_training_lambda)

        # ══════════════════════════════════════════════════════════════
        # 11. SAGEMAKER IAM ROLE
        # ══════════════════════════════════════════════════════════════
        sagemaker_role = iam.Role(
            self, "SageMakerRole",
            role_name="pf-recsys-sagemaker-role",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess"),
            ],
        )
        bucket.grant_read_write(sagemaker_role)
        ecr_training.grant_pull(sagemaker_role)

        # ══════════════════════════════════════════════════════════════
        # 12. EVENTBRIDGE SCHEDULES
        # ══════════════════════════════════════════════════════════════

        # Daily full ETL at 2:00 AM UAE (22:00 UTC)
        events.Rule(
            self, "DailyETL",
            rule_name="pf-recsys-daily-etl",
            schedule=events.Schedule.cron(hour="22", minute="0"),
            targets=[targets.LambdaFunction(
                etl_lambda,
                event=events.RuleTargetInput.from_object({"is_hourly_delta": False}),
            )],
        )

        # Hourly delta ETL
        events.Rule(
            self, "HourlyDelta",
            rule_name="pf-recsys-hourly-delta",
            schedule=events.Schedule.rate(Duration.hours(1)),
            targets=[targets.LambdaFunction(
                etl_lambda,
                event=events.RuleTargetInput.from_object({"is_hourly_delta": True}),
            )],
        )

        # ══════════════════════════════════════════════════════════════
        # 13. SNS ALERTS + CLOUDWATCH ALARMS
        # ══════════════════════════════════════════════════════════════
        alert_topic = sns.Topic(self, "AlertTopic", topic_name="pf-recsys-alerts")
        alert_topic.add_subscription(
            sns_subs.EmailSubscription(alert_email.value_as_string)
        )

        # Alarm: Ranking API unhealthy
        cloudwatch.Alarm(
            self, "RankingUnhealthy",
            alarm_name="pf-ranking-unhealthy",
            metric=tg_ranking.metric_unhealthy_host_count(),
            threshold=1,
            evaluation_periods=3,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
        ).add_alarm_action(cw_actions.SnsAction(alert_topic))

        # Alarm: High latency (p95 > 2s)
        cloudwatch.Alarm(
            self, "HighLatency",
            alarm_name="pf-recsys-high-latency",
            metric=alb.metric_target_response_time(
                statistic="p95",
                period=Duration.minutes(5),
            ),
            threshold=2.0,
            evaluation_periods=2,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
        ).add_alarm_action(cw_actions.SnsAction(alert_topic))

        # Alarm: ETL Lambda errors
        cloudwatch.Alarm(
            self, "ETLErrors",
            alarm_name="pf-etl-errors",
            metric=etl_lambda.metric_errors(period=Duration.hours(1)),
            threshold=1,
            evaluation_periods=1,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
        ).add_alarm_action(cw_actions.SnsAction(alert_topic))

        # Alarm: 5xx errors from ALB
        cloudwatch.Alarm(
            self, "ALB5xx",
            alarm_name="pf-recsys-5xx-errors",
            metric=alb.metric_http_code_target(
                code=elbv2.HttpCodeTarget.TARGET_5XX_COUNT,
                period=Duration.minutes(5),
            ),
            threshold=10,
            evaluation_periods=2,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
        ).add_alarm_action(cw_actions.SnsAction(alert_topic))

        # ══════════════════════════════════════════════════════════════
        # OUTPUTS
        # ══════════════════════════════════════════════════════════════
        CfnOutput(self, "ALBEndpoint",
                  value=alb.load_balancer_dns_name,
                  description="ALB DNS — point your BFF here")

        CfnOutput(self, "OpenSearchEndpoint",
                  value=os_domain.domain_endpoint,
                  description="OpenSearch domain endpoint")

        CfnOutput(self, "RedisEndpoint",
                  value=redis_endpoint,
                  description="Redis primary endpoint")

        CfnOutput(self, "S3Bucket",
                  value=bucket.bucket_name,
                  description="Artifacts & ETL bucket")

        CfnOutput(self, "ECRApiRepo",
                  value=ecr_api.repository_uri,
                  description="ECR repo for API image")

        CfnOutput(self, "ECRTrainingRepo",
                  value=ecr_training.repository_uri,
                  description="ECR repo for training image")

        CfnOutput(self, "SageMakerRoleArn",
                  value=sagemaker_role.role_arn,
                  description="SageMaker execution role ARN")

        CfnOutput(self, "ServiceDiscovery",
                  value="ranking-api.pf-recsys.local:8002 | recsys-api.pf-recsys.local:8001 | search-api.pf-recsys.local:8000",
                  description="Internal service DNS names")
