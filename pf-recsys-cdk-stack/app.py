#!/usr/bin/env python3
"""
CDK App entry point for PropertyFinder RecSys infrastructure.

Deploy:
  cdk bootstrap aws://ACCOUNT_ID/me-south-1
  cdk deploy --all
"""
import aws_cdk as cdk
from pf_recsys_stack import PFRecSysStack

app = cdk.App()

PFRecSysStack(
    app, "PFRecSysStack",
    env=cdk.Environment(
        account=app.node.try_get_context("account") or None,
        region=app.node.try_get_context("region") or "me-south-1",
    ),
    description="PropertyFinder RecSys — ML Recommendations & Search Ranking Pipeline",
)

app.synth()
