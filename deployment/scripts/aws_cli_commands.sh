#!/usr/bin/env bash
# =============================================================================
# aws_cli_commands.sh  --  REFERENCE ONLY. Do NOT run blindly.
# -----------------------------------------------------------------------------
# Ready-to-adapt AWS CLI to stand up the single-instance deployment described in
# deployment/DEPLOYMENT_READINESS_REPORT.md. Every value in <ANGLE_BRACKETS>
# must be filled in. Nothing here has been executed. Review each command, then
# run them one at a time once you have approved the plan.
#
# Assumes AWS CLI is configured (`aws configure`) and you have a default VPC.
# =============================================================================
set -euo pipefail

REGION=<REGION e.g. us-east-1>
KEY_NAME=<EXISTING_KEYPAIR_NAME>            # you already have EC2 keys
MY_IP=$(curl -s https://checkip.amazonaws.com)/32   # your current IP for SSH
NAME=search-ranking
# Ubuntu LTS release to launch. Default is the conservative 22.04 (validated);
# bump to a newer LTS (e.g. 24.04) if you have verified dependency compatibility.
UBUNTU_VERSION=22.04

# ---- 1) Security group: SSH (you only) + HTTP (world) -----------------------
SG_ID=$(aws ec2 create-security-group \
    --group-name "${NAME}-sg" \
    --description "Search ranking app: SSH(me)+HTTP" \
    --region "$REGION" --query GroupId --output text)

aws ec2 authorize-security-group-ingress --group-id "$SG_ID" \
    --protocol tcp --port 22 --cidr "$MY_IP"  --region "$REGION"
aws ec2 authorize-security-group-ingress --group-id "$SG_ID" \
    --protocol tcp --port 80 --cidr 0.0.0.0/0 --region "$REGION"
# Do NOT open 8000/8501 to the world -- they stay on loopback behind nginx.
# (Add 443 here later if you put TLS on the box or an ALB in front.)

# ---- 2) Latest Ubuntu ${UBUNTU_VERSION} LTS AMI (Canonical, x86_64) ----------
# Version-anchored, codename-agnostic filter: the release codename changes every
# cycle (jammy=22.04, noble=24.04, ...) and the image-path prefix moved from
# hvm-ssd to hvm-ssd-gp3 in 24.04+, so we match on the ${UBUNTU_VERSION} version
# string and wildcard both. Architecture stays x86_64 (amd64).
AMI_ID=$(aws ec2 describe-images --region "$REGION" \
    --owners 099720109477 \
    --filters "Name=name,Values=ubuntu/images/hvm-ssd*/ubuntu-*-${UBUNTU_VERSION}-amd64-server-*" \
              "Name=state,Values=available" \
    --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text)
echo "Using AMI: $AMI_ID"

# ---- 3) Launch the instance (t3.medium, 20 GB gp3) --------------------------
INSTANCE_ID=$(aws ec2 run-instances --region "$REGION" \
    --image-id "$AMI_ID" \
    --instance-type t3.medium \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SG_ID" \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":20,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${NAME}}]" \
    --query 'Instances[0].InstanceId' --output text)
echo "Launched: $INSTANCE_ID"

aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"
PUBLIC_DNS=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" --region "$REGION" \
    --query 'Reservations[0].Instances[0].PublicDnsName' --output text)
echo "SSH:  ssh -i <KEY.pem> ubuntu@${PUBLIC_DNS}"
echo "App:  http://${PUBLIC_DNS}/     Docs: http://${PUBLIC_DNS}/api/docs"

# ---- 4) (OPTIONAL) IAM instance profile for S3 artifact pull ----------------
# Only needed for stage_artifacts.sh Strategy B. Create a role with a policy
# allowing s3:GetObject/s3:ListBucket on your artifact bucket, then:
#   aws ec2 associate-iam-instance-profile --instance-id "$INSTANCE_ID" \
#       --iam-instance-profile Name=<INSTANCE_PROFILE_NAME> --region "$REGION"

# ---- 5) (OPTIONAL) CloudWatch agent config is in deployment/cloudwatch/ -----
echo "Next: SCP/clone the repo + artifacts, then run deployment/scripts/setup_ec2.sh"
