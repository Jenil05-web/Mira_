#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════
# deploy_aws.sh — Deploy MIRA to AWS App Runner (Fixed Spacing Path)
# Run from Mira_project root: bash deploy_aws.sh
# ══════════════════════════════════════════════════════════════════════════

set -e

# Define direct path to the AWS executable (handles Git Bash space issue)
AWS_CMD="/c/Program Files/Amazon/AWSCLIV2/aws"

# ── CONFIG — update these two lines ──────────────────────────────────────
AWS_REGION="ap-south-1"          # Mumbai — closest to India
AWS_ACCOUNT_ID=$("$AWS_CMD" sts get-caller-identity --query Account --output text)
SERVICE_NAME="mira-clinical"
IMAGE_NAME="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$SERVICE_NAME"
# ─────────────────────────────────────────────────────────────────────────

echo "🏥 MIRA — AWS App Runner Deployment"
echo "====================================="
echo "Account : $AWS_ACCOUNT_ID"
echo "Region  : $AWS_REGION"
echo "Service : $SERVICE_NAME"
echo ""

# Step 1 — Create ECR repository (if it doesn't exist)
echo "▶ Step 1/5: Setting up ECR repository..."
"$AWS_CMD" ecr describe-repositories --repository-names $SERVICE_NAME \
    --region $AWS_REGION 2>/dev/null || \
"$AWS_CMD" ecr create-repository \
    --repository-name $SERVICE_NAME \
    --region $AWS_REGION \
    --image-scanning-configuration scanOnPush=true \
    --query "repository.repositoryUri" \
    --output text
echo "  ✓ ECR repository ready"

# Step 2 — Authenticate Docker to ECR
echo "▶ Step 2/5: Authenticating Docker to ECR..."
"$AWS_CMD" ecr get-login-password --region $AWS_REGION | \
    docker login --username AWS --password-stdin \
    $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com
echo "  ✓ Docker authenticated"

# Step 3 — Build and push image
echo "▶ Step 3/5: Building Docker image..."
docker build -t $SERVICE_NAME .
docker tag $SERVICE_NAME:latest $IMAGE_NAME:latest
echo "  ✓ Image built"

echo "  Pushing to ECR (this takes 2-4 minutes first time)..."
docker push $IMAGE_NAME:latest
echo "  ✓ Image pushed: $IMAGE_NAME:latest"

# Step 4 — Store secrets in AWS Secrets Manager
echo "▶ Step 4/5: Storing secrets in AWS Secrets Manager..."

store_secret() {
    local NAME=$1
    local VALUE=$2
    "$AWS_CMD" secretsmanager describe-secret --secret-id "mira/$NAME" \
        --region $AWS_REGION 2>/dev/null && \
    "$AWS_CMD" secretsmanager put-secret-value \
        --secret-id "mira/$NAME" \
        --secret-string "$VALUE" \
        --region $AWS_REGION --output text > /dev/null || \
    "$AWS_CMD" secretsmanager create-secret \
        --name "mira/$NAME" \
        --secret-string "$VALUE" \
        --region $AWS_REGION --output text > /dev/null
    echo "  ✓ mira/$NAME stored"
}

if [ -f .env ]; then
    source .env
    store_secret "OPENAI_API_KEY"           "$OPENAI_API_KEY"
    store_secret "SUPABASE_URL"             "$SUPABASE_URL"
    store_secret "SUPABASE_SERVICE_ROLE_KEY" "$SUPABASE_SERVICE_ROLE_KEY"
    store_secret "SUPABASE_DB_PASSWORD"     "$SUPABASE_DB_PASSWORD"
    store_secret "SUPABASE_DIRECT_URL"      "$SUPABASE_DIRECT_URL"
    store_secret "MIRA_JWT_SECRET"          "$MIRA_JWT_SECRET"
else
    echo "  ⚠ .env not found — add secrets manually in AWS Secrets Manager"
fi

# Step 5 — Create/update App Runner service
echo "▶ Step 5/5: Deploying to App Runner..."

# Create IAM role for App Runner if not exists
ROLE_NAME="AppRunnerECRAccessRole"
"$AWS_CMD" iam get-role --role-name $ROLE_NAME 2>/dev/null || {
    "$AWS_CMD" iam create-role \
        --role-name $ROLE_NAME \
        --assume-role-policy-document '{
            "Version":"2012-10-17",
            "Statement":[{
                "Effect":"Allow",
                "Principal":{"Service":"build.apprunner.amazonaws.com"},
                "Action":"sts:AssumeRole"
            }]
        }' --output text > /dev/null
    "$AWS_CMD" iam attach-role-policy \
        --role-name $ROLE_NAME \
        --policy-arn arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess
    echo "  ✓ IAM role created"
}

ROLE_ARN=$("$AWS_CMD" iam get-role --role-name $ROLE_NAME \
    --query "Role.Arn" --output text)

# Build environment variables JSON from .env
ENV_VARS=$(cat << EOF
[
    {"name": "MIRA_ENV",       "value": "production"},
    {"name": "OPENAI_API_KEY", "value": "$OPENAI_API_KEY"},
    {"name": "SUPABASE_URL",   "value": "$SUPABASE_URL"},
    {"name": "SUPABASE_SERVICE_ROLE_KEY", "value": "$SUPABASE_SERVICE_ROLE_KEY"},
    {"name": "SUPABASE_DB_PASSWORD",      "value": "$SUPABASE_DB_PASSWORD"},
    {"name": "SUPABASE_DIRECT_URL",       "value": "$SUPABASE_DIRECT_URL"},
    {"name": "MIRA_JWT_SECRET",           "value": "$MIRA_JWT_SECRET"}
]
EOF
)

# Check if service exists → create or update
SERVICE_EXISTS=$("$AWS_CMD" apprunner list-services \
    --region $AWS_REGION \
    --query "ServiceSummaryList[?ServiceName=='$SERVICE_NAME'].ServiceArn" \
    --output text)

if [ -z "$SERVICE_EXISTS" ]; then
    echo "  Creating new App Runner service..."
    SERVICE_ARN=$("$AWS_CMD" apprunner create-service \
        --service-name $SERVICE_NAME \
        --region $AWS_REGION \
        --source-configuration "{
            \"ImageRepository\": {
                \"ImageIdentifier\": \"$IMAGE_NAME:latest\",
                \"ImageConfiguration\": {
                    \"Port\": \"8080\",
                    \"RuntimeEnvironmentVariables\": $(echo $ENV_VARS | tr -d '\n')
                },
                \"ImageRepositoryType\": \"ECR\"
            },
            \"AutoDeploymentsEnabled\": false,
            \"AuthenticationConfiguration\": {
                \"AccessRoleArn\": \"$ROLE_ARN\"
            }
        }" \
        --instance-configuration '{
            "Cpu": "1 vCPU",
            "Memory": "2 GB"
        }' \
        --health-check-configuration '{
            "Protocol": "HTTP",
            "Path": "/_stcore/health",
            "Interval": 10,
            "Timeout": 5,
            "HealthyThreshold": 1,
            "UnhealthyThreshold": 5
        }' \
        --query "Service.ServiceArn" \
        --output text)
else
    echo "  Updating existing App Runner service..."
    "$AWS_CMD" apprunner update-service \
        --service-arn $SERVICE_EXISTS \
        --region $AWS_REGION \
        --source-configuration "{
            \"ImageRepository\": {
                \"ImageIdentifier\": \"$IMAGE_NAME:latest\",
                \"ImageConfiguration\": {
                    \"Port\": \"8080\",
                    \"RuntimeEnvironmentVariables\": $(echo $ENV_VARS | tr -d '\n')
                },
                \"ImageRepositoryType\": \"ECR\"
            },
            \"AuthenticationConfiguration\": {
                \"AccessRoleArn\": \"$ROLE_ARN\"
            }
        }" > /dev/null
    SERVICE_ARN=$SERVICE_EXISTS
fi

echo "  ⏳ Waiting for deployment (takes 3-5 minutes)..."
"$AWS_CMD" apprunner wait service-running \
    --service-arn $SERVICE_ARN \
    --region $AWS_REGION 2>/dev/null || true

SERVICE_URL=$("$AWS_CMD" describe-service \
    --service-arn $SERVICE_ARN \
    --region $AWS_REGION \
    --query "Service.ServiceUrl" \
    --output text)

echo ""
echo "✅ MIRA deployed successfully!"
echo ""
echo "🌐 Live URL: https://$SERVICE_URL"
echo ""
echo "Login credentials:"
echo "  Clinician : clinician@mira.dev / mira_clinician_2024"
echo "  Admin     : admin@mira.dev / mira_admin_2024"
echo ""
echo "AWS Console: https://$AWS_REGION.console.aws.amazon.com/apprunner"