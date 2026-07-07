#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════
# deploy.sh — Deploy MIRA to GCP Cloud Run
# Run from your Mira_project root directory: bash deploy.sh
# ══════════════════════════════════════════════════════════════════════════

set -e  # Exit on any error

# ── CONFIG — update these ─────────────────────────────────────────────────
PROJECT_ID="mira-clinical-501719"        # gcloud projects list
REGION="asia-south1"                     # closest to India (Mumbai)
SERVICE_NAME="mira-clinical"
IMAGE_NAME="gcr.io/$PROJECT_ID/$SERVICE_NAME"
# ─────────────────────────────────────────────────────────────────────────

echo "🏥 MIRA — Cloud Run Deployment"
echo "================================"
echo "Project : $PROJECT_ID"
echo "Region  : $REGION"
echo "Service : $SERVICE_NAME"
echo ""

# Step 1 — Authenticate and set project
echo "▶ Step 1/6: Setting GCP project..."
gcloud config set project $PROJECT_ID

# Step 2 — Enable required APIs (safe to run multiple times)
echo "▶ Step 2/6: Enabling required APIs..."
gcloud services enable \
    run.googleapis.com \
    containerregistry.googleapis.com \
    secretmanager.googleapis.com \
    --quiet

# Step 3 — Build Docker image
echo "▶ Step 3/6: Building Docker image..."
docker build -t $IMAGE_NAME .

# Step 4 — Push to Google Container Registry
echo "▶ Step 4/6: Pushing to Container Registry..."
docker push $IMAGE_NAME

# Step 5 — Store secrets in GCP Secret Manager
echo "▶ Step 5/6: Storing secrets in GCP Secret Manager..."

store_secret() {
    local SECRET_NAME=$1
    local SECRET_VALUE=$2
    # Create secret if it doesn't exist, then add version
    gcloud secrets describe $SECRET_NAME --quiet 2>/dev/null || \
        gcloud secrets create $SECRET_NAME --replication-policy="automatic" --quiet
    echo -n "$SECRET_VALUE" | gcloud secrets versions add $SECRET_NAME --data-file=- --quiet
    echo "  ✓ $SECRET_NAME stored"
}

# Load from .env file
if [ -f .env ]; then
    source .env
    store_secret "OPENAI_API_KEY"          "$OPENAI_API_KEY"
    store_secret "SUPABASE_URL"            "$SUPABASE_URL"
    store_secret "SUPABASE_SERVICE_ROLE_KEY" "$SUPABASE_SERVICE_ROLE_KEY"
    store_secret "SUPABASE_DB_PASSWORD"    "$SUPABASE_DB_PASSWORD"
    store_secret "SUPABASE_DIRECT_URL"     "$SUPABASE_DIRECT_URL"
    store_secret "MIRA_JWT_SECRET"         "$MIRA_JWT_SECRET"
else
    echo "  ⚠ .env not found — set secrets manually in GCP Secret Manager"
fi

# Step 6 — Deploy to Cloud Run
echo "▶ Step 6/6: Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
    --image $IMAGE_NAME \
    --platform managed \
    --region $REGION \
    --port 8080 \
    --memory 2Gi \
    --cpu 2 \
    --timeout 300 \
    --concurrency 10 \
    --min-instances 0 \
    --max-instances 3 \
    --set-env-vars "MIRA_ENV=production,GCP_PROJECT_ID=$PROJECT_ID" \
    --set-secrets "\
OPENAI_API_KEY=OPENAI_API_KEY:latest,\
SUPABASE_URL=SUPABASE_URL:latest,\
SUPABASE_SERVICE_ROLE_KEY=SUPABASE_SERVICE_ROLE_KEY:latest,\
SUPABASE_DB_PASSWORD=SUPABASE_DB_PASSWORD:latest,\
SUPABASE_DIRECT_URL=SUPABASE_DIRECT_URL:latest,\
MIRA_JWT_SECRET=MIRA_JWT_SECRET:latest" \
    --allow-unauthenticated \
    --quiet

echo ""
echo "✅ MIRA deployed successfully!"
echo ""
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME \
    --platform managed --region $REGION \
    --format "value(status.url)")
echo "🌐 Live URL: $SERVICE_URL"
echo ""
echo "Login credentials:"
echo "  Clinician : clinician@mira.dev / mira_clinician_2024"
echo "  Admin     : admin@mira.dev / mira_admin_2024"