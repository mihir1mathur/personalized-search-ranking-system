#!/usr/bin/env bash
# =============================================================================
# stage_artifacts.sh  --  get the large RUNTIME artifacts onto the instance.
# -----------------------------------------------------------------------------
# The repo's CODE is tiny, but the app needs ~325 MB of binary artifacts to run.
# Some are tracked in git; some are NOT (see the Deployment Readiness Report).
# This script documents the THREE supported transfer strategies. It does not
# run automatically -- read it and run the block that matches your setup.
#
# Artifacts needed at runtime (all relative to the repo root unless noted):
#   data/processed/sample_esci_50k.parquet        ~73 MB   (tracked in git)
#   results/product_embeddings_minilm.npy          ~71 MB   (UNTRACKED)
#   results/product_embeddings_ids.npy             ~2 MB    (UNTRACKED)
#   models/ms-marco-MiniLM-L-6-v2/                 ~88 MB   (UNTRACKED)
#   models/ltr_lightgbm.txt                        ~20 KB   (UNTRACKED)
#   results/week5_ce_cache.pkl                     ~1 MB    (UNTRACKED, optional)
#   ~/.cache/huggingface/.../all-MiniLM-L6-v2      ~88 MB   (OUTSIDE the repo)
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# STRATEGY A -- SCP straight from your dev machine (simplest for one instance)
# ---------------------------------------------------------------------------
# Run these FROM YOUR LOCAL machine (Git Bash), not on EC2. Adjust HOST/KEY.
#
#   HOST=ubuntu@<EC2_PUBLIC_DNS>
#   KEY=~/.ssh/your-key.pem
#   APP=/opt/search-ranking
#
#   # code (or use `git clone` on the instance instead)
#   # then the untracked artifacts:
#   scp -i "$KEY" results/product_embeddings_minilm.npy  "$HOST:$APP/results/"
#   scp -i "$KEY" results/product_embeddings_ids.npy     "$HOST:$APP/results/"
#   scp -i "$KEY" results/week5_ce_cache.pkl             "$HOST:$APP/results/"
#   scp -i "$KEY" models/ltr_lightgbm.txt                "$HOST:$APP/models/"
#   scp -i "$KEY" -r models/ms-marco-MiniLM-L-6-v2       "$HOST:$APP/models/"
#   # data parquet (if you did NOT clone with it / used a shallow export):
#   scp -i "$KEY" data/processed/sample_esci_50k.parquet "$HOST:$APP/data/processed/"
#
#   # the HF embedding-model cache (LOCAL path -> the service user's HOME on EC2)
#   scp -i "$KEY" -r ~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2 \
#        "$HOST:/tmp/emb-model"
#   # then on the instance, move it into the searchapp user's HF cache:
#   #   sudo mkdir -p /home/searchapp/.cache/huggingface/hub
#   #   sudo mv /tmp/emb-model /home/searchapp/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2
#   #   sudo chown -R searchapp:searchapp /home/searchapp/.cache

# ---------------------------------------------------------------------------
# STRATEGY B -- stage via S3 (repeatable, best if you will redeploy)
# ---------------------------------------------------------------------------
# FROM LOCAL: upload once.
#   BUCKET=s3://my-search-artifacts
#   aws s3 cp results/product_embeddings_minilm.npy  $BUCKET/results/
#   aws s3 cp results/product_embeddings_ids.npy     $BUCKET/results/
#   aws s3 cp results/week5_ce_cache.pkl             $BUCKET/results/
#   aws s3 cp models/ltr_lightgbm.txt                $BUCKET/models/
#   aws s3 cp --recursive models/ms-marco-MiniLM-L-6-v2 $BUCKET/models/ms-marco-MiniLM-L-6-v2/
#   aws s3 cp data/processed/sample_esci_50k.parquet $BUCKET/data/processed/
#   aws s3 cp --recursive ~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2 \
#         $BUCKET/hf/models--sentence-transformers--all-MiniLM-L6-v2/
#
# ON THE INSTANCE (needs an IAM instance-profile with s3:GetObject on the bucket):
#   BUCKET=s3://my-search-artifacts; APP=/opt/search-ranking
#   aws s3 cp $BUCKET/results/ $APP/results/ --recursive
#   aws s3 cp $BUCKET/models/  $APP/models/  --recursive
#   aws s3 cp $BUCKET/data/processed/ $APP/data/processed/ --recursive
#   sudo -u searchapp aws s3 cp $BUCKET/hf/ /home/searchapp/.cache/huggingface/hub/ --recursive

# ---------------------------------------------------------------------------
# STRATEGY C -- no staging; let HF download the embedding model on first boot
# ---------------------------------------------------------------------------
# Only the HF embedding model can be obtained this way (the .npy embeddings,
# cross-encoder, LTR model, and parquet still come from git/scp/S3).
# In /etc/search-ranking/search.env set, JUST for the first start:
#     SEARCH_HF_HUB_OFFLINE=false
#     HF_HUB_OFFLINE=0
#     TRANSFORMERS_OFFLINE=0
# start search-api, confirm /health is "ok", then flip all three back to
# true/1/1 and `sudo systemctl restart search-api`. Requires egress to
# huggingface.co (open by default on EC2).

echo "This script is documentation-only. Open it and run the block for your"
echo "chosen strategy (A: scp, B: S3, C: HF download-on-first-boot)."
