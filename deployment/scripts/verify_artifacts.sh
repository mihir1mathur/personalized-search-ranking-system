#!/usr/bin/env bash
# =============================================================================
# verify_artifacts.sh  --  pre-flight check that every RUNTIME file exists.
# -----------------------------------------------------------------------------
# Run from the repo root on the instance BEFORE starting the API. Exits non-zero
# if any required artifact is missing, so it is safe to use in CI / a deploy
# gate. Mirrors exactly what api/services/search_service.py loads.
#
#   bash deployment/scripts/verify_artifacts.sh
# =============================================================================
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
fail=0

check() {  # $1=path  $2=required|optional  $3=description
    if [[ -e "$1" ]]; then
        printf "  [ OK ]  %-45s %s\n" "$1" "$3"
    elif [[ "$2" == "optional" ]]; then
        printf "  [WARN]  %-45s %s (optional, absent)\n" "$1" "$3"
    else
        printf "  [FAIL]  %-45s %s (REQUIRED, MISSING)\n" "$1" "$3"
        fail=1
    fi
}

echo "Runtime artifact check (repo: $ROOT)"
echo "-------------------------------------------------------------------------"
check "data/processed/sample_esci_50k.parquet"     required "corpus + product metadata (~73 MB)"
check "results/product_embeddings_minilm.npy"      required "cached product embeddings (~71 MB)"
check "results/product_embeddings_ids.npy"         required "embedding id alignment (~2 MB)"
check "models/ms-marco-MiniLM-L-6-v2/model.safetensors" required "cross-encoder weights (~87 MB)"
check "models/ms-marco-MiniLM-L-6-v2/config.json"  required "cross-encoder config"
check "models/ms-marco-MiniLM-L-6-v2/tokenizer.json" required "cross-encoder tokenizer"
check "models/ltr_lightgbm.txt"                    required "trained LambdaMART model"
check "results/week5_ce_cache.pkl"                 optional "warm cross-encoder score cache"

echo "-------------------------------------------------------------------------"
echo "Embedding model (loaded by repo-id via the HF cache, NOT from models/):"
HF="${HF_HOME:-$HOME/.cache/huggingface}/hub/models--sentence-transformers--all-MiniLM-L6-v2"
if [[ -d "$HF" ]]; then
    printf "  [ OK ]  %s\n" "$HF"
else
    printf "  [FAIL]  %s\n" "$HF"
    echo "          -> stage it (deployment/scripts/stage_artifacts.sh) OR set"
    echo "             SEARCH_HF_HUB_OFFLINE=false for the first boot to download."
    fail=1
fi

echo "-------------------------------------------------------------------------"
if [[ "$fail" -eq 0 ]]; then
    echo "RESULT: all required artifacts present. Safe to start the API."
else
    echo "RESULT: MISSING required artifacts. The API will start DEGRADED (503)."
fi
exit "$fail"
