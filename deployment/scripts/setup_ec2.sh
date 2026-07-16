#!/usr/bin/env bash
# =============================================================================
# setup_ec2.sh  --  provision an Ubuntu 22.04 EC2 instance for the search app.
# -----------------------------------------------------------------------------
# Idempotent-ish bootstrap: system packages, service user, Python venv, deps,
# systemd units, nginx. Run AS ROOT (or with sudo) on the instance AFTER the
# repo + runtime artifacts have been placed at /opt/search-ranking.
#
#   sudo bash deployment/scripts/setup_ec2.sh
#
# It does NOT create AWS resources and does NOT start anything until the very
# end (and even then only enables the units). Review before running.
# =============================================================================
set -euo pipefail

APP_DIR=/opt/search-ranking
APP_USER=searchapp
ENV_DIR=/etc/search-ranking
PY=python3.10

echo "==> 1/7 System packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y \
    ${PY} ${PY}-venv ${PY}-dev \
    build-essential \
    nginx \
    curl git

echo "==> 2/7 Service user '${APP_USER}'"
if ! id -u "${APP_USER}" >/dev/null 2>&1; then
    useradd --system --create-home --shell /usr/sbin/nologin "${APP_USER}"
fi

echo "==> 3/7 Ownership of ${APP_DIR}"
if [[ ! -d "${APP_DIR}" ]]; then
    echo "ERROR: ${APP_DIR} does not exist. Clone the repo there first." >&2
    exit 1
fi
chown -R "${APP_USER}:${APP_USER}" "${APP_DIR}"
mkdir -p "${APP_DIR}/logs"
chown "${APP_USER}:${APP_USER}" "${APP_DIR}/logs"

echo "==> 4/7 Python virtualenv + dependencies"
sudo -u "${APP_USER}" ${PY} -m venv "${APP_DIR}/.venv"
sudo -u "${APP_USER}" "${APP_DIR}/.venv/bin/pip" install --upgrade pip wheel
# Backend deps. torch (CPU) is pulled transitively by sentence-transformers;
# to force the smaller CPU wheel explicitly, uncomment the next line first:
# sudo -u "${APP_USER}" "${APP_DIR}/.venv/bin/pip" install torch --index-url https://download.pytorch.org/whl/cpu
sudo -u "${APP_USER}" "${APP_DIR}/.venv/bin/pip" install -r "${APP_DIR}/requirements.txt"
# Frontend deps.
sudo -u "${APP_USER}" "${APP_DIR}/.venv/bin/pip" install -r "${APP_DIR}/frontend/requirements.txt"

echo "==> 5/7 Environment file"
mkdir -p "${ENV_DIR}"
if [[ ! -f "${ENV_DIR}/search.env" ]]; then
    cp "${APP_DIR}/deployment/env/production.env.example" "${ENV_DIR}/search.env"
    echo "    wrote ${ENV_DIR}/search.env (review it!)"
fi
chown root:"${APP_USER}" "${ENV_DIR}/search.env"
chmod 640 "${ENV_DIR}/search.env"

echo "==> 6/7 systemd units"
cp "${APP_DIR}/deployment/systemd/search-api.service"      /etc/systemd/system/
cp "${APP_DIR}/deployment/systemd/search-frontend.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable search-api search-frontend

echo "==> 7/7 nginx"
cp "${APP_DIR}/deployment/nginx/search-app.conf" /etc/nginx/sites-available/search-app
ln -sf /etc/nginx/sites-available/search-app /etc/nginx/sites-enabled/search-app
rm -f /etc/nginx/sites-enabled/default
nginx -t

cat <<'EOF'

============================================================================
Provisioning complete. BEFORE starting the services, verify runtime artifacts
are present (see deployment/scripts/verify_artifacts.sh) and the HF embedding
model is staged (see deployment/scripts/stage_artifacts.sh).

Then:
    sudo systemctl start search-api
    # wait ~60s for model load, then:
    curl -s http://127.0.0.1:8000/health | head
    sudo systemctl start search-frontend
    sudo systemctl reload nginx

Watch logs:
    journalctl -u search-api -f
============================================================================
EOF
