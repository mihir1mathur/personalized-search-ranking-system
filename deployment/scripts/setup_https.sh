#!/usr/bin/env bash
# =============================================================================
# setup_https.sh  --  attach a custom domain + Let's Encrypt TLS to the running
#                     nginx deployment via Certbot's nginx plugin.
# -----------------------------------------------------------------------------
# Run this AFTER the app is already serving over HTTP by public IP AND after the
# domain's A record points at this box (attach an Elastic IP first -- see
# deployment/README.md "Custom Domain and HTTPS"). It:
#   * validates args + a working nginx config,
#   * installs certbot + the nginx plugin only if missing,
#   * backs up the live nginx site (timestamped) before editing,
#   * sets server_name to your domain (and www.<domain> when you pass a bare apex),
#   * runs certbot --nginx to obtain the cert and wire up an HTTP->HTTPS redirect,
#   * dry-run tests automatic renewal, and reloads nginx only after `nginx -t`.
#
#   sudo bash deployment/scripts/setup_https.sh search.example.com you@example.com
#
# It NEVER creates AWS resources, never touches your registrar, never prints
# secrets, and is safe to re-run (idempotent): re-running just re-validates and
# lets certbot no-op / renew as needed. Review before running.
# =============================================================================
set -euo pipefail

# ---- pretty, actionable failures -------------------------------------------
die()  { printf '\nERROR: %s\n' "$*" >&2; exit 1; }
note() { printf '==> %s\n' "$*"; }

NGINX_SITE="/etc/nginx/sites-available/search-app"

# ---- 0) must be root -------------------------------------------------------
if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
    die "must run as root. Re-run with: sudo bash $0 <domain> <email>"
fi

# ---- 1) validate arguments -------------------------------------------------
DOMAIN="${1:-}"
EMAIL="${2:-}"
if [[ -z "$DOMAIN" || -z "$EMAIL" ]]; then
    die "usage: sudo bash $0 <domain> <email>
       e.g. sudo bash $0 search.example.com you@example.com
       - <domain>: the FQDN whose A record already points at THIS server
       - <email> : contact for Let's Encrypt expiry / security notices"
fi
# Cheap sanity checks (not a full RFC validator -- certbot does the real check).
if [[ ! "$DOMAIN" =~ ^([A-Za-z0-9]([A-Za-z0-9-]*[A-Za-z0-9])?\.)+[A-Za-z]{2,}$ ]]; then
    die "'$DOMAIN' does not look like a fully-qualified domain name."
fi
if [[ ! "$EMAIL" =~ ^[^@[:space:]]+@[^@[:space:]]+\.[^@[:space:]]+$ ]]; then
    die "'$EMAIL' does not look like a valid email address."
fi

# Include the www. host in the cert only when a bare apex (exactly two labels)
# was given; skip it for subdomains like search.example.com.
CERT_DOMAINS=("$DOMAIN")
if [[ "$(grep -o '\.' <<<"$DOMAIN" | wc -l)" -eq 1 ]]; then
    CERT_DOMAINS+=("www.${DOMAIN}")
fi
SERVER_NAMES="${CERT_DOMAINS[*]}"   # space-separated, for nginx server_name

note "Domain(s) for the certificate: ${CERT_DOMAINS[*]}"
note "Contact email:                 ${EMAIL}"

# ---- 2) the active nginx site must exist -----------------------------------
[[ -f "$NGINX_SITE" ]] || die "active nginx site not found at ${NGINX_SITE}.
       Run deployment/scripts/setup_ec2.sh first so the HTTP site is installed."

# ---- 3) validate the CURRENT nginx config before we touch anything ---------
note "Validating current nginx configuration (nginx -t)"
nginx -t || die "nginx config is invalid BEFORE any change. Fix that first;
       this script refuses to edit a broken config."

# ---- 4) install certbot + nginx plugin only if needed ----------------------
if command -v certbot >/dev/null 2>&1 \
   && certbot plugins 2>/dev/null | grep -q '\bnginx\b'; then
    note "certbot + nginx plugin already present; skipping install."
else
    note "Installing certbot + python3-certbot-nginx"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -y  || die "apt-get update failed."
    apt-get install -y certbot python3-certbot-nginx \
        || die "failed to install certbot / python3-certbot-nginx."
fi

# ---- 5) timestamped backup of the live site BEFORE editing -----------------
# Date.now-style stamp; used only for the backup filename.
STAMP="$(date +%Y%m%d-%H%M%S)"
BACKUP="${NGINX_SITE}.bak.${STAMP}"
cp -a "$NGINX_SITE" "$BACKUP" || die "could not create backup ${BACKUP}."
note "Backed up active nginx site -> ${BACKUP}"
echo "    (rollback: sudo cp -a '${BACKUP}' '${NGINX_SITE}' && sudo nginx -t && sudo systemctl reload nginx)"

# ---- 6) set server_name safely ---------------------------------------------
# Replace the FIRST `server_name ...;` line (the HTTP :80 block) with our
# domain(s). Certbot will clone the relevant directives into the :443 block it
# creates, so we only need to fix the one server_name here.
if grep -qE '^\s*server_name\s' "$NGINX_SITE"; then
    sed -i -E "0,/^\s*server_name\s.*;/s//    server_name ${SERVER_NAMES};/" "$NGINX_SITE"
else
    die "no 'server_name' directive found in ${NGINX_SITE}; refusing to guess.
       Restore is not needed (nothing changed except the backup copy)."
fi
note "Set server_name to: ${SERVER_NAMES}"

# ---- 7) re-validate, then reload so certbot sees the updated server_name ----
if ! nginx -t; then
    note "nginx -t failed after editing server_name; restoring backup."
    cp -a "$BACKUP" "$NGINX_SITE"
    die "reverted ${NGINX_SITE} from ${BACKUP}. No certificate was requested."
fi
systemctl reload nginx || die "nginx reload failed after server_name update."

# ---- 8) obtain the certificate + enable HTTP->HTTPS redirect ---------------
# --redirect          : add the 80 -> 443 redirect automatically
# --nginx             : use the nginx plugin (edits config, no downtime)
# --keep-until-expiring: idempotent re-runs won't force a needless reissue
# -n --agree-tos -m   : non-interactive; ToS accepted; contact email set
note "Requesting certificate via certbot (nginx plugin)"
CERTBOT_DOMAIN_ARGS=()
for d in "${CERT_DOMAINS[@]}"; do CERTBOT_DOMAIN_ARGS+=(-d "$d"); done
certbot --nginx \
    "${CERTBOT_DOMAIN_ARGS[@]}" \
    --redirect \
    --keep-until-expiring \
    --non-interactive \
    --agree-tos \
    -m "$EMAIL" \
    || die "certbot failed. Common causes:
       - DNS A record for ${DOMAIN} is not (yet) pointing at THIS server's IP
       - port 80/443 blocked in the EC2 security group
       - Let's Encrypt rate limit hit (retry later)
       Your HTTP site + backup are intact; re-run once DNS/ports are correct."

# ---- 9) verify automatic renewal (safe dry-run; issues nothing) ------------
note "Verifying automatic renewal (certbot renew --dry-run)"
certbot renew --dry-run \
    || die "renewal dry-run failed. The certificate is installed, but auto-renew
       needs attention -- check 'systemctl status certbot.timer' and the nginx
       plugin. Re-run the dry-run after fixing."

# ---- 10) final nginx syntax check + reload ---------------------------------
note "Final nginx validation + reload"
nginx -t || die "final nginx -t failed (unexpected after certbot). Inspect
       ${NGINX_SITE} and roll back with the backup printed above."
systemctl reload nginx || die "final nginx reload failed."

cat <<EOF

============================================================================
HTTPS is configured for: ${CERT_DOMAINS[*]}

  * TLS terminates at nginx; backends stay on loopback (8000/8501 unchanged).
  * HTTP (80) now redirects to HTTPS (443) automatically.
  * Routing preserved:  /  -> Streamlit,  /api/ -> FastAPI,  /healthz -> /health
  * Renewal is handled by the system certbot timer (dry-run passed above).

Verify:
    curl -I https://${DOMAIN}/            # expect HTTP/2 200 (or 301 on :80)
    curl -s  https://${DOMAIN}/healthz    # expect the health JSON

Rollback (revert nginx to the pre-HTTPS config):
    sudo cp -a '${BACKUP}' '${NGINX_SITE}'
    sudo nginx -t && sudo systemctl reload nginx
============================================================================
EOF
