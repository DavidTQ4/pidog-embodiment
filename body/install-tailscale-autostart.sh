#!/usr/bin/env bash
set -euo pipefail

REPO_BODY=/home/david/pidog-embodiment/body
CONFIG=/etc/pidog-brain-link.env
BRAIN_HOST=${1:-}
BRAIN_USER=${2:-d_s_h}
PI_USER=david
PI_HOME=/home/david
KEY="$PI_HOME/.ssh/pidog_brain_ed25519"

log() { printf '[install-tailscale] %s\n' "$*"; }
fail() { printf '[install-tailscale] ERROR: %s\n' "$*" >&2; exit 1; }

[[ ${EUID} -eq 0 ]] || fail "run with sudo"
[[ -d "$REPO_BODY" ]] || fail "missing $REPO_BODY"
for file in pidog-start pidog-boot.service pidog-brain-link pidog-brain-link.service pidog-brain-link.env.example; do
    [[ -f "$REPO_BODY/$file" ]] || fail "missing $REPO_BODY/$file"
done

command -v tailscale >/dev/null 2>&1 ||
    fail "Tailscale is not installed. Install it first, then run this installer again."

log "installing PiDog boot components"
install -m 755 "$REPO_BODY/pidog-start" /usr/local/sbin/pidog-start
install -m 644 "$REPO_BODY/pidog-boot.service" /etc/systemd/system/pidog-boot.service
install -m 755 "$REPO_BODY/pidog-brain-link" /usr/local/sbin/pidog-brain-link
install -m 644 "$REPO_BODY/pidog-brain-link.service" /etc/systemd/system/pidog-brain-link.service

install -d -m 700 -o "$PI_USER" -g "$PI_USER" "$PI_HOME/.ssh"
if [[ ! -f "$KEY" ]]; then
    log "generating dedicated Pi-to-brain SSH key"
    sudo -u "$PI_USER" ssh-keygen -q -t ed25519 -N '' -C 'pidog-brain-link' -f "$KEY"
fi

if [[ ! -e "$CONFIG" ]]; then
    install -m 640 -o root -g "$PI_USER" "$REPO_BODY/pidog-brain-link.env.example" "$CONFIG"
fi

chown root:"$PI_USER" "$CONFIG"
chmod 640 "$CONFIG"

if [[ -n "$BRAIN_HOST" ]]; then
    sed -i "s/^PIDOG_BRAIN_HOST=.*/PIDOG_BRAIN_HOST=$BRAIN_HOST/" "$CONFIG"
    sed -i "s/^PIDOG_BRAIN_USER=.*/PIDOG_BRAIN_USER=$BRAIN_USER/" "$CONFIG"
fi

systemctl daemon-reload
systemctl enable tailscaled.service pidog-boot.service >/dev/null

log "starting Tailscale and local PiDog boot service"
systemctl start tailscaled.service
systemctl restart pidog-boot.service

if [[ -n "$BRAIN_HOST" ]]; then
    systemctl enable pidog-brain-link.service >/dev/null
    log "brain link configured; install this public key on the desktop:"
    printf '\n'
    cat "$KEY.pub"
    printf '\n'
    log "after authorising the key on the desktop, run:"
    printf '  sudo systemctl restart pidog-brain-link.service\n'
else
    log "brain link not enabled because no desktop hostname was supplied"
    log "edit $CONFIG, authorise $KEY.pub on the desktop, then enable the service"
fi

log "local boot automation is installed"
