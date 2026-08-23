#!/bin/bash
# Run once with sudo on the Pi to set memory limits and prevent OOM freezes.
# Usage: sudo bash setup_memory_limits.sh

set -e

echo "=== Pi Scanner Memory Protection Setup ==="

# 1. Set memory limits on systemd services
echo "[1] Setting memory limits on scanner services..."

# Scanner: max 150MB (capture + audio analysis, no Whisper)
mkdir -p /etc/systemd/system/pi-scanner.service.d
cat > /etc/systemd/system/pi-scanner.service.d/memory.conf << 'EOF'
[Service]
MemoryMax=150M
MemoryHigh=120M
OOMPolicy=restart
Restart=always
RestartSec=5
EOF

# Dashboard: max 80MB (Flask + SQLite queries, no heavy processing)
mkdir -p /etc/systemd/system/pi-dashboard.service.d
cat > /etc/systemd/system/pi-dashboard.service.d/memory.conf << 'EOF'
[Service]
MemoryMax=80M
MemoryHigh=60M
OOMPolicy=restart
Restart=always
RestartSec=5
EOF

# Transcriber: max 400MB (Whisper tiny.en model is ~200MB when loaded)
mkdir -p /etc/systemd/system/pi-transcriber.service.d
cat > /etc/systemd/system/pi-transcriber.service.d/memory.conf << 'EOF'
[Service]
MemoryMax=400M
MemoryHigh=350M
OOMPolicy=restart
Restart=always
RestartSec=10
Nice=15
EOF

# 2. Install udev rule for stable audio device naming
echo "[2] Installing udev rule for USB audio..."
cp /home/pi/scanner/89-scanner-audio.rules /etc/udev/rules.d/ 2>/dev/null || true
udevadm control --reload-rules 2>/dev/null || true

# 3. Disable pipewire/wireplumber (desktop audio, not needed for scanner)
echo "[3] Disabling unnecessary audio services..."
sudo -u pi systemctl --user stop pipewire wireplumber pipewire-pulse 2>/dev/null || true
sudo -u pi systemctl --user disable pipewire wireplumber pipewire-pulse 2>/dev/null || true
sudo -u pi systemctl --user mask pipewire wireplumber pipewire-pulse 2>/dev/null || true

# 4. Ensure swap is adequate
echo "[4] Checking swap..."
SWAP_TOTAL=$(free -m | awk '/Swap:/ {print $2}')
echo "    Current swap: ${SWAP_TOTAL}MB"
if [ "$SWAP_TOTAL" -lt 512 ]; then
    echo "    WARNING: Swap is low. Consider adding a swap file:"
    echo "    sudo dd if=/dev/zero of=/swapfile bs=1M count=1024"
    echo "    sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile"
fi

# 5. Reload systemd
echo "[5] Reloading systemd..."
systemctl daemon-reload

echo ""
echo "=== Done! Memory limits active after next service restart ==="
echo "  pi-scanner:     max 150MB (restart on OOM)"
echo "  pi-dashboard:   max  80MB (restart on OOM)"
echo "  pi-transcriber: max 400MB (restart on OOM)"
echo ""
echo "If a service hits its limit, systemd restarts it (not the whole Pi)."
echo "Run: sudo systemctl restart pi-scanner pi-dashboard pi-transcriber"
