#!/bin/bash
# ============================================================================
# Pi Scanner - One-Command Installation
# ============================================================================
# Usage: curl -sSL <github-raw-url>/pi/install.sh | bash
#   or:  chmod +x install.sh && ./install.sh
#
# What this does:
#   1. Installs system dependencies (ffmpeg, alsa-utils, python3-venv)
#   2. Creates ~/scanner directory with Python virtual environment
#   3. Installs Python packages
#   4. Installs systemd services
#   5. Initializes the SQLite database
#   6. Prints next steps
# ============================================================================
set -e

SCANNER_DIR="$HOME/scanner"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "============================================"
echo "  Pi Scanner - Installation"
echo "============================================"
echo ""

# --- System packages ---
echo "[1/6] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv python3-dev ffmpeg alsa-utils

# --- Create directory ---
echo "[2/6] Creating $SCANNER_DIR..."
mkdir -p "$SCANNER_DIR/clips"

# --- Copy project files ---
echo "[3/6] Copying project files..."
cp "$SCRIPT_DIR"/*.py "$SCANNER_DIR/"
cp "$SCRIPT_DIR"/*.sh "$SCANNER_DIR/" 2>/dev/null || true
chmod +x "$SCANNER_DIR"/*.sh 2>/dev/null || true

# --- Python venv ---
echo "[4/6] Setting up Python environment..."
if [ ! -d "$SCANNER_DIR/venv" ]; then
    python3 -m venv "$SCANNER_DIR/venv"
fi
source "$SCANNER_DIR/venv/bin/activate"
pip install --quiet --upgrade pip
pip install --quiet -r "$SCRIPT_DIR/requirements.txt"

# --- Systemd services ---
echo "[5/6] Installing systemd services..."
sudo cp "$SCRIPT_DIR/systemd/"*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable pi-scanner pi-dashboard pi-transcriber

# --- Initialize database ---
echo "[6/6] Initializing database..."
cd "$SCANNER_DIR"
python3 -c "import scanner_db; scanner_db.init_db(); print('  Database ready at', scanner_db.DB_PATH)"

echo ""
echo "============================================"
echo "  Installation Complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo ""
echo "  1. Find your audio device:"
echo "     arecord -l"
echo ""
echo "  2. Find your serial port:"
echo "     ls /dev/ttyACM* /dev/ttyUSB*"
echo ""
echo "  3. Edit configuration:"
echo "     nano ~/scanner/config.py"
echo "     (set AUDIO_DEVICE and SERIAL_PORT)"
echo ""
echo "  4. Start services:"
echo "     sudo systemctl start pi-scanner pi-dashboard pi-transcriber"
echo ""
echo "  5. Access dashboard:"
echo "     http://$(hostname -I | awk '{print $1}'):8080"
echo ""
echo "  Optional: For GPU transcription, set GPU_SERVER_URL in config.py"
echo ""
