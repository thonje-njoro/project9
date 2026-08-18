#!/bin/bash
# Project9 GCP VM Setup Script
# Run once on a fresh e2-medium VM as gathimbu user
set -euo pipefail

echo "=== Project9 VM Setup ==="

# System packages
sudo apt-get update -y
sudo apt-get install -y python3.11 python3.11-venv python3-pip git screen htop

# Clone repo (or use existing)
cd /home/gathimbu
if [ ! -d "project9" ]; then
    git clone https://github.com/thonje-njoro/project9.git
fi
cd project9

# Python environment
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Create .env (edit manually after setup)
if [ ! -f backtest/.env ]; then
    cat > backtest/.env << 'EOF'
ALPACA_API_KEY=your_alpaca_api_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_key_here
MIMO_API_KEY=your_mimo_api_key_here
LSE_API_KEY=your_lse_api_key_here
EOF
    chmod 600 backtest/.env
    echo "Created backtest/.env — EDIT THIS FILE with real API keys"
fi

# Create state directory
mkdir -p backtest/paper_trading/state

# Install systemd services
sudo cp systemd/paper-trader.service /etc/systemd/system/
sudo cp systemd/ai-loop.timer /etc/systemd/system/
sudo cp systemd/ai-loop.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable paper-trader
sudo systemctl enable ai-loop.timer

# Log rotation
sudo tee /etc/logrotate.d/project9 > /dev/null << 'EOF'
/home/gathimbu/project9/backtest/paper_trading/state/*.log {
    daily
    rotate 14
    compress
    missingok
    notifempty
}
EOF

echo ""
echo "=== Setup Complete ==="
echo "Next steps:"
echo "  1. Edit backtest/.env with real API keys"
echo "  2. Run: python backtest/main.py --validate"
echo "  3. Start: sudo systemctl start paper-trader"
echo "  4. Check: journalctl -u paper-trader -f"
