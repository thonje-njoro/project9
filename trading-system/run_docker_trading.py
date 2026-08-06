#!/usr/bin/env python3
"""
Docker-based Hermes Trading System Runner
Automates the build and execution of the trading loop in a clean Python 3.11 environment.
"""

import subprocess
import sys
import os
from pathlib import Path

PROJECT_DIR = Path("/home/admin1/project9/trading-system")
DOCKERFILE = PROJECT_DIR / "Dockerfile"
DOCKER_IMAGE = "hermes-trading-system:latest"

DOCKERFILE_CONTENT = """FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    git \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy requirements first for better caching
COPY requirements.txt /workspace/requirements.txt

# Install Python dependencies with compatible NumPy version
RUN pip install --no-cache-dir numpy==1.21.5 \\
    && pip install --no-cache-dir -r /workspace/requirements.txt

# Copy the rest of the project
COPY . /workspace/

# Default command
CMD ["bash"]
"""

REQUIREMENTS_CONTENT = """backtesting
pandas
pyarrow
"""

def run_command(cmd, cwd=None, capture=True):
    """Run a shell command and return the result."""
    print(f"🔧 Running: {cmd}")
    result = subprocess.run(
        cmd,
        shell=True,
        cwd=cwd or PROJECT_DIR,
        capture_output=capture,
        text=True
    )
    if result.returncode != 0:
        print(f"❌ Command failed with exit code {result.returncode}")
        if capture:
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
    else:
        if capture and result.stdout:
            print(f"✅ Output: {result.stdout.strip()}")
    return result

def create_dockerfile():
    """Create Dockerfile and requirements.txt."""
    print("📝 Creating Dockerfile and requirements.txt...")
    DOCKERFILE.write_text(DOCKERFILE_CONTENT)
    (PROJECT_DIR / "requirements.txt").write_text(REQUIREMENTS_CONTENT)
    print("✅ Dockerfile created")

def build_docker_image():
    """Build the Docker image."""
    print(f"🐳 Building Docker image: {DOCKER_IMAGE}")
    result = run_command(f"docker build -t {DOCKER_IMAGE} .", capture=False)
    if result.returncode != 0:
        raise RuntimeError("Docker build failed")
    print("✅ Docker image built successfully")

def run_trading_loop():
    """Run the trading loop inside the Docker container."""
    print("🚀 Starting trading loop in Docker container...")
    cmd = (
        f"docker run -it --rm "
        f"-v {PROJECT_DIR}:/workspace "
        f"{DOCKER_IMAGE} "
        f"bash -c 'cd /workspace && source venv/bin/activate && python3 loop/run_loop.py'"
    )
    result = run_command(cmd, capture=False)
    return result.returncode == 0

def main():
    print("=" * 60)
    print("HERMES TRADING SYSTEM - DOCKER AUTOMATION")
    print("=" * 60)
    
    # Check if Docker is available
    result = run_command("docker --version")
    if result.returncode != 0:
        print("❌ Docker not found. Installing...")
        run_command("sudo apt-get update && sudo apt-get install -y docker.io")
        run_command("sudo systemctl start docker")
        run_command("sudo systemctl enable docker")
    
    try:
        create_dockerfile()
        build_docker_image()
        success = run_trading_loop()
        
        if success:
            print("\n✅ Trading loop completed successfully!")
        else:
            print("\n❌ Trading loop failed. Check logs above.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n💥 Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()