#!/bin/bash

set -e

echo "=========================================="
echo "1. Activating Micromamba Environment"
echo "=========================================="
eval "$(micromamba shell hook --shell bash)"
micromamba activate fql_env

cat << 'EOF' > fast_download.py
import urllib.request
import ogbench
import subprocess
import os

original_urlretrieve = urllib.request.urlretrieve

def fast_urlretrieve(url, filename, reporthook=None, data=None):
    print(f"\n🚀 Intercepted URL: {url}")
    print(f"📦 Target file: {filename}")
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # STEALTH SETTINGS: 4 connections, 10M chunks, auto-resume
    cmd = [
        "aria2c", 
        "-x", "4", 
        "-s", "4", 
        "--max-connection-per-server=4",
        "--min-split-size=10M",
        "--continue=true",
        "--split=4",
        "--file-allocation=none",
        "-d", os.path.dirname(filename),
        "-o", os.path.basename(filename),
        url
    ]
    
    print(f"⚡ Running stealth download to bypass firewall throttling...")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ aria2c failed ({e}). Falling back to Python...")
        original_urlretrieve(url, filename, reporthook, data)
        
    return (filename, None)

urllib.request.urlretrieve = fast_urlretrieve

print("Fetching metadata and starting download...")
ogbench.download_datasets(['cube-triple-play-v0'])
EOF

python fast_download.py

echo "=========================================="
echo "2. Moving Dataset to Target Directory"
echo "=========================================="
OGBENCH_CACHE="$HOME/.ogbench/data"
TARGET_DIR="$HOME/abdelghani_work/datasets/cube-triple"

mkdir -p "$TARGET_DIR"

if ls "$OGBENCH_CACHE"/cube-triple-play*.npz 1> /dev/null 2>&1; then
    mv "$OGBENCH_CACHE"/cube-triple-play*.npz "$TARGET_DIR/"
    FILE_COUNT=$(ls -1 "$TARGET_DIR"/*.npz | wc -l)
    echo "🎉 Success! $FILE_COUNT dataset shards are now sitting in $TARGET_DIR."
fi
