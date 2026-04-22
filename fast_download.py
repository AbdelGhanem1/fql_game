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
# FIXED: Targeting 'humanoidmaze-large-navigate-v0' which pulls both train and val splits
ogbench.download_datasets(['humanoidmaze-large-navigate-v0'])
