import urllib.request
import ogbench
import subprocess
import os

# Set the persistent path for RunPod
DATASET_ROOT = "/workspace/datasets"
os.environ["OGBENCH_DATASET_DIR"] = DATASET_ROOT

original_urlretrieve = urllib.request.urlretrieve

def fast_urlretrieve(url, filename, reporthook=None, data=None):
    print(f"\n🚀 Intercepted URL: {url}")
    print(f"📦 Target file: {filename}")
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # aria2c settings for maximum speed on cloud backbone
    cmd = [
        "aria2c", 
        "-x", "16",           # Increased connections for cloud network
        "-s", "16", 
        "--max-connection-per-server=16",
        "--min-split-size=5M",
        "--continue=true",
        "--split=16",
        "--file-allocation=none",
        "-d", os.path.dirname(filename),
        "-o", os.path.basename(filename),
        url
    ]
    
    print(f"⚡ Running accelerated cloud download...")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ aria2c failed ({e}). Falling back to Python...")
        original_urlretrieve(url, filename, reporthook, data)
        
    return (filename, None)

urllib.request.urlretrieve = fast_urlretrieve

print("Starting Antmaze-Giant Download...")
# This pulls the default navigate dataset used in the QAM paper[cite: 1]
ogbench.download_datasets(['antmaze-giant-navigate-v0'])
