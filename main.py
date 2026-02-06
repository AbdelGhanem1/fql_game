import os
import sys

def check_memory_limits():
    print("-" * 30)
    print("MEMORY DIAGNOSTICS")
    
    # 1. Check the Hard Limit (Cgroups) - This is what kills you
    try:
        # Common path for SLURM cgroups
        with open("/sys/fs/cgroup/memory/memory.limit_in_bytes", "r") as f:
            limit_bytes = int(f.read())
            print(f"JOB MEMORY LIMIT: {limit_bytes / (1024**3):.2f} GB")
    except FileNotFoundError:
        print("JOB MEMORY LIMIT: Could not determine (cgroup file not found)")

    # 2. Check Total Node RAM (Physical hardware)
    try:
        total_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
        print(f"NODE TOTAL RAM:   {total_bytes / (1024**3):.2f} GB")
    except ValueError:
         print("NODE TOTAL RAM:   Could not determine")
         
    print("-" * 30)

check_memory_limits()