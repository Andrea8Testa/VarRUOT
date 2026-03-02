import subprocess
import os
import sys

SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

for seed in SEEDS:
    cmd = [
        sys.executable, "main.py",
        "--name", str(seed),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

print("All runs finished.")