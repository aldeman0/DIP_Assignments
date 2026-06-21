import os

log_dir = '/root/autodl-tmp/data/lego/output/official_rgb'

# 1. PLY header to get vertex count
ply = os.path.join(log_dir, 'point_cloud', 'iteration_7000', 'point_cloud.ply')
with open(ply, 'rb') as f:
    for line in f:
        line = line.decode('ascii', errors='ignore').strip()
        if 'element vertex' in line:
            n_verts = int(line.split()[-1])
            print(f'Gaussian count: {n_verts}')
            break

# 2. Training duration from single event file timestamp
events = [f for f in os.listdir(log_dir) if 'events' in f]
for e in events:
    parts = e.split('.')
    for p in parts:
        if p.isdigit() and len(p) >= 10:
            import datetime
            dt = datetime.datetime.fromtimestamp(float(p))
            print(f'Timestamp: {dt}')
            break

# 3. GPU memory
import torch
print(f'Peak GPU memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB')

# 4. Check if there's training output we captured
cfg = os.path.join(log_dir, 'cfg_args')
if os.path.exists(cfg):
    with open(cfg) as f:
        print(f'Config: {f.read().strip()}')
