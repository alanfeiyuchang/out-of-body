"""从 ARKitScenes 的清单里挑扫描：每个住宅（visit）只要一个，彩色图压缩包在 250~900 MB 之间。
用法: python pick_arkit.py <splits.csv> <训练几个> <验证几个> > list.txt     只读文件头，不下载。"""
import csv
import random
import subprocess
import sys

BASE = "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw"
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r["visit_id"] not in ("NA", "")]
need = {"Training": int(sys.argv[2]), "Validation": int(sys.argv[3])}
random.seed(7)
random.shuffle(rows)
visits, pick, total = set(), {k: [] for k in need}, 0
for r in rows:
    fold, vid = r["fold"], r["video_id"]
    if fold not in need or len(pick[fold]) >= need[fold] or r["visit_id"] in visits:
        continue
    url = f"{BASE}/{fold}/{vid}/vga_wide.zip"
    head = subprocess.run(["curl", "-sI", "-m", "15", url], capture_output=True, text=True).stdout
    size = [int(line.split()[1]) for line in head.lower().splitlines() if line.startswith("content-length")]
    if not size or not 250e6 < size[0] < 900e6:
        continue
    visits.add(r["visit_id"])
    pick[fold].append(vid)
    total += size[0]
    if all(len(pick[k]) >= need[k] for k in need):
        break
for fold in pick:
    for vid in pick[fold]:
        print(fold, vid)
print(f"彩色图合计 {total / 1e9:.1f} GB", file=sys.stderr)
