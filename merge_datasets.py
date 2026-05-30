# merge_datasets.py
import json, shutil, os
import pandas as pd
import subprocess

BASE = "/tmp/baki1"    
EXTRA = "/tmp/advance_new"
OUT = "/tmp/merged_final2"

# 清掉舊的 output
if os.path.exists(OUT):
    shutil.rmtree(OUT)
shutil.copytree(BASE, OUT)

# =========================================================
# 讀取 info
# =========================================================
with open(f"{BASE}/meta/info.json") as f:
    base_info = json.load(f)
base_episodes = base_info["total_episodes"]
base_frames = base_info["total_frames"]
print(f"Base: {base_episodes} episodes, {base_frames} frames")

with open(f"{EXTRA}/meta/info.json") as f:
    extra_info = json.load(f)
extra_episodes = extra_info["total_episodes"]
extra_frames = extra_info["total_frames"]
print(f"Extra: {extra_episodes} episodes, {extra_frames} frames")

# =========================================================
# Merge data parquet
# =========================================================
base_data = pd.read_parquet(f"{BASE}/data/chunk-000/file-000.parquet")
extra_data = pd.read_parquet(f"{EXTRA}/data/chunk-000/file-000.parquet")

# episode_index offset
extra_data["episode_index"] = extra_data["episode_index"] + base_episodes

# timestamp: 每個 episode 從 0 開始，不需要 offset
# 確保 extra 每個 episode 的 timestamp 從 0 開始
for ep_idx in extra_data["episode_index"].unique():
    mask = extra_data["episode_index"] == ep_idx
    min_ts = extra_data.loc[mask, "timestamp"].min()
    extra_data.loc[mask, "timestamp"] = extra_data.loc[mask, "timestamp"] - min_ts

# index offset
extra_data["index"] = extra_data["index"] + base_frames

merged_data = pd.concat([base_data, extra_data], ignore_index=True)
merged_data["index"] = range(len(merged_data))
merged_data.to_parquet(f"{OUT}/data/chunk-000/file-000.parquet", index=False)
print(f"Merged data: {len(merged_data)} frames")

# =========================================================
# Merge episodes parquet
# =========================================================
base_ep = pd.read_parquet(f"{BASE}/meta/episodes/chunk-000/file-000.parquet")
extra_ep = pd.read_parquet(f"{EXTRA}/meta/episodes/chunk-000/file-000.parquet")

extra_ep["episode_index"] = extra_ep["episode_index"] + base_episodes

merged_ep = pd.concat([base_ep, extra_ep], ignore_index=True)
merged_ep.to_parquet(f"{OUT}/meta/episodes/chunk-000/file-000.parquet", index=False)
print(f"Merged episodes: {len(merged_ep)}")

# =========================================================
# Merge videos（ffmpeg concat）
# =========================================================
for cam in ["observation.images.wrist", "observation.images.front"]:
    base_vid = f"{BASE}/videos/{cam}/chunk-000/file-000.mp4"
    extra_vid = f"{EXTRA}/videos/{cam}/chunk-000/file-000.mp4"
    out_vid = f"{OUT}/videos/{cam}/chunk-000/file-000.mp4"

    concat_file = f"/tmp/concat_{cam.replace('/', '_').replace('.', '_')}.txt"
    with open(concat_file, "w") as f:
        f.write(f"file '{base_vid}'\n")
        f.write(f"file '{extra_vid}'\n")

    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", concat_file,
        "-c", "copy",
        out_vid
    ], check=True)
    print(f"Merged video: {cam}")

# =========================================================
# Update info.json
# =========================================================
merged_info = base_info.copy()
merged_info["total_episodes"] = base_episodes + extra_episodes
merged_info["total_frames"] = base_frames + extra_frames

# 如果 extra 的 info.json 說 0 episodes，手動從 parquet 算
actual_episodes = len(merged_ep)
actual_frames = len(merged_data)
merged_info["total_episodes"] = actual_episodes
merged_info["total_frames"] = actual_frames

with open(f"{OUT}/meta/info.json", "w") as f:
    json.dump(merged_info, f, indent=2)
print(f"Updated info.json: {actual_episodes} episodes, {actual_frames} frames")
print(f"Done! Merged at {OUT}")
