# delete_episode.py
import pandas as pd
import json
import subprocess
import shutil
import os

LOCAL = "/tmp/advance_new"
OUT = "/tmp/advance_del"
REMOVE_EPISODE = 5

# 清掉舊的 output
if os.path.exists(OUT):
    shutil.rmtree(OUT)
shutil.copytree(LOCAL, OUT)

# =========================================================
# 讀取 data parquet
# =========================================================
data = pd.read_parquet(f"{LOCAL}/data/chunk-000/file-000.parquet")
print(f"Before: {len(data)} frames, episodes: {sorted(data['episode_index'].unique().tolist())}")

orig_data = data.copy()  # 保留原始資料供影片切段用

# 刪掉 episode REMOVE_EPISODE 的 frames
removed_frames = len(data[data['episode_index'] == REMOVE_EPISODE])
data = data[data['episode_index'] != REMOVE_EPISODE].copy()

# 重新 offset episode_index
data.loc[data['episode_index'] > REMOVE_EPISODE, 'episode_index'] -= 1

# 重新 index
data['index'] = range(len(data))

# timestamp 每個 episode 保持從 0 開始（原本就是，不需要改）
data.to_parquet(f"{OUT}/data/chunk-000/file-000.parquet", index=False)
print(f"After: {len(data)} frames, episodes: {sorted(data['episode_index'].unique().tolist())}")

# =========================================================
# 讀取 episodes parquet
# =========================================================
ep = pd.read_parquet(f"{LOCAL}/meta/episodes/chunk-000/file-000.parquet")
ep = ep[ep['episode_index'] != REMOVE_EPISODE].copy()
ep.loc[ep['episode_index'] > REMOVE_EPISODE, 'episode_index'] -= 1
ep = ep.reset_index(drop=True)
ep.to_parquet(f"{OUT}/meta/episodes/chunk-000/file-000.parquet", index=False)
print(f"Episodes parquet: {len(ep)} episodes")

# =========================================================
# Update info.json
# =========================================================
with open(f"{LOCAL}/meta/info.json") as f:
    info = json.load(f)
info['total_episodes'] = len(data['episode_index'].unique())
info['total_frames'] = len(data)
with open(f"{OUT}/meta/info.json", "w") as f:
    json.dump(info, f, indent=2)
print(f"info.json updated: {info['total_episodes']} episodes, {info['total_frames']} frames")

# =========================================================
# 計算每個 episode 在影片裡的全局時間範圍
# =========================================================
fps = info.get("fps", 30)

segments = []
global_time = 0.0
for ep_idx in sorted(orig_data['episode_index'].unique()):
    ep_data = orig_data[orig_data['episode_index'] == ep_idx]
    ep_duration = ep_data['timestamp'].max() + (1.0 / fps)
    if ep_idx != REMOVE_EPISODE:
        segments.append((global_time, global_time + ep_duration))
    global_time += ep_duration

print(f"\nSegments to keep ({len(segments)} episodes):")
for i, (s, e) in enumerate(segments):
    print(f"  [{i}] {s:.4f}s ~ {e:.4f}s")

# =========================================================
# 影片處理：切段 + concat（stream copy，不重新編碼）
# =========================================================
for cam in ["observation.images.wrist", "observation.images.front"]:
    in_vid = f"{LOCAL}/videos/{cam}/chunk-000/file-000.mp4"
    out_vid = f"{OUT}/videos/{cam}/chunk-000/file-000.mp4"

    segment_files = []

    # 切出每個保留的 segment
    for i, (start, end) in enumerate(segments):
        seg_path = f"/tmp/seg_{cam.replace('.','_').replace('/','_')}_{i}.mp4"
        subprocess.run([
            "ffmpeg", "-y",
            "-ss", f"{start:.4f}",
            "-to", f"{end:.4f}",
            "-i", in_vid,
            "-c", "copy",
            seg_path
        ], check=True)
        segment_files.append(seg_path)
        print(f"  Cut segment {i}: {start:.4f}s ~ {end:.4f}s → {seg_path}")

    # 用 concat demuxer 拼接所有 segment
    concat_file = f"/tmp/concat_{cam.replace('.','_').replace('/','_')}.txt"
    with open(concat_file, "w") as f:
        for seg in segment_files:
            f.write(f"file '{seg}'\n")

    subprocess.run([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", concat_file,
        "-c", "copy",
        out_vid
    ], check=True)

    # 清掉暫存 segment 檔
    for seg in segment_files:
        os.remove(seg)

    print(f"✅ {cam} done")

print(f"\nDone! Fixed dataset at {OUT}")
