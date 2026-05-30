# verify_merge.py
import pandas as pd
import json
import subprocess

OUT = "/tmp/merged_final2"

print("=" * 50)
print("1. 檢查 info.json")
print("=" * 50)
with open(f"{OUT}/meta/info.json") as f:
    info = json.load(f)
print(f"total_episodes: {info['total_episodes']}")
print(f"total_frames: {info['total_frames']}")

print("\n" + "=" * 50)
print("2. 檢查 data parquet")
print("=" * 50)
data = pd.read_parquet(f"{OUT}/data/chunk-000/file-000.parquet")
print(f"Total rows: {len(data)}")
print(f"Columns: {list(data.columns)}")
print(f"Episode indices: {sorted(data['episode_index'].unique().tolist())}")
print(f"Index range: {data['index'].min()} ~ {data['index'].max()}")

# index 連續不重複
assert len(data['index'].unique()) == len(data), "ERROR: index 有重複！"
assert data['index'].max() == len(data) - 1, "ERROR: index 不連續！"
print("✅ index 連續且不重複")

# episode_index 正確
expected_episodes = list(range(info['total_episodes']))
actual_episodes = sorted(data['episode_index'].unique().tolist())
assert actual_episodes == expected_episodes, \
    f"ERROR: episode_index 不對！expected {expected_episodes}, got {actual_episodes}"
print("✅ episode_index 正確")

# info.json 數字符合 parquet
assert len(data) == info['total_frames'], \
    f"ERROR: total_frames 不符！parquet={len(data)}, info.json={info['total_frames']}"
print("✅ total_frames 符合")

assert len(actual_episodes) == info['total_episodes'], \
    f"ERROR: total_episodes 不符！parquet={len(actual_episodes)}, info.json={info['total_episodes']}"
print("✅ total_episodes 符合")

# 每個 episode 的 timestamp 從 0 開始且遞增
print("\n每個 episode 的 timestamp 狀況：")
for ep_idx in sorted(data['episode_index'].unique()):
    ep_data = data[data['episode_index'] == ep_idx].sort_values('index')
    timestamps = ep_data['timestamp'].values
    min_ts = timestamps[0]
    max_ts = timestamps[-1]
    n_frames = len(ep_data)

    assert min_ts < 0.1, \
        f"ERROR: episode {ep_idx} timestamp 不是從 0 開始！min={min_ts:.4f}"
    assert all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1)), \
        f"ERROR: episode {ep_idx} timestamp 不遞增！"

    print(f"  Episode {ep_idx}: {min_ts:.3f} ~ {max_ts:.3f}, frames={n_frames}")

print("✅ 每個 episode 的 timestamp 從 0 開始且遞增")

print("\n" + "=" * 50)
print("3. 檢查 episodes parquet")
print("=" * 50)
ep = pd.read_parquet(f"{OUT}/meta/episodes/chunk-000/file-000.parquet")
print(f"Total episodes: {len(ep)}")
print(f"Columns: {list(ep.columns)}")
assert len(ep) == info['total_episodes'], \
    f"ERROR: episodes parquet 筆數不符！parquet={len(ep)}, info.json={info['total_episodes']}"
print("✅ episodes parquet 筆數正確")

print("\n" + "=" * 50)
print("4. 檢查影片")
print("=" * 50)
def get_video_info(path):
    result = subprocess.run([
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", path
    ], capture_output=True, text=True)
    return json.loads(result.stdout)

for cam in ["observation.images.wrist", "observation.images.front"]:
    vid_path = f"{OUT}/videos/{cam}/chunk-000/file-000.mp4"
    vid_info = get_video_info(vid_path)
    duration = float(vid_info["format"]["duration"])
    print(f"{cam}:")
    print(f"  duration: {duration:.4f}s")

    max_ts = data['timestamp'].max()
    assert max_ts <= duration + 1.0, \
        f"ERROR: timestamp 超出影片長度！max_ts={max_ts:.4f}, duration={duration:.4f}"
    print(f"  ✅ timestamp 最大值 ({max_ts:.4f}s) 在影片範圍內")

print("\n" + "=" * 50)
print("✅ 全部檢查通過！可以安全 push")
print("=" * 50)
