# verify_delete.py
import pandas as pd
import json
import subprocess

ORIGINAL = "/tmp/advance_new"
FIXED = "/tmp/advance_del"
REMOVED_EPISODE = 5

print("=" * 50)
print("1. 比較 info.json")
print("=" * 50)
with open(f"{ORIGINAL}/meta/info.json") as f:
    orig_info = json.load(f)
with open(f"{FIXED}/meta/info.json") as f:
    fixed_info = json.load(f)

print(f"Original: {orig_info['total_episodes']} episodes, {orig_info['total_frames']} frames")
print(f"Fixed:    {fixed_info['total_episodes']} episodes, {fixed_info['total_frames']} frames")
assert fixed_info['total_episodes'] == orig_info['total_episodes'] - 1, "ERROR: episode 數不對！"
assert fixed_info['total_frames'] < orig_info['total_frames'], "ERROR: frame 數沒有減少！"
print("✅ info.json 正確")

print("\n" + "=" * 50)
print("2. 檢查 data parquet")
print("=" * 50)
orig_data = pd.read_parquet(f"{ORIGINAL}/data/chunk-000/file-000.parquet")
fixed_data = pd.read_parquet(f"{FIXED}/data/chunk-000/file-000.parquet")

print(f"Original frames: {len(orig_data)}")
print(f"Fixed frames:    {len(fixed_data)}")

removed_frames = len(orig_data[orig_data['episode_index'] == REMOVED_EPISODE])
print(f"Removed frames (episode {REMOVED_EPISODE}): {removed_frames}")
assert len(fixed_data) == len(orig_data) - removed_frames, "ERROR: frame 數不符！"
print("✅ frame 數正確")

# 確認原本 episode 8 的資料不存在（用後續 episode 的 frame 數來驗證）
orig_ep8_frames = len(orig_data[orig_data['episode_index'] == REMOVED_EPISODE])
orig_ep9_frames = len(orig_data[orig_data['episode_index'] == REMOVED_EPISODE + 1])
new_ep8_frames = len(fixed_data[fixed_data['episode_index'] == REMOVED_EPISODE])
assert new_ep8_frames == orig_ep9_frames, \
    f"ERROR: 新的 episode 8 frame 數不對！expected {orig_ep9_frames}, got {new_ep8_frames}"
print(f"✅ episode {REMOVED_EPISODE} 已被正確移除，原 episode 9 → 新 episode 8")

# episode_index 連續
expected_episodes = list(range(orig_info['total_episodes'] - 1))
actual_episodes = sorted(fixed_data['episode_index'].unique().tolist())
assert actual_episodes == expected_episodes, \
    f"ERROR: episode_index 不連續！expected {expected_episodes}, got {actual_episodes}"
print(f"✅ episode_index 連續: {actual_episodes}")

# index 連續不重複
assert len(fixed_data['index'].unique()) == len(fixed_data), "ERROR: index 有重複！"
assert fixed_data['index'].max() == len(fixed_data) - 1, "ERROR: index 不連續！"
print("✅ index 連續且不重複")

print("\n每個 episode 的 timestamp：")
for ep_idx in sorted(fixed_data['episode_index'].unique()):
    ep_data = fixed_data[fixed_data['episode_index'] == ep_idx]
    min_ts = ep_data['timestamp'].min()
    max_ts = ep_data['timestamp'].max()
    assert min_ts < 0.1, f"ERROR: episode {ep_idx} timestamp 不從 0 開始！"
    print(f"  Episode {ep_idx}: {min_ts:.3f} ~ {max_ts:.3f}, frames={len(ep_data)}")
print("✅ 每個 episode timestamp 從 0 開始")

print("\n" + "=" * 50)
print("3. 檢查 episodes parquet")
print("=" * 50)
fixed_ep = pd.read_parquet(f"{FIXED}/meta/episodes/chunk-000/file-000.parquet")
print(f"Episodes: {len(fixed_ep)}")
assert len(fixed_ep) == fixed_info['total_episodes'], "ERROR: episodes parquet 筆數不符！"
print("✅ episodes parquet 筆數正確")

print("\n" + "=" * 50)
print("4. 檢查影片長度")
print("=" * 50)
def get_duration(path):
    result = subprocess.run([
        "ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", path
    ], capture_output=True, text=True)
    return float(json.loads(result.stdout)["format"]["duration"])

for cam in ["observation.images.wrist", "observation.images.front"]:
    orig_dur = get_duration(f"{ORIGINAL}/videos/{cam}/chunk-000/file-000.mp4")
    fixed_dur = get_duration(f"{FIXED}/videos/{cam}/chunk-000/file-000.mp4")
    print(f"{cam}:")
    print(f"  Original: {orig_dur:.4f}s")
    print(f"  Fixed:    {fixed_dur:.4f}s")
    assert fixed_dur < orig_dur, "ERROR: 影片沒有變短！"
    print(f"  ✅ 影片已縮短 {orig_dur - fixed_dur:.4f}s")

    max_ts = fixed_data['timestamp'].max()
    assert max_ts <= fixed_dur + 1.0, \
        f"ERROR: timestamp 超出影片長度！max_ts={max_ts:.4f}, duration={fixed_dur:.4f}"
    print(f"  ✅ timestamp 最大值 ({max_ts:.4f}s) 在影片範圍內")

print("\n" + "=" * 50)
print("✅ 全部檢查通過！可以安全 push")
print("=" * 50)
