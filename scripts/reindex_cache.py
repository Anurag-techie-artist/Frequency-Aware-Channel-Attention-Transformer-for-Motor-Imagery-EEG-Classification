"""
Reindex outputs/cache directory and update metadata.json with all valid cached trial files.
"""

import os
import sys
import json
import time
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.cache import CacheManager, CACHE_VERSION, compute_config_hash
from configs.config_loader import load_master_config
from datasets.pipeline import EEGPreprocessingPipeline

def reindex_cache():
    cache_dir = os.path.join(PROJECT_ROOT, "outputs", "cache")
    metadata_path = os.path.join(cache_dir, "metadata.json")
    config = load_master_config(project_root=PROJECT_ROOT)
    pipeline = EEGPreprocessingPipeline(config=config)
    config_hash = compute_config_hash(pipeline.config, "frequency")

    if not os.path.exists(cache_dir):
        print("Cache directory missing.")
        return

    pt_files = [f for f in os.listdir(cache_dir) if f.endswith("_trials.pt")]
    print(f"Found {len(pt_files)} trial cache files in {cache_dir}.")

    existing_files_meta = {}
    win_size = 250
    stride = 50

    for fname in sorted(pt_files):
        fpath = os.path.join(cache_dir, fname)
        try:
            data = torch.load(fpath, map_location="cpu", weights_only=False)
            edf_path = data.get("edf_path")
            edf_mtime = data.get("edf_mtime", 0)
            trials = data.get("trials")
            labels = data.get("labels")

            if trials is None or edf_path is None:
                print(f"Skipping incomplete file: {fname}")
                continue

            n_trials = len(trials)
            trial_length_samples = trials.shape[-1]
            trial_start_indices = [0]
            cumulative_windows = 0

            for trial_idx in range(n_trials):
                num_windows_in_trial = len(range(0, trial_length_samples - win_size + 1, stride))
                cumulative_windows += num_windows_in_trial
                trial_start_indices.append(cumulative_windows)

            existing_files_meta[edf_path] = {
                "edf_path": edf_path,
                "cache": fname,
                "num_trials": n_trials,
                "trial_length_samples": trial_length_samples,
                "trial_start_indices": trial_start_indices,
                "total_windows": cumulative_windows,
                "edf_mtime": edf_mtime,
                "config_hash": config_hash,
                "tensor_shape": list(trials.shape),
            }
            print(f"Indexed: {fname} -> {edf_path} ({cumulative_windows} windows)")
        except Exception as e:
            print(f"Error reading {fname}: {e}")

    # Build master metadata
    ordered_files_meta = []
    global_offset = 0

    for abs_path, meta_entry in existing_files_meta.items():
        n_windows = meta_entry["total_windows"]
        meta_entry["start_index"] = global_offset
        meta_entry["end_index"] = global_offset + n_windows - 1 if n_windows > 0 else global_offset
        global_offset += n_windows
        ordered_files_meta.append(meta_entry)

    master_metadata = {
        "cache_version": CACHE_VERSION,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_root": os.path.dirname(os.path.dirname(list(existing_files_meta.keys())[0])) if existing_files_meta else "",
        "representation": "frequency",
        "config_hash": config_hash,
        "window_size": win_size,
        "stride": stride,
        "total_samples": global_offset,
        "files": ordered_files_meta,
    }

    CacheManager.atomic_json_save(master_metadata, metadata_path)
    print(f"\nSuccessfully saved unified metadata index with {len(ordered_files_meta)} files ({global_offset} total windows) to {metadata_path}.")

if __name__ == "__main__":
    reindex_cache()
