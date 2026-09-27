"""
Cache and Weight Manager for colab-model-station.
Provides high-speed model prefetching, Hugging Face transfer acceleration,
LoRA asset caching, and storage tier configuration.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

# Enable HF Transfer acceleration by default
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


def get_default_cache_dir() -> Path:
    """Resolve optimal cache directory, prioritizing Google Drive if mounted."""
    drive_cache = Path("/content/drive/MyDrive/model_cache")
    if drive_cache.parent.exists():
        drive_cache.mkdir(parents=True, exist_ok=True)
        print(f"[CACHE] Persistent Google Drive storage detected: {drive_cache}")
        return drive_cache

    local_cache = Path("/content/cache/huggingface")
    local_cache.mkdir(parents=True, exist_ok=True)
    return local_cache


def get_lora_dir() -> Path:
    """Return local directory for storing standalone LoRA weights."""
    lora_dir = Path("/content/models/loras")
    lora_dir.mkdir(parents=True, exist_ok=True)
    return lora_dir


def prefetch_hf_model(repo_id: str, cache_dir: Optional[Path] = None, token: Optional[str] = None) -> Path:
    """Download model repository snapshot using accelerated huggingface_hub."""
    from huggingface_hub import snapshot_download

    target_cache = cache_dir or get_default_cache_dir()
    hf_token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    print(f"[CACHE] Prefetching model snapshot '{repo_id}' into {target_cache}...")
    t_start = time.time()

    local_snapshot = snapshot_download(
        repo_id=repo_id,
        cache_dir=str(target_cache),
        token=hf_token,
        resume_download=True,
    )

    elapsed = time.time() - t_start
    print(f"[CACHE] Prefetch completed in {elapsed:.2f}s. Path: {local_snapshot}")
    return Path(local_snapshot)


def prefetch_lora(lora_source: str, filename: Optional[str] = None, token: Optional[str] = None) -> Path:
    """Download a LoRA file from Hugging Face or direct HTTP URL."""
    target_dir = get_lora_dir()
    hf_token = token or os.environ.get("HF_TOKEN")

    if lora_source.startswith("http://") or lora_source.startswith("https://"):
        import urllib.request
        out_name = filename or Path(urllib.parse.urlparse(lora_source).path).name or "lora.safetensors"
        out_path = target_dir / out_name
        if out_path.is_file():
            print(f"[CACHE] LoRA file already cached at {out_path}")
            return out_path
        print(f"[CACHE] Downloading LoRA from URL: {lora_source} -> {out_path}")
        urllib.request.urlretrieve(lora_source, str(out_path))
        return out_path

    # Assume Hugging Face repo ID
    from huggingface_hub import hf_hub_download
    print(f"[CACHE] Downloading LoRA from HF hub: {lora_source}...")
    file_to_download = filename or "lora.safetensors"
    downloaded = hf_hub_download(
        repo_id=lora_source,
        filename=file_to_download,
        token=hf_token,
        local_dir=str(target_dir),
    )
    return Path(downloaded)


def get_cache_stats() -> dict:
    """Return storage consumption metrics for cache directories."""
    cache_dir = get_default_cache_dir()
    lora_dir = get_lora_dir()

    def get_dir_size(path: Path) -> float:
        total = 0
        if path.exists():
            for p in path.rglob("*"):
                if p.is_file():
                    total += p.stat().st_size
        return round(total / (1024 * 1024 * 1024), 2)

    disk_usage = shutil.disk_usage(str(cache_dir.parent if cache_dir.exists() else "/content"))
    return {
        "cache_path": str(cache_dir),
        "cache_size_gb": get_dir_size(cache_dir),
        "lora_path": str(lora_dir),
        "lora_size_gb": get_dir_size(lora_dir),
        "disk_free_gb": round(disk_usage.free / (1024 * 1024 * 1024), 2),
        "disk_total_gb": round(disk_usage.total / (1024 * 1024 * 1024), 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Colab Model Station Cache Manager")
    subparsers = parser.add_subparsers(dest="command", required=True)

    model_p = subparsers.add_parser("model")
    model_p.add_argument("repo_id", help="Hugging Face model repository ID")
    model_p.add_argument("--token", default=None, help="Hugging Face auth token")

    lora_p = subparsers.add_parser("lora")
    lora_p.add_argument("source", help="LoRA repo ID or URL")
    lora_p.add_argument("--filename", default=None, help="Target weight filename")
    lora_p.add_argument("--token", default=None, help="Hugging Face auth token")

    subparsers.add_parser("stats")

    args = parser.parse_args()
    if args.command == "model":
        prefetch_hf_model(args.repo_id, token=args.token)
    elif args.command == "lora":
        prefetch_lora(args.source, filename=args.filename, token=args.token)
    elif args.command == "stats":
        stats = get_cache_stats()
        print("=== Storage & Cache Statistics ===")
        for k, v in stats.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
