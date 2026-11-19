"""Upgrade round, step 7 — list the weight files of the GitHub release, with size and SHA-256.

Served files: the models in experiments/deployment.json (the version-2 models the apps and the API run).
Reference files: the original study's models that the final benchmark compares against — not served, kept in the
release so the comparison can be reproduced (scripts/download_weights.py --all).

GitHub refuses files over 100 MB inside a repository, so weights are release assets. Asset names are flat
(folders become "__"); experiments/weights_manifest.json maps each asset to its local path and hash, and
scripts/download_weights.py fetches and verifies them.
Usage: python scripts/26_package_release.py [--copy-to DIR]   (DIR receives the renamed asset files)
"""
import argparse
import hashlib
import shutil
from pathlib import Path

from tomatoleaf.config import EXPERIMENTS_DIR, MODELS_DIR
from tomatoleaf.utils import load_json, save_json

RELEASE_TAG = "weights-v2"
REFERENCE = ["ensemble_5class", "convnext_tiny_v2_tta", "convnext_tiny_v2", "efficientnet_b0_v2", "mobilenetv2_ft",
             "rf", "rf_5class"]  # keys in experiments/benchmark_models.json


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def needed_files(deployment: dict) -> list[str]:
    paths = []
    for e in deployment["models"]:
        if e["type"] == "ensemble":
            paths += [m["path"] for m in e["members"]]
        else:
            paths.append(e["path"])
    return sorted(set(paths))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--copy-to", type=Path, default=None)
    parser.add_argument("--repo", default="CodewithBhaumik/Tomato-Care")
    args = parser.parse_args()
    deployment = load_json(EXPERIMENTS_DIR / "deployment.json")
    served = set(needed_files(deployment))
    bench = load_json(EXPERIMENTS_DIR / "benchmark_models.json")
    reference = set(needed_files({"models": [m for m in bench["models"] if m["key"] in REFERENCE]})) - served
    files = []
    for rel in sorted(served | reference):
        path = (MODELS_DIR / rel).resolve()
        asset = rel.replace("../", "").replace("/", "__")
        files.append({"path": rel, "asset": asset, "served": rel in served, "bytes": path.stat().st_size,
                      "sha256": sha256(path),
                      "url": f"https://github.com/{args.repo}/releases/download/{RELEASE_TAG}/{asset}"})
        if args.copy_to:
            args.copy_to.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, args.copy_to / asset)
    manifest = {"release": RELEASE_TAG, "repo": args.repo, "files": files,
                "total_bytes": sum(f["bytes"] for f in files)}
    save_json(manifest, EXPERIMENTS_DIR / "weights_manifest.json")
    served_mb = sum(f["bytes"] for f in files if f["served"]) / 2**20
    print(f"{len(files)} files ({len(served)} served, {served_mb:.0f} MB), {manifest['total_bytes'] / 2**20:.0f} MB "
          "-> experiments/weights_manifest.json")


if __name__ == "__main__":
    main()
