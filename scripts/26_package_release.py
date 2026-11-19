"""Upgrade round, step 7 — list every weight file the API needs, with size and SHA-256, for the GitHub release.

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
    files = []
    for rel in needed_files(deployment):
        path = (MODELS_DIR / rel).resolve()
        asset = rel.replace("../", "").replace("/", "__")
        files.append({"path": rel, "asset": asset, "bytes": path.stat().st_size, "sha256": sha256(path),
                      "url": f"https://github.com/{args.repo}/releases/download/{RELEASE_TAG}/{asset}"})
        if args.copy_to:
            args.copy_to.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, args.copy_to / asset)
    manifest = {"release": RELEASE_TAG, "repo": args.repo, "files": files,
                "total_bytes": sum(f["bytes"] for f in files)}
    save_json(manifest, EXPERIMENTS_DIR / "weights_manifest.json")
    print(f"{len(files)} files, {manifest['total_bytes'] / 2**20:.0f} MB -> experiments/weights_manifest.json")


if __name__ == "__main__":
    main()
