"""Exports one held-out TEST photograph per class for the web app's "try an unseen test photo" gallery.

The photo is chosen deterministically (first test photo of each class in splits.csv order — the same photos as in
the report's Grad-CAM figure), NOT by whether the model classifies it correctly.
Output: frontend/public/test-samples/<key>.jpg and manifest.json (with the true label).
"""
import shutil

from tomatoleaf.config import CLASSES, DATASET_DIR, ROOT
from tomatoleaf.data import load_splits
from tomatoleaf.utils import save_json


def main():
    splits = load_splits()
    test = splits[splits.split == "test"]
    out = ROOT / "frontend" / "public" / "test-samples"
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for c in CLASSES:
        row = test[test.label == c["index"]].iloc[0]
        # byte-identical copy: re-encoding would change pixels and could flip borderline predictions
        shutil.copyfile(DATASET_DIR / row.path, out / f"{c['key']}.jpg")
        manifest.append({"key": c["key"], "name": c["name"], "file": f"/test-samples/{c['key']}.jpg",
                         "photo_id": row.photo_id})
    save_json({"note": "One held-out test photograph per class (never used for training or model selection). "
                       "Chosen as the first test photo of each class, not filtered by model correctness.",
               "samples": manifest}, out / "manifest.json")
    print(f"Exported {len(manifest)} test samples to {out}")


if __name__ == "__main__":
    main()
