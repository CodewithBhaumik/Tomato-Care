"""Exports one held-out TEST photograph per class for the web apps' example gallery.

Rules (both deterministic and NOT based on any model's answer):
  --rule typical (default since 2026-11-19): the most typical test photo of each class — the medoid of the class's
      test photos in a GENERIC image-feature space (ConvNeXt-Tiny with ImageNet weights only, never trained on this
      dataset): the photo with the highest mean cosine similarity to the other test photos of its class. The rule was
      fixed before the final models were scored; it uses images and labels only, no prediction.
  --rule first: the first test photo of each class in splits.csv order (the gallery up to v2.1).
Files are byte-identical copies of the dataset photos (re-encoding could change pixels and flip borderline answers).
Output: frontend/public/test-samples/<key>.jpg and manifest.json (with the true label and the rule).
"""
import argparse
import shutil

import numpy as np
import torch

from tomatoleaf.config import CLASSES, DATASET_DIR, ROOT
from tomatoleaf.data import load_images, load_splits
from tomatoleaf.utils import save_json

NOTES = {
    "typical": ("One held-out test photograph per class (never used for training or model selection): the most typical "
                "test photo of each class, i.e. the medoid in a generic ImageNet feature space (ConvNeXt-Tiny, not trained "
                "on this dataset). Chosen from images and labels only, without looking at any model's answer."),
    "first": ("One held-out test photograph per class (never used for training or model selection). Chosen as the first "
              "test photo of each class, not filtered by model correctness."),
}


@torch.no_grad()
def generic_features(images: np.ndarray) -> np.ndarray:
    """L2-normalised 768-d ImageNet ConvNeXt-Tiny features (global average pool), CPU, float32."""
    from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

    from tomatoleaf.preprocessing import preprocess_for_cnn

    net = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1).eval()
    feats = []
    for rgb in images:
        x = preprocess_for_cnn(rgb, 224)
        f = net.classifier[:2](net.avgpool(net.features(x)))  # LayerNorm2d + Flatten: the penultimate features
        feats.append(torch.nn.functional.normalize(f, dim=1)[0].numpy())
    return np.stack(feats)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rule", choices=list(NOTES), default="typical")
    args = parser.parse_args()
    splits = load_splits()
    test = splits[splits.split == "test"].sort_values("photo_id").reset_index(drop=True)
    out = ROOT / "frontend" / "public" / "test-samples"
    out.mkdir(parents=True, exist_ok=True)
    feats = generic_features(load_images(test.path)) if args.rule == "typical" else None
    manifest = []
    for c in CLASSES:
        rows = test[test.label == c["index"]]
        if args.rule == "typical":
            f = feats[rows.index.to_numpy()]
            sim = f @ f.T
            mean_other = (sim.sum(1) - 1.0) / max(len(rows) - 1, 1)
            row = rows.iloc[int(np.argmax(mean_other))]  # ties: first by photo_id
            extra = {"typicality": float(mean_other.max()), "class_test_photos": int(len(rows))}
        else:
            row = splits[(splits.split == "test") & (splits.label == c["index"])].iloc[0]
            extra = {}
        shutil.copyfile(DATASET_DIR / row.path, out / f"{c['key']}.jpg")
        manifest.append({"key": c["key"], "name": c["name"], "file": f"/test-samples/{c['key']}.jpg",
                         "photo_id": row.photo_id, **extra})
    save_json({"note": NOTES[args.rule], "rule": args.rule, "samples": manifest}, out / "manifest.json")
    print(f"Exported {len(manifest)} test samples ({args.rule}) to {out}: "
          + ", ".join(f"{m['key']}={m['photo_id']}" for m in manifest))


if __name__ == "__main__":
    main()
