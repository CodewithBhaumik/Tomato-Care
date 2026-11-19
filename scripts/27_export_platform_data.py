"""Upgrade round, step 8 — everything the web platform shows, in one JSON document, plus the Grad-CAM gallery.

Reads only files written by earlier steps (no new evaluation of models, except Grad-CAM pictures for the gallery):
  experiments/results/final_benchmark.json, selection_*.json, baseline_audit.json, cv summaries, the original
  study's results (dataset statistics, leakage benchmark).
Writes:
  experiments/platform.json            served by the API at /api/analytics
  frontend/public/data/platform.json   bundled snapshot (the site works read-only without the API)
  frontend/public/data/disease_info.json
  frontend/public/gradcam/*.png        Grad-CAM gallery for the held-out sample photos
"""
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.services.predictor import ModelRegistry  # noqa: E402
from tomatoleaf.config import CLASS_KEYS, EXPERIMENTS_DIR, RESULTS_DIR  # noqa: E402
from tomatoleaf.data import load_splits  # noqa: E402
from tomatoleaf.features import leaf_mask  # noqa: E402
from tomatoleaf.gradcam import GradCAM, heatmap_png, overlay  # noqa: E402
from tomatoleaf.models import gradcam_layer, gradcam_reshape  # noqa: E402
from tomatoleaf.preprocessing import load_canonical_array  # noqa: E402
from tomatoleaf.recipes import RECIPES  # noqa: E402
from tomatoleaf.utils import load_json, save_json  # noqa: E402

FRONT = ROOT / "frontend" / "public"
FIVE = ["healthy", "leaf_miner", "magnesium_deficiency", "nitrogen_deficiency", "spotted_wilt_virus"]
TASK_LABEL = {"5class": "Focused · 5 classes", "8class": "Full · 8 classes"}
SHORT_RECIPE = {
    "convnext_t_in1k_base": "ConvNeXt-T 1k · orig.",
    "convnext_t_in1k_robust": "ConvNeXt-T 1k · robust",
    "convnext_t_in22k_base": "ConvNeXt-T 22k · orig.",
    "convnext_t_in22k_robust": "ConvNeXt-T 22k · robust",
    "convnextv2_t_robust": "ConvNeXt-V2-T · robust",
    "dinov2_s_robust": "ViT-S DINOv2 · robust",
    "effnetv2_s_robust": "EffNetV2-S · robust",
}
PERTURBATION_LABEL = {
    "jpeg_q75": "JPEG quality 75", "jpeg_q50": "JPEG quality 50", "blur_sigma1": "Mild blur (σ = 1)",
    "brightness_+15%": "Brightness +15%", "brightness_-15%": "Brightness −15%", "half_resolution": "Half resolution",
    "upload_p3_original_path": "Phone photo (P3) · old upload path", "upload_p3_new_path": "Phone photo (P3) · new upload path",
    "upload_srgb_original_path": "sRGB photo · old upload path", "upload_srgb_new_path": "sRGB photo · new upload path",
}


def png(arr: np.ndarray, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path, optimize=True)
    return "/" + path.relative_to(FRONT).as_posix()


def gradcam_for(predictor, rgb, class_index):
    target = getattr(predictor, "members", [predictor])[0]
    cam = GradCAM(target.model, gradcam_layer(target.model, target.arch), gradcam_reshape(target.model, target.arch))
    try:
        heat, _, _ = cam(target.tensor(rgb), class_index=class_index)
    finally:
        cam.close()
        target.model.zero_grad(set_to_none=True)
    return heat


def build_gallery(defaults, previous, bench_models):
    manifest = load_json(FRONT / "test-samples" / "manifest.json")
    reg = ModelRegistry(EXPERIMENTS_DIR / "deployment.json", device="cuda" if torch.cuda.is_available() else "cpu", lazy=True)
    models = [defaults["5class"], defaults["8class"]]  # the apps serve only the version-2 models
    names = {m["key"]: m["name"] for m in bench_models}
    items, hero = [], None
    for s in manifest["samples"]:
        rgb = load_canonical_array(FRONT / s["file"].lstrip("/"))
        entry = {"sample": s["key"], "true_key": s["key"], "photo_id": s.get("photo_id"),
                 "input": png(rgb, FRONT / "gradcam" / "input" / f"{s['key']}.png"), "overlays": []}
        mask = leaf_mask(rgb)
        for key in models:
            pred = reg.load(key)
            probs, _ = pred.analyse(rgb)
            top = int(np.argmax(probs))
            heat = gradcam_for(pred, rgb, top)
            heat256 = np.asarray(Image.fromarray((heat * 255).astype(np.uint8)).resize((256, 256), Image.BILINEAR)) / 255.0
            pred_key = pred.class_keys[top]
            o = {"model": key, "model_name": names.get(key, key), "pred_key": pred_key, "confidence": float(probs[top]),
                 "correct": pred_key == s["key"], "out_of_scope": s["key"] not in pred.class_keys,
                 "overlay": png(overlay(rgb, heat), FRONT / "gradcam" / key / f"{s['key']}_overlay.png"),
                 "heatmap": png(heatmap_png(heat), FRONT / "gradcam" / key / f"{s['key']}_heat.png"),
                 "on_leaf_share": float((heat256 * mask).sum() / max(heat256.sum(), 1e-9))}
            entry["overlays"].append(o)
            if hero is None and key == defaults["5class"] and s["key"] == "leaf_miner":
                order = np.argsort(probs)[::-1]
                hero = {"input": entry["input"], "heatmap": o["heatmap"], "true_key": s["key"], "pred_key": pred_key,
                        "confidence": float(probs[top]), "model_name": names.get(key, key),
                        "probabilities": [{"key": pred.class_keys[i], "p": float(probs[i])} for i in order]}
        items.append(entry)
        print(f"gradcam {s['key']}: " + ", ".join(f"{o['model']}->{o['pred_key']} {o['confidence']:.2f} leaf {o['on_leaf_share']:.2f}"
                                                  for o in entry["overlays"]))
    shares = {m: float(np.mean([o["on_leaf_share"] for it in items for o in it["overlays"] if o["model"] == m])) for m in models}
    summary = ("Mean share of Grad-CAM mass on the leaf (leaf mask from saturation thresholding): "
               + "; ".join(f"{names.get(m, m)} {100 * v:.0f}%" for m, v in shares.items()) + ".")
    return {"items": items, "models": models, "summary": summary, "on_leaf_share": shares}, hero


def curves_for(task, recipe):
    rows = []
    for k in range(5):
        p = EXPERIMENTS_DIR / "cv" / task / recipe / f"fold{k}.json"
        if p.exists():
            rows.append(load_json(p)["history"])
    if not rows:
        return []
    n = min(len(r) for r in rows)
    out = []
    for i in range(n):
        out.append({"epoch": rows[0][i]["epoch"],
                    **{k: float(np.mean([r[i][k] for r in rows if k in r[i]])) for k in ("train_loss", "val_f1", "val_acc")
                       if all(k in r[i] for r in rows)}})
    return out


def main():
    import sys as _sys
    bench_file = _sys.argv[_sys.argv.index("--benchmark") + 1] if "--benchmark" in _sys.argv else "final_benchmark.json"
    bench = load_json(RESULTS_DIR / bench_file)
    bench_models = load_json(EXPERIMENTS_DIR / "benchmark_models.json")["models"]
    meta = {m["key"]: m for m in bench_models}
    defaults, previous = bench["defaults"], bench["previous_defaults"]
    sel = {t: load_json(RESULTS_DIR / f"selection_{t}.json") for t in ("5class", "8class")}
    old = load_json(EXPERIMENTS_DIR / "analytics.json")
    final8 = load_json(RESULTS_DIR / "final_results.json")
    five = load_json(RESULTS_DIR / "five_class_results.json")
    leak = load_json(RESULTS_DIR / "leakage_official_split.json")
    cvsum = load_json(RESULTS_DIR / "cv_folds_summary.json")

    # ---- dataset
    classes = []
    for c in old["dataset"]["class_distribution"]:
        cv_count = sum(f["per_class"][c["key"]] for f in cvsum["per_fold"])
        classes.append({"key": c["key"], "name": c["name"], "photos": c["photos"], "files": c["images"],
                        "train": c["clean"]["train"], "val": c["clean"]["val"], "test": c["clean"]["test"], "cv": cv_count})
    splits = load_splits()
    dataset = {"name": old["dataset"]["name"], "citation": old["dataset"]["citation"], "doi": old["dataset"]["doi"],
               "files": leak["total_images"], "photos": leak["unique_photos"], "groups": int(splits.group_id.nunique()),
               "classes": classes, "split": old["dataset"]["clean_split"], "official_split": old["dataset"]["official_split"],
               "cv": {"folds": cvsum["n_folds"], "photos": cvsum["photos"]},
               "source": {"phone": "realme GT Master Edition", "resolution": "3468×4624 / 4624×3468",
                          "stored": "256×256 (whole frame squashed)", "encoding": "Display P3, JPEG quality 90, 4:2:0"},
               "leakage": {"test_with_train_sibling": leak["test_images_with_train_sibling"] / leak["test_images"],
                           "val_with_train_sibling": leak["val_images_with_train_sibling"] / leak["val_images"],
                           "families_crossing": leak["families_crossing_splits"],
                           "benchmark": [{"model": b["model"], "official_accuracy": b["official"]["accuracy"],
                                          "clean_accuracy": b["clean"]["accuracy"], "official_f1": b["official"]["f1_macro"],
                                          "clean_f1": b["clean"]["f1_macro"]} for b in final8["leakage_benchmark"]]}}

    # ---- models
    legacy_val = {m["key"]: m["val"] for m in final8["models"]}
    legacy_val.update({m["key"]: m["val"] for m in five["models"]})
    rec_oof = {}
    for t in ("5class", "8class"):
        for c in sel[t]["candidates"]:
            if len(c["members"]) == 1:
                r, v = c["members"][0]
                rec_oof[(t, r, v)] = c["oof"]
    models = []
    for r in bench["models"]:
        m = meta[r["key"]]
        recipes = m.get("recipes")
        cv = None
        if r["key"].startswith("upgrade_"):
            cv = sel[r["task"]]["chosen"]["oof"]
        elif recipes and len(recipes) == 1:
            view = str(m.get("tta_views", 1))
            cv = rec_oof.get((r["task"], recipes[0], view))
        lat = bench["latency"].get(r["key"], {}).get("median_ms")
        temps = sorted({mm.get("temperature") for mm in m.get("members", []) if mm.get("temperature")}) or (
            [m["temperature"]] if m.get("temperature") else [])
        models.append({
            "key": r["key"], "name": r["name"], "task": r["task"], "family": r["family"], "generation": r["generation"],
            "params": m.get("params"), "tta_views": m.get("tta_views") or max([mm.get("tta_views", 1) for mm in m.get("members", [])] or [1]),
            "recipe": recipes[0] if recipes and len(recipes) == 1 else None, "recipes": recipes,
            "deployed": r["key"] in {e["key"] for e in load_json(EXPERIMENTS_DIR / "deployment.json")["models"]},
            "is_default": r["key"] in defaults.values(), "description": m.get("feature_set"),
            "cv": {k: cv[k] for k in ("f1_macro", "accuracy")} if cv else None,
            "val": {k: legacy_val[r["key"]][k] for k in ("f1_macro", "accuracy")} if r["key"] in legacy_val else None,
            "test": r["test"], "per_class": r["per_class"], "confusion": r["confusion"],
            "selective": {"threshold": r["threshold"], "coverage": r["selective"]["coverage"],
                          "accuracy_kept": r["selective"]["accuracy_kept"],
                          "test": {"coverage": r["selective"]["coverage"], "accuracy_kept": r["selective"]["accuracy_kept"]}},
            "calibration": {"temperature": temps[0] if len(temps) == 1 else (float(np.mean(temps)) if temps else None),
                            "ece_after": r["test"]["ece"], "reliability_after": r["reliability"], "fitted_on":
                                "out-of-fold predictions" if r["generation"] == "upgrade" else "the validation split"},
            "latency_ms": lat, "flagged_unusual": r.get("flagged_unusual"), "out_of_scope": r.get("out_of_scope")})
    for t in ("5class", "8class"):  # OOF reliability before/after for the upgrade defaults
        d = next(m for m in models if m["key"] == defaults[t])
        d["calibration"].update({"reliability_before": sel[t]["chosen"]["reliability_oof_raw"],
                                 "reliability_after": sel[t]["chosen"]["reliability_oof"],
                                 "ece_before": sel[t]["chosen"]["oof"]["ece_raw"], "ece_after": sel[t]["chosen"]["oof"]["ece"],
                                 "fitted_on": "out-of-fold predictions (diagram: out-of-fold, before vs after)"})

    # ---- cross-validation recipes
    recipes_out = []
    for t in ("5class", "8class"):
        for rname, info in sel[t]["recipes"].items():
            summ = load_json(EXPERIMENTS_DIR / "cv" / t / rname / "summary.json")
            recipes_out.append({
                "recipe": rname, "label": RECIPES[rname].label, "short": SHORT_RECIPE.get(rname, rname), "task": t,
                "arch": RECIPES[rname].arch, "aug": RECIPES[rname].aug,
                "f1_1": summ["views"]["1"]["f1_macro"], "f1_8": summ["views"]["8"]["f1_macro"],
                "acc_1": summ["views"]["1"]["accuracy"], "acc_8": summ["views"]["8"]["accuracy"],
                "per_fold": [{"fold": f["fold"], "f1_1": f["held_1view"]["f1_macro"], "f1_8": f["held_tta8"]["f1_macro"],
                              "acc_1": f["held_1view"]["accuracy"], "acc_8": f["held_tta8"]["accuracy"]} for f in summ["per_fold"]],
                "train_minutes": summ["train_minutes_total"], "latency_1": info["latency_1"], "latency_8": info["latency_8"],
                "curves": curves_for(t, rname), "chosen": rname in [m["recipe"] for m in sel[t]["chosen"]["members"]]})
    recipes_out.sort(key=lambda r: (r["task"], -r["f1_8"]))
    screening = []  # fold-0 results of EVERY recipe tried (including ones stopped after fold 0)
    for t in ("8class", "5class"):
        for rname, rec in RECIPES.items():
            p = EXPERIMENTS_DIR / "cv" / t / rname / "fold0.json"
            if p.exists():
                f0 = load_json(p)
                screening.append({"task": t, "recipe": rname, "label": rec.label,
                                  "f1_1": f0["held_1view"]["f1_macro"], "f1_8": f0["held_tta8"]["f1_macro"],
                                  "acc_1": f0["held_1view"]["accuracy"], "acc_8": f0["held_tta8"]["accuracy"],
                                  "full_cv": (EXPERIMENTS_DIR / "cv" / t / rname / "summary.json").exists()})

    # ---- audit: preprocessing (validation, original default) and robustness (test, old vs new defaults)
    audit = load_json(RESULTS_DIR / "baseline_audit.json")
    pre = []
    a = audit["models"]["ensemble_5class"]
    for k in ("clean", "upload_p3_original_path", "upload_p3_new_path", "upload_srgb_original_path", "upload_srgb_new_path"):
        if k in a:
            pre.append({"condition": k, "label": "Dataset file (reference)" if k == "clean" else PERTURBATION_LABEL[k],
                        "accuracy": a[k]["accuracy"], "f1_macro": a[k]["f1_macro"], "flip_rate": a[k].get("flip_rate", 0.0)})
    robustness = {}
    for t in ("5class", "8class"):
        rb = bench["robustness"][t]
        names = list(PERTURBATION_LABEL)
        robustness[t] = {"rows": [{"label": PERTURBATION_LABEL[n], **{key: rb[key][n]["flip_rate"] for key in rb}} for n in names
                                  if all(n in rb[key] for key in rb)],
                         "series": [{"key": defaults[t], "label": "Upgrade default", "color": "#15834e"},
                                    {"key": previous[t], "label": "Original default", "color": "#0891b2"}]}

    if "dryrun" in bench_file:  # pipeline check only: never score test photos (the gallery uses held-out samples)
        gallery, hero = {"items": [], "models": [], "summary": "dry run", "on_leaf_share": {}}, None
    else:
        gallery, hero = build_gallery(defaults, previous, bench_models)

    selective = {}
    for t in ("5class", "8class"):
        rc = bench["risk_coverage"][t]
        d = next(m for m in models if m["key"] == defaults[t])
        selective[t] = {"model": rc["model"], "curve": rc["curve"], "rule": sel[t]["selective"]["rule"],
                        "chosen": {"threshold": d["selective"]["threshold"], "coverage": d["selective"]["coverage"],
                                   "accuracy_kept": d["selective"]["accuracy_kept"]}}

    ood = {"method": f"{sel['5class']['ood']['method']} distance in the networks' feature space",
           "keep": sel["5class"]["ood"]["keep"],
           "oof": {t: {k: v for k, v in sel[t]["ood"]["evaluation_oof"][sel[t]["ood"]["method"]].items() if k != "probe_ratios"}
                   for t in ("5class", "8class")},
           "test": {"flagged_in": bench["ood"]["5class"]["flagged_in"], "flagged_near": bench["ood"]["5class"].get("flagged_near"),
                    "flagged_far": bench["ood"]["5class"]["flagged_far_B"], "by_task": bench["ood"]}}

    limitations = [
        "One phone model, two districts of Rajasthan, single leaves on paper: accuracy on field photos, other cameras or regions is unknown.",
        "Potassium Deficiency has 9 photographs in total (2 in the test set) and Nitrogen Deficiency 45: their per-class scores are very uncertain.",
        "The test set is small (290 photos, 265 in the focused scope); differences of a few points are within noise — read the confidence intervals and paired tests.",
        "The same test photos were used to report the original study, so they are not perfectly fresh; no upgrade decision used them.",
        "Temperatures and thresholds were fitted on out-of-fold predictions of the fold models, which are also the deployed models; coverage on new photos can still differ.",
        "The out-of-distribution check was tested on out-of-scope leaf conditions and 30 generic images, not on other crops or field scenes.",
        "Grad-CAM shows where the network looked, not why a leaf is diseased.",
    ]
    env = {"python": platform.python_version(), "torch": torch.__version__,
           "timm": __import__("timm").__version__, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}

    doc = {"version": 2, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "project": {"name": "Tomato Care"}, "dataset": dataset,
           "tasks": {t: {"label": TASK_LABEL[t], "classes": FIVE if t == "5class" else list(CLASS_KEYS),
                         "test_photos": bench["test_photos_5class"] if t == "5class" else bench["test_photos"],
                         "default_model": defaults[t], "previous_default": previous[t]} for t in ("5class", "8class")},
           "models": models, "comparisons": bench["comparisons"],
           "cv": {"selection_rule": sel["8class"]["selection_rule"], "recipes": recipes_out, "screening": screening,
                  "chosen": {t: sel[t]["chosen"]["key"] for t in sel}},
           "audit": {"preprocessing": pre, "robustness": robustness, "dihedral": {k: v.get("dihedral_views") for k, v in audit["models"].items() if v.get("dihedral_views")}},
           "selective": selective, "ood": ood, "gradcam": gallery, "hero": hero, "limitations": limitations,
           "environment": env}
    save_json(doc, EXPERIMENTS_DIR / "platform.json")
    (FRONT / "data").mkdir(parents=True, exist_ok=True)
    save_json(doc, FRONT / "data" / "platform.json")
    shutil.copy(ROOT / "backend" / "data" / "disease_info.json", FRONT / "data" / "disease_info.json")
    size = (EXPERIMENTS_DIR / "platform.json").stat().st_size / 1024
    print(f"wrote platform.json ({size:.0f} KB): {len(models)} models, {len(recipes_out)} recipes")


if __name__ == "__main__":
    main()
