"""Shared state of the Streamlit app: model weights, the model registry, cached predictions and the published results.

Inference uses exactly the code the REST API uses (backend.services.predictor.ModelRegistry) on the same canonical
256x256 image (tomatoleaf.preprocessing.prepare_upload), so the app, the API and the offline benchmark agree.
Defaults suit a small server such as Streamlit Community Cloud (about 2.7 GB RAM, 2 CPUs):
  TOMATO_LOWMEM=1   the fold models of an ensemble share one float32 network; their float16 weights stay
                    memory-mapped on disk (identical predictions, ~1 GB less RAM)
  TOMATO_THREADS    CPU threads (default: the CPUs this container may use)
  TOMATO_DOWNLOAD_WEIGHTS=0  never download weights (tests, offline use)
"""
import base64
import io
import json
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
os.environ.setdefault("TOMATO_LOWMEM", "1")
os.environ.setdefault("TOMATO_RUNTIME", "torch")

import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402
from PIL import Image  # noqa: E402

from tomatoleaf import weights  # noqa: E402
from tomatoleaf.config import CLASS_KEYS, CLASSES  # noqa: E402

EXPERIMENTS = ROOT / "experiments"
PUBLIC = ROOT / "frontend" / "public"
BRAND = ROOT / "assets" / "brand"
REPO_URL = "https://github.com/CodewithBhaumik/Tomato-Care"
DOI_URL = "https://doi.org/10.1007/s00530-023-01158-y"
CLASS_NAME = {c["key"]: c["name"] for c in CLASSES}
TASK_LABEL = {"5class": "Focused · 5 conditions", "8class": "Full · 8 conditions"}
TASK_HELP = {
    "5class": "Healthy, Leaf Miner, Magnesium and Nitrogen Deficiency, Spotted Wilt Virus — the conditions the "
              "models recognise most reliably. Recommended.",
    "8class": "All eight classes of the dataset, including Early Blight, Late Blight and Potassium Deficiency, "
              "which are much harder (few training photos).",
}
UPLOAD_TYPES = ["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"]


# ---------------------------------------------------------------------------------------------------------- setup
def cpu_threads() -> int:
    """CPUs this process may use: TOMATO_THREADS, else the container's CPU quota (cgroup v2) or affinity, max 8."""
    if os.getenv("TOMATO_THREADS"):
        return max(1, int(os.environ["TOMATO_THREADS"]))
    n = os.cpu_count() or 2
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
        if quota != "max":
            n = min(n, max(1, round(int(quota) / int(period))))
    except (OSError, ValueError):
        pass
    if hasattr(os, "sched_getaffinity"):
        n = min(n, len(os.sched_getaffinity(0)))
    return max(1, min(n, 8))


@st.cache_data(show_spinner=False)
def deployment() -> dict:
    return json.loads((EXPERIMENTS / "deployment.json").read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def platform() -> dict:
    """Every published result (written by scripts/27_export_platform_data.py from the final benchmark)."""
    return json.loads((EXPERIMENTS / "platform.json").read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def disease_info() -> dict:
    doc = json.loads((ROOT / "backend" / "data" / "disease_info.json").read_text(encoding="utf-8"))
    return {c["key"]: c for c in doc["classes"]} | {"_disclaimer": doc.get("disclaimer", "")}


def missing_weights(keys=None) -> list[dict]:
    try:
        return [f for f in weights.files_for(keys) if not weights.is_present(f)]
    except FileNotFoundError:  # no manifest: running from a training checkout, weights are local
        return []


_DOWNLOAD_LOCK = threading.Lock()  # one download at a time per server process, whatever the number of visitors
_READY = threading.Event()


def ensure_weights(keys=None) -> None:
    """Download every missing weight file (default: of all deployed models) with a progress bar. Called once per
    page run outside any cached function; after the first success it returns immediately. Files are verified
    with SHA-256 and kept on disk for the life of the server."""
    if os.getenv("TOMATO_DOWNLOAD_WEIGHTS", "1") == "0":  # tests / offline use: never touch the network
        return
    if _READY.is_set() or not missing_weights(keys):
        if keys is None:
            _READY.set()
        return
    with st.spinner("Preparing the models (first start only)…"), _DOWNLOAD_LOCK:
        todo = missing_weights(keys)  # another visitor may have fetched them while we waited
        total = sum(f["bytes"] for f in todo)
        bar = st.progress(0.0, text="Downloading model weights…") if todo else None
        done = 0
        for f in todo:
            base = done

            def progress(got, _size, base=base):
                bar.progress(min(1.0, (base + got) / total),
                             text=f"Downloading model weights (first start only) · {(base + got) / 2**20:,.0f} of "
                                  f"{total / 2**20:,.0f} MB")

            weights.fetch(f, on_progress=progress)
            done += f["bytes"]
        if bar is not None:
            bar.empty()
    if keys is None:
        _READY.set()


@st.cache_resource(show_spinner=False)
def registry():
    """The model registry, shared by every visitor. Only the default model loads now; others on first use."""
    import torch

    from backend.services.predictor import ModelRegistry

    torch.set_num_threads(cpu_threads())
    return ModelRegistry(EXPERIMENTS / "deployment.json", device="cpu", lazy=True)


def model_entry(key: str) -> dict:
    return next(e for e in deployment()["models"] if e["key"] == key)


def models_for(task: str) -> list[dict]:
    """Deployed models of one task: the upgrade default first, then the original study's models."""
    entries = [e for e in deployment()["models"] if e.get("task") == task]
    default = deployment()["defaults_by_task"][task]
    return sorted(entries, key=lambda e: (e["key"] != default, e.get("generation") != "upgrade", e["name"]))


def short_name(entry: dict) -> str:
    """Display name: the version-2 fold ensembles by size, the original study's models by architecture."""
    if entry.get("generation") == "upgrade":
        n = len(entry.get("members") or [1])
        return f"Tomato Care v2 ({n} networks)"
    return entry["name"].split(" (original")[0] + " · v1"


def model_label(entry: dict) -> str:
    star = "★ " if entry["key"] in deployment()["defaults_by_task"].values() else ""
    acc = (entry.get("test_metrics") or {}).get("accuracy")
    return f"{star}{short_name(entry)}" + (f" · {100 * acc:.1f}%" if acc is not None else "")


# ---------------------------------------------------------------------------------------------------- inference
def canonical(image_bytes: bytes):
    """(256x256 RGB array as the models see it, details of the conversion)."""
    from tomatoleaf.preprocessing import prepare_upload

    return prepare_upload(io.BytesIO(image_bytes))


def data_url_to_rgba(url: str | None) -> np.ndarray | None:
    if not url:
        return None
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGBA"))


@st.cache_data(max_entries=64, show_spinner=False)
def predict(image_bytes: bytes, model_key: str, explain: bool = True) -> dict:
    """One diagnosis (the API's /api/predict response) plus the canonical input and the Grad-CAM heat map."""
    rgb, info = canonical(image_bytes)
    t0 = time.perf_counter()
    out = registry().predict(rgb, model_key, explain=explain)
    out["total_ms"] = 1000 * (time.perf_counter() - t0)
    out["input"] = info
    out["canonical"] = rgb
    exp = out.get("explanation") or {}
    out["heatmap_rgba"] = data_url_to_rgba(exp.get("heatmap"))
    return out


@st.cache_data(max_entries=16, show_spinner=False)
def compare(image_bytes: bytes, keys: tuple) -> list[dict]:
    rgb, _ = canonical(image_bytes)
    reg = registry()
    rows = []
    for key in keys:
        t0 = time.perf_counter()
        r = reg.predict(rgb, key)
        r["total_ms"] = 1000 * (time.perf_counter() - t0)
        rows.append(r)
    return rows


def blend(rgb: np.ndarray, heat_rgba: np.ndarray, opacity: float) -> np.ndarray:
    """Grad-CAM heat map over the photo; the map's own alpha (= strength) is scaled by `opacity`."""
    heat = np.asarray(Image.fromarray(heat_rgba).resize(rgb.shape[1::-1], Image.BILINEAR)).astype(np.float32)
    alpha = heat[..., 3:4] / 255.0 * opacity
    return (rgb.astype(np.float32) * (1 - alpha) + heat[..., :3] * alpha).clip(0, 255).astype(np.uint8)


# ------------------------------------------------------------------------------------------------------ samples
@st.cache_data(show_spinner=False)
def samples() -> list[dict]:
    """Demo photos shipped with the app (Tomato-Village dataset, cited): one held-out test photo per class first,
    then three more per class."""
    out = []
    manifest = json.loads((PUBLIC / "test-samples" / "manifest.json").read_text(encoding="utf-8"))
    for s in manifest["samples"]:
        out.append({"key": s["key"], "name": s["name"], "path": PUBLIC / s["file"].lstrip("/"), "held_out": True})
    for p in sorted((PUBLIC / "samples").glob("*.jpg")):
        key = p.stem.rsplit("-", 1)[0]
        if key in CLASS_NAME:
            out.append({"key": key, "name": CLASS_NAME[key], "path": p, "held_out": False})
    return out


def sample_bytes(path) -> bytes:
    return Path(path).read_bytes()


# ------------------------------------------------------------------------------------------------- result helpers
def model_result(key: str) -> dict | None:
    return next((m for m in platform()["models"] if m["key"] == key), None)


def task_defaults(task: str) -> tuple[dict, dict]:
    t = platform()["tasks"][task]
    return model_result(t["default_model"]), model_result(t["previous_default"])


def headline_comparison(task: str) -> dict | None:
    t = platform()["tasks"][task]
    return next((c for c in platform()["comparisons"]
                 if c["task"] == task and c["a"] == t["default_model"] and c["b"] == t["previous_default"]), None)


def pct(x, digits=1) -> str:
    return "—" if x is None else f"{100 * x:.{digits}f}%"


def pts(x, digits=1) -> str:
    return "—" if x is None else f"{100 * x:+.{digits}f} pts"


def class_index(key: str) -> int:
    return CLASS_KEYS.index(key)
