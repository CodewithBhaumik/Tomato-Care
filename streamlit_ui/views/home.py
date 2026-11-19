"""Home: what Tomato Care does, the headline results (v2 against v1) and what changed."""
import base64
import io

import numpy as np
import streamlit as st
from PIL import Image

from .. import core, nav, theme


def _hero_figure(hero: dict | None) -> str:
    if not hero:
        return ""
    rgb = np.asarray(Image.open(core.PUBLIC / hero["input"].lstrip("/")).convert("RGB"))
    heat = np.asarray(Image.open(core.PUBLIC / hero["heatmap"].lstrip("/")).convert("RGBA"))
    composite = Image.fromarray(core.blend(rgb, heat, 0.75)).resize((520, 520), Image.LANCZOS)
    buf = io.BytesIO()
    composite.save(buf, format="JPEG", quality=88)
    src = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    name = core.CLASS_NAME.get(hero["pred_key"], hero["pred_key"])
    return (f'<div class="tc-hero-figure"><img src="{src}" alt="Grad-CAM heat map on a tomato leaf">'
            f'<div class="tc-hero-badge"><div><b>{theme.esc(name)}</b><br><span style="opacity:.75">Held-out test photo · '
            f'Grad-CAM of the focused model</span></div><div class="p">{100 * hero["confidence"]:.0f}%</div></div></div>')


def render() -> None:
    data = core.platform()
    new5, old5 = core.task_defaults("5class")
    new8, old8 = core.task_defaults("8class")

    with st.container(key="hero"):
        left, right = st.columns([1.25, 1], gap="large", vertical_alignment="center")
        with left:
            st.html('<span class="tc-eyebrow">Tomato leaf diagnosis · version 2</span>'
                    '<h1 class="tc-hero-title">Healthy leaf or a problem?<br><span>Know from one photo.</span></h1>'
                    '<p class="tc-hero-sub">Tomato Care recognises diseases, a pest and nutrient deficiencies on tomato '
                    'leaves. It tells you how sure it is, says <b>“not sure”</b> instead of guessing, flags photos unlike '
                    'anything it was trained on, and shows where it looked.</p>')
            st.space("small")
            with st.container(horizontal=True):
                if st.button("Diagnose a leaf", type="primary", icon=":material/photo_camera:"):
                    nav.go("diagnose")
                if st.button("See the evidence", icon=":material/monitoring:"):
                    nav.go("lab")
        with right:
            st.html(_hero_figure(data.get("hero")))

    st.space("medium")
    theme.section("Measured, not claimed",
                  "Every number below comes from the same 290 held-out test photos, scored once after every model "
                  "choice was fixed on cross-validation data. Differences are against the original study (v1).",
                  kicker="Results")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Focused · accuracy", core.pct(new5["test"]["accuracy"]),
              delta=f"{core.pts(new5['test']['accuracy'] - old5['test']['accuracy'])} vs v1", border=True,
              help="Five conditions (the app's default task), 265 test photos. 95% CI "
                   f"{core.pct(new5['test']['accuracy_ci'][0])}–{core.pct(new5['test']['accuracy_ci'][1])}.")
    sel = new5["selective"]
    c2.metric("Focused · when it answers", core.pct(sel["accuracy_kept"]),
              delta=f"answers {core.pct(sel['coverage'], 0)} of photos", delta_color="off", border=True,
              help="Below its confidence threshold (or for an unusual photo) the app says 'not sure' instead. "
                   f"v1 at its own threshold: {core.pct(old5['selective']['accuracy_kept'])} on "
                   f"{core.pct(old5['selective']['coverage'], 0)} of photos.")
    c3.metric("8 conditions · accuracy", core.pct(new8["test"]["accuracy"]),
              delta=f"{core.pts(new8['test']['accuracy'] - old8['test']['accuracy'])} vs v1", border=True,
              help="Includes the three hard classes (Early Blight, Late Blight, Potassium Deficiency). 95% CI "
                   f"{core.pct(new8['test']['accuracy_ci'][0])}–{core.pct(new8['test']['accuracy_ci'][1])}.")
    far = data["ood"]["test"]["flagged_far"]
    c4.metric("Non-leaf photos caught", core.pct(far, 0), delta="new in v2 · v1 had no check", delta_color="off",
              border=True, help="Share of held-out non-leaf probe images (photos, textures, drawings) flagged as "
                                "'Unusual image' instead of receiving a diagnosis.")
    comp5, comp8 = core.headline_comparison("5class"), core.headline_comparison("8class")
    if comp5 and comp8:
        st.caption(
            f"Paired bootstrap over leaf groups (v2 − v1): focused task accuracy {core.pts(comp5['accuracy_diff'])} "
            f"(95% CI {core.pts(comp5['accuracy_ci'][0])} to {core.pts(comp5['accuracy_ci'][1])}); full task "
            f"{core.pts(comp8['accuracy_diff'])} (95% CI {core.pts(comp8['accuracy_ci'][0])} to "
            f"{core.pts(comp8['accuracy_ci'][1])}). Intervals that include 0 mean the difference is within the noise "
            "of a 290-photo test set.")

    st.space("medium")
    theme.section("What changed in version 2", kicker="Upgrade")
    rows = [
        [("tune", "Sees uploads the way it was trained",
          "Dataset photos are whole 3:4 phone frames squashed to 256×256 in Display P3 colour. Uploads now go through "
          "the same steps (the old app centre-cropped them and ignored colour profiles)."),
         ("account_tree", "Chosen by cross-validation",
          "Five-fold cross-validation on 1,813 photos, leaf groups never split, replaced a single 311-photo "
          "validation split on which one photo could move the score by 0.12."),
         ("groups", "A committee of five",
          "Each deployed model is the five fold networks voting together; every member has data it never saw, so "
          "its calibration was fitted honestly.")],
        [("help", "Says “not sure”",
          "Temperature-scaled confidence and a threshold chosen on out-of-fold data: below it, no diagnosis."),
         ("image_not_supported", "Flags unusual photos",
          "Feature-space distance to the training photos catches non-leaf images and many unknown conditions."),
         ("visibility", "Shows where it looked",
          "Grad-CAM heat maps for every diagnosis, so you can check the model looked at the leaf.")],
    ]
    for row in rows:
        cols = st.columns(3)
        for col, (icon, title, body) in zip(cols, row):
            with col:
                theme.feature(icon, title, body)

    st.space("medium")
    theme.section("How it works", kicker="Three steps")
    s1, s2, s3 = st.columns(3)
    for col, (n, title, body) in zip((s1, s2, s3), [
        (1, "Photograph one leaf", "On plain paper, in daylight, leaf filling most of the frame — like the training photos."),
        (2, "Same steps as training", "EXIF rotation, colour into Display P3, the whole frame to 256×256, JPEG quality 90."),
        (3, "A calibrated answer", "Five networks vote; you get the condition and its confidence, or an honest “not sure”."),
    ]):
        with col:
            st.html(f'<span class="tc-step">{n}</span><b>{title}</b>')
            st.caption(body)
    st.space("small")
    with st.container(horizontal=True):
        nav.link("diagnose", "Try it on an example photo", icon=":material/arrow_forward:")
        nav.link("method", "Read the methodology", icon=":material/menu_book:")
