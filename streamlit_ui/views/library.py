"""Disease library: the eight leaf conditions — what they look like, what to do, and how well each model knows them."""
import pandas as pd
import streamlit as st

from tomatoleaf.config import CLASS_KEYS

from .. import core, theme

FOCUSED = {"healthy", "leaf_miner", "magnesium_deficiency", "nitrogen_deficiency", "spotted_wilt_virus"}


def _photos(key: str) -> list:
    return [s for s in core.samples() if s["key"] == key]


def _scores(key: str) -> pd.DataFrame:
    rows = []
    for task in ("5class", "8class"):
        new, old = core.task_defaults(task)
        for m, ver in ((new, "v2"), (old, "v1")):
            c = next((c for c in m["per_class"] if c["key"] == key), None)
            if c:
                rows.append({"Model": f"{m['name']} ({ver})", "Task": core.TASK_LABEL[task], "Precision": c["precision"],
                             "Recall": c["recall"], "F1": c["f1"], "Test photos": c["support"]})
    return pd.DataFrame(rows)


def _detail(key: str) -> None:
    info = core.disease_info()[key]
    data = core.platform()
    left, right = st.columns([1, 1.3], gap="large")
    with left:
        photos = _photos(key)
        if photos:
            st.image(str(photos[0]["path"]), width="stretch", caption="Held-out test photo" if photos[0]["held_out"] else None)
            if len(photos) > 1:
                cols = st.columns(3)
                for col, s in zip(cols, photos[1:4]):
                    col.image(str(s["path"]), width="stretch")
        item = next((g for g in data["gradcam"]["items"] if g["sample"] == key), None)
        if item:
            st.markdown("**Where the models looked** (held-out test photo)")
            cols = st.columns(len(item["overlays"]))
            for col, o in zip(cols, item["overlays"]):
                verdict = "✓" if o["correct"] else ("outside its classes" if o["out_of_scope"] else "✗")
                col.image(str(core.PUBLIC / o["overlay"].lstrip("/")), width="stretch",
                          caption=f"{o['model_name']}: {core.CLASS_NAME.get(o['pred_key'], o['pred_key'])} "
                                  f"{100 * o['confidence']:.0f}% {verdict}")
    with right:
        chips = theme.chip(info.get("category", ""), "g" if info.get("is_healthy") else "r")
        chips += theme.chip("In the focused model" if key in FOCUSED else "Full model only", "c" if key in FOCUSED else "a")
        st.html(chips)
        if info.get("cause"):
            st.markdown(f"**Cause:** {info['cause']}")
        st.markdown(info["summary"])
        for title, field in (("Symptoms", "symptoms"), ("What it looks like in a photo", "visual_characteristics"),
                             ("Management", "management")):
            if info.get(field):
                st.markdown(f"**{title}**")
                st.markdown("\n".join(f"- {s}" for s in info[field]))
        df = _scores(key)
        if not df.empty:
            st.markdown("**How well the models recognise it** (test photos)")
            st.dataframe(df, hide_index=True, width="stretch", column_config={
                k: st.column_config.NumberColumn(format="%.2f") for k in ("Precision", "Recall", "F1")})
        st.caption(core.disease_info().get("_disclaimer", ""))


def render() -> None:
    theme.section("Disease library", "The eight conditions in the Tomato-Village dataset: symptoms, what they look like "
                  "in a photo, management, and how reliably each model recognises them.", kicker="Learn")
    info = core.disease_info()
    picked = st.session_state.get("lib_key", "leaf_miner")
    for row in (CLASS_KEYS[:4], CLASS_KEYS[4:]):
        cols = st.columns(4)
        for col, key in zip(cols, row):
            c = info[key]
            with col, st.container(border=True):
                photos = _photos(key)
                if photos:
                    st.image(str(photos[0]["path"]), width="stretch")
                st.markdown(f"**{c['name']}**")
                st.caption(c.get("category", ""))
                if st.button("Selected" if key == picked else "Read more", key=f"lib_{key}", width="stretch",
                             type="primary" if key == picked else "secondary"):
                    st.session_state["lib_key"] = key
                    st.rerun()
    st.space("small")
    with st.container(border=True):
        st.markdown(f"### {info[picked]['name']}")
        _detail(picked)
