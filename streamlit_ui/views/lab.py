"""Model lab: every model on identical test photos — leaderboard, paired tests, per-class results, confusion matrices,
calibration and abstention, robustness, speed, and the cross-validation that chose the version-2 models."""
import pandas as pd
import streamlit as st

from .. import charts, core, theme


def _verdict(comp: dict) -> str:
    lo, hi = comp["accuracy_ci"]
    if lo > 0:
        return "Version 2 is more accurate, and the 95% interval excludes zero."
    if hi < 0:
        return "Version 2 is less accurate, and the 95% interval excludes zero."
    word = "higher" if comp["accuracy_diff"] > 0 else "lower" if comp["accuracy_diff"] < 0 else "equal"
    return (f"Version 2's accuracy is {word}, but the 95% interval includes zero: on 290 test photos the "
            "difference is within noise.")


def _leaderboard(models: list[dict]):
    rows = []
    for m in sorted(models, key=lambda m: -m["test"]["f1_macro"]):
        t, s = m["test"], m["selective"]
        rows.append({
            "Model": ("★ " if m["is_default"] else "") + m["name"], "Version": charts.generation(m),
            "Accuracy": t["accuracy"], "Accuracy 95% CI": f"{100 * t['accuracy_ci'][0]:.1f}–{100 * t['accuracy_ci'][1]:.1f}%",
            "Macro-F1": t["f1_macro"], "F1 95% CI": f"{t['f1_macro_ci'][0]:.3f}–{t['f1_macro_ci'][1]:.3f}",
            "ECE": t.get("ece"), "Answers": s.get("coverage"), "Accuracy when answering": s.get("accuracy_kept"),
            "CPU ms": m.get("latency_ms"),
            "Selection score": (m["cv"] or {}).get("f1_macro") or (m["val"] or {}).get("f1_macro"),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
        "Accuracy": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
        "Macro-F1": st.column_config.ProgressColumn(format="%.3f", min_value=0, max_value=1,
                                                    help="Mean F1 over the conditions: every condition counts equally"),
        "ECE": st.column_config.NumberColumn(format="%.3f", help="Expected calibration error (lower is better)"),
        "Answers": st.column_config.NumberColumn(format="percent", help="Share of photos given a diagnosis (not "
                                                                         "'not sure' or 'unusual')"),
        "Accuracy when answering": st.column_config.NumberColumn(format="percent"),
        "CPU ms": st.column_config.NumberColumn(format="%.0f", help="Median time per photo on an 8-thread laptop CPU"),
        "Selection score": st.column_config.NumberColumn(
            format="%.3f", help="Macro-F1 on the data the model was chosen with: out-of-fold (v2) or the "
                                "validation split (v1). Not comparable with test scores."),
    })
    st.caption("★ = deployed default. Confidence intervals: 2,000 bootstrap resamples of whole leaf groups.")


def _per_class(models: list[dict], task: str):
    new, old = core.task_defaults(task)
    picks = [new, old]
    best_classical = max((m for m in models if m["family"] == "Classical ML"), key=lambda m: m["test"]["f1_macro"], default=None)
    if best_classical:
        picks.append(best_classical)
    label = {new["key"]: f"{new['name']} (v2)", old["key"]: f"{old['name']} (v1)"}
    rows = [{"Condition": c["name"], "Model": label.get(m["key"], m["name"]), "F1": c["f1"], "Support": c["support"]}
            for m in picks for c in m["per_class"]]
    st.altair_chart(charts.per_class(rows), width="stretch")
    table = pd.DataFrame([{"Condition": c["name"], "Test photos": c["support"], "Precision": c["precision"],
                           "Recall": c["recall"], "F1": c["f1"], "F1 (v1)": o["f1"]}
                          for c, o in zip(new["per_class"], old["per_class"])])
    st.dataframe(table, hide_index=True, width="stretch",
                 column_config={k: st.column_config.NumberColumn(format="%.3f") for k in ("Precision", "Recall", "F1", "F1 (v1)")})
    st.caption("Classes with very few test photos (Potassium Deficiency: 2, Nitrogen Deficiency: 8) have very "
               "uncertain per-class scores.")


def _confusion(models: list[dict], task: str):
    new, _ = core.task_defaults(task)
    keys = [m["key"] for m in models]
    key = st.selectbox("Model", keys, index=keys.index(new["key"]), key=f"cm_{task}",
                       format_func=lambda k: next(m["name"] for m in models if m["key"] == k))
    m = next(m for m in models if m["key"] == key)
    labels = [c["name"] for c in m["per_class"]]
    st.altair_chart(charts.confusion(m["confusion"], labels), width="stretch")
    st.caption("Rows: true condition; columns: prediction (before any 'not sure'). Colour: share of the row.")


def _calibration(models: list[dict], task: str):
    data = core.platform()
    new, old = core.task_defaults(task)
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("**Reliability of the version-2 model** (out-of-fold photos)")
        cal = new["calibration"]
        series = {}
        if cal.get("reliability_before"):
            series["Before temperature scaling"] = cal["reliability_before"]
        series["After temperature scaling"] = cal["reliability_after"]
        st.altair_chart(charts.reliability(series), width="stretch")
        if cal.get("ece_before") is not None:
            st.caption(f"Expected calibration error {cal['ece_before']:.3f} → {cal['ece_after']:.3f} on out-of-fold "
                       f"photos; {new['test']['ece']:.3f} on the test photos (v1: {old['test']['ece']:.3f}).")
    with b:
        st.markdown("**Saying “not sure”** (test photos)")
        sel = data["selective"][task]
        st.altair_chart(charts.risk_coverage(sel["curve"], sel["chosen"]), width="stretch")
        ch = sel["chosen"]
        st.caption(f"Rule (fixed on out-of-fold data): {sel['rule']}. Threshold {ch['threshold']:.2f} → answers "
                   f"{core.pct(ch['coverage'])} of test photos with {core.pct(ch['accuracy_kept'])} accuracy. v1 at its "
                   f"own threshold: {core.pct(old['selective']['coverage'])} answered, "
                   f"{core.pct(old['selective']['accuracy_kept'])} accurate.")


def _robustness(task: str):
    data = core.platform()
    rb = data["audit"]["robustness"][task]
    series = {s["key"]: s["label"] for s in rb["series"]}
    rows = [{"Change": r["label"], "Model": label, "Changed predictions": r[key]}
            for r in rb["rows"] for key, label in series.items() if key in r]
    st.altair_chart(charts.grouped_bars(rows, "Change", "Model", "Changed predictions",
                                        "Share of test predictions that change", colors=(charts.GREEN, charts.CYAN)),
                    width="stretch")
    st.caption("Each test photo is changed harmlessly (re-compressed, blurred, brightened, halved) or turned back into "
               "a full-size phone photo and uploaded through the old or the new upload path. Lower is better.")
    pre = data["audit"]["preprocessing"]
    if pre:
        st.markdown("**The upload fix, measured on validation photos** (original default model)")
        st.dataframe(pd.DataFrame([{"Condition": p["label"], "Accuracy": p["accuracy"], "Macro-F1": p["f1_macro"],
                                    "Changed predictions": p["flip_rate"]} for p in pre]),
                     hide_index=True, width="stretch", column_config={
                         "Accuracy": st.column_config.NumberColumn(format="percent"),
                         "Macro-F1": st.column_config.NumberColumn(format="%.3f"),
                         "Changed predictions": st.column_config.NumberColumn(format="percent")})


def _cross_validation(task: str):
    data = core.platform()
    recipes = [r for r in data["cv"]["recipes"] if r["task"] == task]
    st.markdown(f"**Selection rule.** {data['cv']['selection_rule']}")
    st.dataframe(pd.DataFrame([{
        "Recipe": ("★ " if r["chosen"] else "") + r["label"], "Macro-F1 (1 view)": r["f1_1"], "Macro-F1 (8 views)": r["f1_8"],
        "Accuracy (1 view)": r["acc_1"], "Accuracy (8 views)": r["acc_8"], "Training (min, 5 folds)": r["train_minutes"]}
        for r in recipes]), hide_index=True, width="stretch", column_config={
            "Macro-F1 (1 view)": st.column_config.NumberColumn(format="%.3f"),
            "Macro-F1 (8 views)": st.column_config.NumberColumn(format="%.3f"),
            "Accuracy (1 view)": st.column_config.NumberColumn(format="percent"),
            "Accuracy (8 views)": st.column_config.NumberColumn(format="percent"),
            "Training (min, 5 folds)": st.column_config.NumberColumn(format="%.0f")})
    st.caption("Out-of-fold scores: every one of the 1,813 train+validation photos is predicted by the fold model "
               "that never saw its leaf. ★ = member of the deployed model.")
    if recipes:
        st.altair_chart(charts.fold_strip(recipes), width="stretch")
    screen = [s for s in data["cv"]["screening"] if s["task"] == task]
    if screen:
        with st.expander("Every recipe tried (fold 0), including the ones stopped early"):
            st.dataframe(pd.DataFrame([{"Recipe": s["label"], "Macro-F1 (1 view)": s["f1_1"], "Macro-F1 (8 views)": s["f1_8"],
                                        "Accuracy (1 view)": s["acc_1"], "All 5 folds run": s["full_cv"]} for s in screen]),
                         hide_index=True, width="stretch")


def render() -> None:
    theme.section("Model lab", "Every model — classical machine learning, the original study's networks and the "
                  "version-2 upgrade — scored by the same code on the same held-out test photos, once.",
                  kicker="Evidence")
    task = st.segmented_control("Task", ["5class", "8class"], format_func=lambda t: core.TASK_LABEL[t], default="5class",
                                key="lab_task", required=True, label_visibility="collapsed")
    data = core.platform()
    models = [m for m in data["models"] if m["task"] == task]
    new, old = core.task_defaults(task)
    comp = core.headline_comparison(task)
    n = data["tasks"][task]["test_photos"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Version 2 · accuracy", core.pct(new["test"]["accuracy"]),
              delta=f"{core.pts(new['test']['accuracy'] - old['test']['accuracy'])} vs v1", border=True)
    c2.metric("Version 2 · macro-F1", f"{new['test']['f1_macro']:.3f}",
              delta=f"{new['test']['f1_macro'] - old['test']['f1_macro']:+.3f} vs v1", border=True)
    c3.metric("Version 1 · accuracy", core.pct(old["test"]["accuracy"]), delta=old["name"], delta_color="off", border=True)
    c4.metric("Test photos", f"{n}", delta="scored once", delta_color="off", border=True)
    if comp:
        st.info(f"**Paired comparison (v2 − v1, bootstrap over leaf groups):** accuracy {core.pts(comp['accuracy_diff'])} "
                f"[95% CI {core.pts(comp['accuracy_ci'][0])}, {core.pts(comp['accuracy_ci'][1])}], macro-F1 "
                f"{comp['f1_diff']:+.3f} [{comp['f1_ci'][0]:+.3f}, {comp['f1_ci'][1]:+.3f}]. {_verdict(comp)}",
                icon=":material/balance:")
    tabs = st.tabs(["Leaderboard", "Per condition", "Confusion", "Calibration", "Robustness", "Speed", "Cross-validation"])
    with tabs[0]:
        _leaderboard(models)
        st.altair_chart(charts.metric_with_ci(models, "f1_macro", "Macro-F1 (test, 95% CI)"), width="stretch")
    with tabs[1]:
        _per_class(models, task)
    with tabs[2]:
        _confusion(models, task)
    with tabs[3]:
        _calibration(models, task)
    with tabs[4]:
        _robustness(task)
    with tabs[5]:
        st.altair_chart(charts.latency_scatter(models), width="stretch")
        st.caption("Median CPU time per photo through the API's own predictor (ONNX Runtime, 8 threads, laptop CPU). "
                   "The online demo runs on a smaller shared server and is slower.")
    with tabs[6]:
        _cross_validation(task)
