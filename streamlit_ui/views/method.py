"""How it works: the data, the leakage-free split, the upload fix, model selection, calibration, the unusual-image
check, explanations, limitations and how to reproduce every number."""
import pandas as pd
import streamlit as st

from .. import charts, core, nav, theme


def render() -> None:
    data = core.platform()
    ds = data["dataset"]
    theme.section("How Tomato Care works", "The data, the checks and the decisions behind every number in this app — "
                  "including the results that did not improve.", kicker="Methodology")

    with st.container(border=True):
        st.markdown("**In short**")
        st.markdown(
            f"- **Data:** {ds['photos']:,} distinct photographs of single tomato leaves on paper ({ds['files']:,} files "
            f"with augmented copies), eight classes, one phone, two districts of Rajasthan.\n"
            f"- **Honest split:** whole leaf groups ({ds['groups']:,}) never cross train, validation and test.\n"
            "- **Same input as training:** uploads are converted exactly like the dataset photos.\n"
            "- **Models chosen on cross-validation**, the test set scored **once** afterwards.\n"
            "- **Calibrated confidence**, an abstention threshold and an **unusual-image** check.")

    theme.section("1 · A benchmark without leakage", kicker="Data")
    leak = ds["leakage"]
    st.markdown(
        f"The dataset's official split puts augmented copies of one photograph on both sides: "
        f"**{core.pct(leak['test_with_train_sibling'])}** of the official test images have a sibling in the training "
        "folder. A model can score highly by recognising photos it has already seen, so Tomato Care splits **whole leaf "
        f"groups** (copies and re-shots of the same leaf) into {ds['split']['train']:,} / {ds['split']['val']:,} / "
        f"{ds['split']['test']:,} train / validation / test photographs.")
    bench = leak.get("benchmark") or []
    if bench:
        rows = [{"Model": b["model"], "Split": s, "Accuracy": b[k]} for b in bench
                for s, k in (("Official split (leaky)", "official_accuracy"), ("Leaf-group split (honest)", "clean_accuracy"))]
        st.altair_chart(charts.grouped_bars(rows, "Model", "Split", "Accuracy", "Test accuracy",
                                            colors=(charts.GRAY, charts.GREEN)), width="stretch")
        st.caption("Same recipes, two splits: the gap is the inflation caused by leakage.")

    theme.section("2 · Uploads that look like the training photos", kicker="Preprocessing")
    src = ds["source"]
    st.markdown(
        f"Every dataset photo was taken with a {src['phone']} at {src['resolution']} and stored as {src['stored']} in "
        f"{src['encoding']}. The original app centre-cropped uploads and ignored colour profiles, so a phone photo "
        "reached the model narrower and in different colours than any training photo — one cause of inconsistent "
        "answers. Uploads now follow the dataset's steps: EXIF rotation → colour into Display P3 → aspect ratio "
        "clamped to 3:4…4:3 → whole frame resized to 256×256 → JPEG quality 90. Dataset photos pass through "
        "bit-for-bit unchanged.")
    pre = data["audit"]["preprocessing"]
    if pre:
        st.dataframe(pd.DataFrame([{"Photo": p["label"], "Accuracy": p["accuracy"], "Changed predictions": p["flip_rate"]}
                                   for p in pre]), hide_index=True, width="stretch", column_config={
            "Accuracy": st.column_config.NumberColumn(format="percent"),
            "Changed predictions": st.column_config.NumberColumn(format="percent")})
        st.caption("Validation photos turned back into full-size phone photos and uploaded through each path "
                   "(original default model).")

    theme.section("3 · Choosing models without looking at the test set", kicker="Selection")
    st.markdown(
        f"Grouped, stratified **{ds['cv']['folds']}-fold cross-validation** over the {ds['cv']['photos']:,} train + "
        "validation photos. Each recipe trains for a fixed number of epochs (no early stopping on the held-out fold), "
        "every photo gets an out-of-fold prediction, and the rule below picks the deployed model. The deployed model "
        "is the five fold networks voting together.")
    st.markdown(f"> {data['cv']['selection_rule']}")
    st.markdown(
        "On the test set the ranking of close recipes did not hold: for the full 8-class task the ConvNeXt robust "
        "recipe alone scored higher than the chosen two-recipe ensemble. Re-choosing after seeing the test results "
        "would make the test score meaningless, so the pre-registered choice stays and both results are reported.")
    nav.link("lab", "See the cross-validation results in the Model lab", icon=":material/science:")

    theme.section("4 · Confidence you can trust", kicker="Calibration")
    new5, _ = core.task_defaults("5class")
    cal = new5["calibration"]
    a, b = st.columns([1.2, 1], gap="large")
    with a:
        series = {}
        if cal.get("reliability_before"):
            series["Before temperature scaling"] = cal["reliability_before"]
        series["After temperature scaling"] = cal["reliability_after"]
        st.altair_chart(charts.reliability(series), width="stretch")
    with b:
        st.markdown(
            "Networks are often over-confident. **Temperature scaling** (one number per model, fitted on out-of-fold "
            "predictions) makes a stated 80% mean about 80% right. The **answer threshold** is the lowest confidence "
            "at which out-of-fold accuracy on answered photos reaches the target — 95% for the focused task, 90% for "
            "all eight conditions. Below it, the app says **“not sure”**.")

    theme.section("4b · Class bias and even confidence", kicker="Fairness to rare conditions")
    st.markdown(
        "Leaf Miner makes up over half of the photographs, Potassium Deficiency 0.4%. Training already weights each "
        "class by its rarity; on top of that the served models apply two corrections fitted on out-of-fold data: "
        "**per-class calibration** (vector scaling), so a stated 90% means about 90% right for every condition, and "
        "a **class-prior correction** (logit adjustment, Menon et al. 2021), so frequent conditions cannot out-vote "
        "rare ones. The confidence shown stays the calibrated probability of the answer. Numbers: Model lab → "
        "Calibration. A GAN was not used: from 9 Potassium photographs it would reproduce those photos, not add "
        "information.")

    theme.section("5 · Recognising unusual photos", kicker="Out-of-distribution")
    ood = data["ood"]["test"]
    st.markdown(
        f"Each fold network measures how far a photo lies from its own training photos in feature space "
        f"({data['ood']['method']}). The limit keeps {core.pct(data['ood']['keep'], 0)} of held-out training-like photos; "
        "above it the app answers **“Unusual image”**.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Non-leaf images flagged", core.pct(ood["flagged_far"], 0), border=True,
              help="Held-out half of 30 probe images: photographs, textures, drawings, flat colours.")
    if ood.get("flagged_near") is not None:
        c2.metric("Out-of-scope leaves flagged", core.pct(ood["flagged_near"], 0), border=True,
                  help="Focused model: test photos of the three conditions it was not trained on.")
    c3.metric("Normal test photos flagged", core.pct(ood["flagged_in"], 1), border=True,
              help="The cost: in-scope test photos wrongly sent to 'Unusual image'.")
    st.caption("Unknown leaf conditions are hard to separate from known ones: most out-of-scope leaves still get a "
               "confident answer, which is why the focused model is only recommended for its five conditions.")

    theme.section("6 · Explanations", kicker="Grad-CAM")
    st.markdown("Grad-CAM weights the last convolutional feature maps by the gradient of the predicted class: bright "
                "regions raised that class's score. " + data["gradcam"].get("summary", ""))

    theme.section("Limitations", kicker="Read before use")
    st.markdown("\n".join(f"- {x}" for x in data["limitations"]))

    theme.section("Reproduce every number", kicker="Code")
    st.code("git clone https://github.com/CodewithBhaumik/Tomato-Care.git && cd Tomato-Care\n"
            "pip install -r requirements-train.txt\n"
            "python scripts/download_weights.py          # deployed weights (SHA-256 checked)\n"
            "streamlit run streamlit_app.py              # this app\n"
            "uvicorn backend.main:app --port 8000        # REST API + React site (after npm run build)", language="bash")
    st.markdown(f"Full methodology, model card and results: [docs on GitHub]({core.REPO_URL}/tree/main/docs). "
                f"Dataset: {ds['citation']} ([doi]({core.DOI_URL})).")
