import os
import re

import altair as alt
import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
import torch
from transformers import AutoModel, AutoTokenizer

# ----------------------------------------------------------------------------
# Settings. Keep POOLING identical to what you used when you trained the model.
# ----------------------------------------------------------------------------
ESM_NAME = "facebook/esm2_t30_150M_UR50D"
MODEL_FILES = ["petase_xgboost_model.pkl", "petase_xgboost_model (1).pkl"]
UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/A0A0K8P6T7.fasta"
WT_OVERRIDE_FILE = "wt_sequence.txt"  # optional: put the exact WT sequence you trained on here
POOLING = "cls"  # "cls" or "mean"
MAX_LEN = 300  # tokens, including the two special tokens
BATCH_SIZE = 8
MAX_BATCH_ROWS = 500
MAX_SCAN_POSITIONS = 25
AA = "ACDEFGHIKLMNPQRSTVWY"

PRESETS = {
    "Wild type": "",
    "ThermoPETase": "S121E/D186H/R280A",
    "FAST-PETase": "S121E/D186H/R224Q/N233K/R280A",
}

st.set_page_config(page_title="ThermoPET", page_icon="🧬", layout="wide")


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading ESM-2 and the XGBoost model...")
def load_models():
    tokenizer = AutoTokenizer.from_pretrained(ESM_NAME)
    esm_model = AutoModel.from_pretrained(ESM_NAME)
    esm_model.eval()
    path = next((p for p in MODEL_FILES if os.path.exists(p)), None)
    if path is None:
        raise FileNotFoundError("XGBoost model file not found: " + ", ".join(MODEL_FILES))
    xgb_model = joblib.load(path)
    return tokenizer, esm_model, xgb_model


@st.cache_resource
def embedding_cache():
    return {}


def clean_sequence(text):
    lines = [
        ln.strip()
        for ln in text.strip().splitlines()
        if ln.strip() and not ln.strip().startswith(">")
    ]
    seq = "".join("".join(lines).split()).upper()
    if not seq:
        raise ValueError("The sequence is empty.")
    bad = sorted(set(seq) - set(AA))
    if bad:
        raise ValueError("Invalid characters in sequence: " + ", ".join(bad))
    return seq


@st.cache_data(show_spinner=False)
def get_wt_sequence():
    if os.path.exists(WT_OVERRIDE_FILE):
        with open(WT_OVERRIDE_FILE) as f:
            return clean_sequence(f.read()), WT_OVERRIDE_FILE
    r = requests.get(UNIPROT_URL, timeout=20)
    r.raise_for_status()
    return clean_sequence(r.text), "UniProt A0A0K8P6T7"


def wt_problems(wt):
    """Known ThermoPETase positions (UniProt numbering) should be S121, D186, R280."""
    expected = {121: "S", 186: "D", 280: "R"}
    out = []
    for pos, aa in expected.items():
        if len(wt) < pos or wt[pos - 1] != aa:
            found = wt[pos - 1] if len(wt) >= pos else "none"
            out.append(f"position {pos} should be {aa} but is {found}")
    return out


# ----------------------------------------------------------------------------
# Mutations
# ----------------------------------------------------------------------------
MUT_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")


def apply_mutations(wt, text):
    text = text.strip().upper()
    if not text:
        return wt, []
    tokens = [t for t in re.split(r"[\s,;/+]+", text) if t]
    seq = list(wt)
    seen = set()
    canon = []
    for tok in tokens:
        m = MUT_RE.match(tok)
        if not m:
            raise ValueError(f"Could not read '{tok}'. Use the format S121E.")
        old, pos, new = m.group(1), int(m.group(2)), m.group(3)
        if pos < 1 or pos > len(wt):
            raise ValueError(f"Position {pos} is outside the sequence (length {len(wt)}).")
        if pos in seen:
            raise ValueError(f"Position {pos} appears more than once.")
        if wt[pos - 1] != old:
            raise ValueError(f"{tok}: wild type has {wt[pos - 1]} at position {pos}, not {old}.")
        if new not in AA:
            raise ValueError(f"{tok}: '{new}' is not a standard amino acid.")
        seen.add(pos)
        seq[pos - 1] = new
        canon.append(f"{old}{pos}{new}")
    return "".join(seq), canon


def diff_from_wt(wt, seq):
    if len(wt) != len(seq):
        return None
    return [f"{a}{i + 1}{b}" for i, (a, b) in enumerate(zip(wt, seq)) if a != b]


# ----------------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------------
def predict_many(seqs, progress=None):
    tokenizer, esm_model, xgb_model = load_models()
    cache = embedding_cache()
    todo = [s for s in dict.fromkeys(seqs) if s not in cache]
    for i in range(0, len(todo), BATCH_SIZE):
        chunk = todo[i : i + BATCH_SIZE]
        enc = tokenizer(
            chunk, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN
        )
        with torch.no_grad():
            hidden = esm_model(**enc).last_hidden_state
        if POOLING == "cls":
            emb = hidden[:, 0, :]
        else:
            mask = enc["attention_mask"].clone()
            lengths = mask.sum(dim=1)
            mask[:, 0] = 0
            mask[torch.arange(mask.size(0)), lengths - 1] = 0
            mask = mask.unsqueeze(-1).float()
            emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        emb = emb.numpy()
        for s, e in zip(chunk, emb):
            cache[s] = e
        if progress:
            progress.progress(min(1.0, (i + len(chunk)) / max(1, len(todo))))
    X = np.vstack([cache[s] for s in seqs])
    return xgb_model.predict(X).astype(float)


def df_download(df, name, label="Download CSV"):
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), file_name=name, mime="text/csv")


# ----------------------------------------------------------------------------
# Header and wild type
# ----------------------------------------------------------------------------
st.title("ThermoPET")
st.caption("Predict the melting temperature (Tm) of IsPETase mutants from sequence")

try:
    WT, WT_SOURCE = get_wt_sequence()
except Exception as e:
    st.error(
        "Could not load the wild type sequence. Add a file named wt_sequence.txt to the "
        f"repository with the IsPETase sequence. Details: {e}"
    )
    st.stop()

problems = wt_problems(WT)
if problems:
    st.warning(
        "The wild type sequence does not match the expected UniProt numbering ("
        + "; ".join(problems)
        + "). Check wt_sequence.txt against the sequence used for training."
    )

with st.sidebar:
    st.subheader("Wild type")
    st.write(f"Source: {WT_SOURCE}")
    st.write(f"Length: {len(WT)} residues")
    st.caption("Mutation numbering follows the full length UniProt sequence, for example S121E.")
    with st.expander("Show sequence"):
        st.code(WT, language=None)

if "history" not in st.session_state:
    st.session_state.history = []

tab_predict, tab_batch, tab_scan, tab_about = st.tabs(
    ["Predict", "Batch", "Mutation scanner", "About"]
)

# ----------------------------------------------------------------------------
# Predict tab
# ----------------------------------------------------------------------------
with tab_predict:
    mode = st.radio("Input type", ["Mutations on wild type", "Full sequence"], horizontal=True)

    mutant_seq, label, mut_list = None, "", []
    if mode == "Mutations on wild type":
        c1, c2 = st.columns([1, 2])
        preset = c1.selectbox("Example", list(PRESETS.keys()), index=1)
        muts_text = c2.text_input(
            "Mutations (separate with / or commas)",
            value=PRESETS[preset],
            key=f"muts_{preset}",
            placeholder="S121E/D186H/R280A",
        )
        try:
            mutant_seq, mut_list = apply_mutations(WT, muts_text)
            label = "/".join(mut_list) if mut_list else "Wild type"
        except ValueError as e:
            st.error(str(e))
    else:
        raw = st.text_area("Protein sequence (plain text or FASTA)", value=WT, height=160)
        try:
            mutant_seq = clean_sequence(raw)
            d = diff_from_wt(WT, mutant_seq)
            if d is None:
                st.info(f"Length is {len(mutant_seq)}, wild type is {len(WT)}, so mutations cannot be listed.")
                label = "Custom sequence"
            else:
                mut_list = d
                label = "/".join(d) if d else "Wild type"
                st.caption("Differences from wild type: " + (", ".join(d) if d else "none"))
            if len(mutant_seq) > MAX_LEN - 2:
                st.warning(f"Sequences longer than {MAX_LEN - 2} residues are cut off before prediction.")
        except ValueError as e:
            st.error(str(e))
            mutant_seq = None

    if st.button("Predict melting temperature", type="primary", disabled=mutant_seq is None):
        with st.spinner("Predicting..."):
            wt_tm, mut_tm = predict_many([WT, mutant_seq])
        delta = mut_tm - wt_tm
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted Tm", f"{mut_tm:.2f} °C")
        m2.metric("Wild type (same model)", f"{wt_tm:.2f} °C")
        m3.metric("Change vs wild type", f"{delta:+.2f} °C", delta=f"{delta:+.2f} °C")
        if abs(delta) < 0.5:
            st.info("Predicted to be about as stable as the wild type.")
        elif delta > 0:
            st.success("Predicted to be more stable than the wild type.")
        else:
            st.warning("Predicted to be less stable than the wild type.")
        st.session_state.history.append(
            {"variant": label, "predicted_tm_C": round(mut_tm, 2),
             "wt_tm_C": round(wt_tm, 2), "delta_C": round(delta, 2)}
        )

    if st.session_state.history:
        st.subheader("This session")
        hist = pd.DataFrame(st.session_state.history)
        st.dataframe(hist, use_container_width=True, hide_index=True)
        if len(hist) > 1:
            st.bar_chart(hist.set_index("variant")["delta_C"])
        c1, c2 = st.columns(2)
        with c1:
            df_download(hist, "thermopet_session.csv")
        with c2:
            if st.button("Clear history"):
                st.session_state.history = []
                st.rerun()

# ----------------------------------------------------------------------------
# Batch tab
# ----------------------------------------------------------------------------
with tab_batch:
    st.write(
        "Upload a CSV with a column named mutations (for example S121E/D186H) or sequence. "
        "An optional name column is used as the label."
    )
    up = st.file_uploader("CSV file", type=["csv"])
    pasted = st.text_area(
        "Or paste one variant per line (mutations such as S121E/D186H, or a full sequence)",
        height=120,
    )

    rows = []
    if up is not None:
        try:
            df_in = pd.read_csv(up)
            cols = {c.lower().strip(): c for c in df_in.columns}
            key = cols.get("mutations") or cols.get("sequence")
            if key is None:
                st.error("The CSV needs a column named mutations or sequence.")
            else:
                is_seq = key.lower().strip() == "sequence"
                names = df_in[cols["name"]] if "name" in cols else None
                for i, v in enumerate(df_in[key].fillna("").astype(str)):
                    rows.append((str(names.iloc[i]) if names is not None else None, v, is_seq))
        except Exception as e:
            st.error(f"Could not read the file: {e}")
    elif pasted.strip():
        for line in pasted.strip().splitlines():
            line = line.strip()
            if line:
                rows.append((None, line, len(line) > 40 and MUT_RE.match(line.split("/")[0].upper()) is None))

    if rows and st.button("Run batch prediction", type="primary"):
        if len(rows) > MAX_BATCH_ROWS:
            st.warning(f"Only the first {MAX_BATCH_ROWS} rows are used.")
            rows = rows[:MAX_BATCH_ROWS]
        valid, errors = [], []
        for name, value, is_seq in rows:
            try:
                if is_seq:
                    seq = clean_sequence(value)
                    d = diff_from_wt(WT, seq)
                    if d is None:
                        lab = name or "Custom sequence"
                    else:
                        lab = name or ("/".join(d) if d else "Wild type")
                else:
                    seq, ml = apply_mutations(WT, value)
                    lab = name or ("/".join(ml) if ml else "Wild type")
                valid.append((lab, seq))
            except ValueError as e:
                errors.append({"input": value[:60], "problem": str(e)})
        if valid:
            bar = st.progress(0.0)
            wt_tm = predict_many([WT])[0]
            preds = predict_many([s for _, s in valid], progress=bar)
            bar.empty()
            out = pd.DataFrame(
                {
                    "variant": [l for l, _ in valid],
                    "predicted_tm_C": np.round(preds, 2),
                    "delta_vs_wt_C": np.round(preds - wt_tm, 2),
                }
            ).sort_values("predicted_tm_C", ascending=False)
            st.write(f"Wild type prediction from the same model: {wt_tm:.2f} °C")
            st.dataframe(out, use_container_width=True, hide_index=True)
            df_download(out, "thermopet_batch.csv")
        if errors:
            st.warning(f"{len(errors)} row(s) could not be used.")
            st.dataframe(pd.DataFrame(errors), use_container_width=True, hide_index=True)

# ----------------------------------------------------------------------------
# Scanner tab
# ----------------------------------------------------------------------------
with tab_scan:
    st.write(
        "Predict every single amino acid substitution across a range of positions and see "
        "which ones the model expects to raise or lower Tm."
    )
    c1, c2 = st.columns(2)
    start = c1.number_input("First position", 1, len(WT), 118)
    end = c2.number_input("Last position", 1, len(WT), 124)
    n_pos = int(end) - int(start) + 1
    if n_pos < 1:
        st.error("The last position must be at or after the first.")
    elif n_pos > MAX_SCAN_POSITIONS:
        st.error(f"Choose at most {MAX_SCAN_POSITIONS} positions at a time.")
    else:
        st.caption(f"{n_pos * 19} variants will be predicted. On a CPU this can take a few minutes.")
        if st.button("Run scan", type="primary"):
            variants = []
            for pos in range(int(start), int(end) + 1):
                for aa in AA:
                    if aa != WT[pos - 1]:
                        s = WT[: pos - 1] + aa + WT[pos:]
                        variants.append((pos, WT[pos - 1], aa, s))
            bar = st.progress(0.0)
            wt_tm = predict_many([WT])[0]
            preds = predict_many([v[3] for v in variants], progress=bar)
            bar.empty()
            scan = pd.DataFrame(
                {
                    "position": [v[0] for v in variants],
                    "wt_residue": [v[1] for v in variants],
                    "substitution": [v[2] for v in variants],
                    "mutation": [f"{v[1]}{v[0]}{v[2]}" for v in variants],
                    "predicted_tm_C": np.round(preds, 2),
                    "delta_vs_wt_C": np.round(preds - wt_tm, 2),
                }
            )
            st.session_state.scan = scan
            st.session_state.scan_wt = wt_tm

    if "scan" in st.session_state:
        scan = st.session_state.scan
        st.write(f"Wild type prediction: {st.session_state.scan_wt:.2f} °C")
        heat = (
            alt.Chart(scan)
            .mark_rect()
            .encode(
                x=alt.X("position:O", title="Position"),
                y=alt.Y("substitution:N", title="Substituted residue", sort=list(AA)),
                color=alt.Color(
                    "delta_vs_wt_C:Q",
                    title="Change in Tm (°C)",
                    scale=alt.Scale(scheme="redblue", domainMid=0, reverse=True),
                ),
                tooltip=["mutation", "predicted_tm_C", "delta_vs_wt_C"],
            )
            .properties(height=420)
        )
        st.altair_chart(heat, use_container_width=True)
        st.subheader("Top 10 predicted stabilising substitutions")
        top = scan.sort_values("delta_vs_wt_C", ascending=False).head(10)
        st.dataframe(
            top[["mutation", "predicted_tm_C", "delta_vs_wt_C"]],
            use_container_width=True,
            hide_index=True,
        )
        df_download(scan, "thermopet_scan.csv", "Download full scan")

# ----------------------------------------------------------------------------
# About tab
# ----------------------------------------------------------------------------
with tab_about:
    st.subheader("How it works")
    st.write(
        "The sequence is passed through the ESM-2 150M protein language model "
        f"({ESM_NAME}). The {'first token (CLS)' if POOLING == 'cls' else 'mean residue'} "
        "embedding, 640 numbers per sequence, is given to an XGBoost regressor that predicts Tm in °C. "
        "The change versus wild type is calculated from the same model, so both numbers come from "
        "the same predictor."
    )
    st.subheader("Things to keep in mind")
    st.write(
        "Predictions are estimates from a model trained on a limited set of PETase variants. "
        "They are most reliable for variants with a few point mutations close to those in the "
        "training data, and less reliable for large changes or sequences far from IsPETase. "
        "Treat the output as a way to rank candidates before lab testing, not as a measured Tm."
    )
    st.write(
        "Add your test set metrics (for example MAE and R squared), training set size and data "
        "sources here so users can judge how far to trust the numbers."
    )
    st.caption("Code: github.com/bigthinker-code/ThermoPet")
