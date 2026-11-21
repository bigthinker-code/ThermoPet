import json, os, re
import joblib, numpy as np, pandas as pd, requests, streamlit as st, torch
import plotly.graph_objects as go
import streamlit.components.v1 as components
from transformers import AutoModel, AutoTokenizer

# Settings. For version 2, change MODEL_FILES / ESM_NAME / POOLING here.
ESM_NAME = "facebook/esm2_t30_150M_UR50D"
MODEL_FILES = ["petase_xgboost_model.pkl", "petase_xgboost_model (1).pkl"]
POOLING = "cls"  # must match training: "cls" or "mean"
UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/A0A0K8P6T7.fasta"
WT_FILE = "wt_sequence.txt"  # optional override with the exact training WT
PDB_ID = "6EQE"  # IsPETase crystal structure, numbering follows UniProt
MAX_LEN, BATCH, MAX_ROWS, MAX_SCAN = 300, 8, 500, 25
AA = "ACDEFGHIKLMNPQRSTVWY"
TRIAD = {160: "S", 206: "D", 237: "H"}
PRESETS = {"Wild type": "", "ThermoPETase": "S121E/D186H/R280A",
           "FAST-PETase": "S121E/D186H/R224Q/N233K/R280A"}
AMBER, CYAN, BG, CARD = "#ffb020", "#22d3ee", "#07080c", "#10121a"

st.set_page_config(page_title="ThermoPET", page_icon="🧬", layout="wide")
st.markdown(f"""<style>
.stApp{{background:radial-gradient(1100px 520px at 8% -8%,#2a1a06 0,{BG} 58%)}}
header[data-testid=stHeader]{{background:transparent}}
.hero{{padding:28px 32px;border-radius:20px;border:1px solid #2a2f3d;margin-bottom:18px;
background:linear-gradient(120deg,#14161f,#0b0c11 60%,#1b1206)}}
.hero h1{{font-size:3rem;margin:0;letter-spacing:-1px}}
.hero h1 span{{background:linear-gradient(90deg,{CYAN},{AMBER},#fb7185);-webkit-background-clip:text;color:transparent}}
.hero p{{color:#9aa3b5;max-width:760px;margin:6px 0 14px}}
.kick{{color:{AMBER};font-size:.78rem;letter-spacing:3px;text-transform:uppercase}}
.chip{{display:inline-block;padding:4px 12px;margin:0 8px 6px 0;border-radius:99px;border:1px solid #2f3547;color:#cbd2e1;font-size:.8rem;background:#0f1118}}
div[data-testid=stMetric]{{background:{CARD};border:1px solid #232838;border-radius:14px;padding:14px 16px}}
.stButton>button[kind=primary]{{background:linear-gradient(90deg,#f59e0b,#fb7185);border:0;color:#111;font-weight:700}}
.stTabs [data-baseweb=tab]{{font-size:1rem;padding:10px 18px}}
.seq{{font-family:monospace;line-height:2;background:{CARD};border:1px solid #232838;border-radius:12px;padding:12px}}
.blk{{display:inline-block;margin:0 10px 0 0}} .blk i{{display:block;font-size:.6rem;color:#667085;font-style:normal}}
.mut{{color:#111;background:{AMBER};border-radius:3px;padding:0 2px}}
.leg span{{margin-right:18px;font-size:.85rem;color:#b6bdcc}} .dot{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}}
</style>""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading ESM-2 and the XGBoost model...")
def load_models():
    tok = AutoTokenizer.from_pretrained(ESM_NAME)
    esm = AutoModel.from_pretrained(ESM_NAME).eval()
    path = next((p for p in MODEL_FILES if os.path.exists(p)), None)
    if path is None:
        raise FileNotFoundError("Model file not found: " + ", ".join(MODEL_FILES))
    return tok, esm, joblib.load(path)

@st.cache_resource
def emb_cache():
    return {}

def clean(text):
    lines = [l.strip() for l in text.strip().splitlines() if l.strip() and not l.startswith(">")]
    s = "".join("".join(lines).split()).upper()
    if not s: raise ValueError("The sequence is empty.")
    bad = sorted(set(s) - set(AA))
    if bad: raise ValueError("Invalid characters: " + ", ".join(bad))
    return s

@st.cache_data(show_spinner=False)
def get_wt():
    if os.path.exists(WT_FILE):
        return clean(open(WT_FILE).read()), WT_FILE
    r = requests.get(UNIPROT_URL, timeout=20); r.raise_for_status()
    return clean(r.text), "UniProt A0A0K8P6T7"

MUT = re.compile(r"^([A-Z])(\d+)([A-Z])$")
def apply_muts(wt, text):
    toks = [t for t in re.split(r"[\s,;/+]+", text.strip().upper()) if t]
    seq, seen, out = list(wt), set(), []
    for t in toks:
        m = MUT.match(t)
        if not m: raise ValueError(f"Could not read '{t}'. Use the format S121E.")
        o, p, n = m.group(1), int(m.group(2)), m.group(3)
        if not 1 <= p <= len(wt): raise ValueError(f"Position {p} is outside the sequence.")
        if p in seen: raise ValueError(f"Position {p} appears twice.")
        if wt[p-1] != o: raise ValueError(f"{t}: wild type has {wt[p-1]} at {p}, not {o}.")
        if n not in AA: raise ValueError(f"{t}: not a standard amino acid.")
        seen.add(p); seq[p-1] = n; out.append(t)
    return "".join(seq), out

def diff(wt, s):
    return [f"{a}{i+1}{b}" for i, (a, b) in enumerate(zip(wt, s)) if a != b] if len(wt) == len(s) else None

def predict_many(seqs, bar=None):
    tok, esm, xgb = load_models()
    cache = emb_cache()
    todo = [s for s in dict.fromkeys(seqs) if s not in cache]
    for i in range(0, len(todo), BATCH):
        ch = todo[i:i+BATCH]
        enc = tok(ch, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN)
        with torch.no_grad():
            h = esm(**enc).last_hidden_state
        if POOLING == "cls":
            e = h[:, 0, :]
        else:
            m = enc["attention_mask"].clone(); n = m.sum(1)
            m[:, 0] = 0; m[torch.arange(m.size(0)), n-1] = 0
            m = m.unsqueeze(-1).float(); e = (h*m).sum(1)/m.sum(1).clamp(min=1)
        for s, v in zip(ch, e.numpy()): cache[s] = v
        if bar: bar.progress(min(1.0, (i+len(ch))/len(todo)))
    return xgb.predict(np.vstack([cache[s] for s in seqs])).astype(float)

def dl(df, name, label="Download CSV"):
    st.download_button(label, df.to_csv(index=False).encode(), file_name=name, mime="text/csv")

def dark(fig, h=300):
    fig.update_layout(height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#e8eaf0", margin=dict(t=30, b=10, l=10, r=10))
    return fig

def gauge(tm, wt):
    f = go.Figure(go.Indicator(mode="gauge+number+delta", value=tm, number={"suffix": " °C"},
        delta={"reference": wt, "suffix": " °C"},
        gauge={"axis": {"range": [min(30, tm-5), max(90, tm+5)]}, "bar": {"color": AMBER},
               "bgcolor": CARD, "threshold": {"line": {"color": CYAN, "width": 4}, "value": wt},
               "steps": [{"range": [0, 50], "color": "#0e2a33"}, {"range": [50, 70], "color": "#2a2210"},
                         {"range": [70, 200], "color": "#3a1a14"}]}))
    return dark(f, 270)

def seq_html(wt, s):
    blocks = []
    for i in range(0, len(s), 10):
        b = "".join(f'<b class="mut">{c}</b>' if i+j < len(wt) and c != wt[i+j] else c
                    for j, c in enumerate(s[i:i+10]))
        blocks.append(f'<span class="blk"><i>{i+1}</i>{b}</span>')
    return '<div class="seq">' + "".join(blocks) + "</div>"

def dcolor(d, m):
    t = max(-1, min(1, d/m)) if m else 0
    cold, mid, hot = (59, 130, 246), (55, 65, 81), (255, 176, 32)
    a, b = (mid, hot) if t >= 0 else (mid, cold)
    c = [int(a[i] + (b[i]-a[i])*abs(t)) for i in range(3)]
    return "#%02x%02x%02x" % tuple(c)

VIEW_JS = """<div id="v" style="width:100%;height:__H__px;border-radius:14px;overflow:hidden;position:relative"></div>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script><script>
const C=__CFG__;const v=$3Dmol.createViewer("v",{backgroundColor:"#07080c"});
const lab=(t,c)=>({fontSize:12,fontColor:c,backgroundColor:"#000",backgroundOpacity:0.65,showBackground:true});
fetch("https://files.rcsb.org/download/"+C.pdb+".pdb").then(r=>{if(!r.ok)throw 0;return r.text()}).then(t=>{
v.addModel(t,"pdb");v.setStyle({},{});let c={opacity:1};
if(C.mode==="spectrum")c.color="spectrum";else if(C.mode==="grey")c.color="#4b5563";else c.colorfunc=a=>C.colors[a.resi]||"#374151";
v.setStyle({chain:"A"},{cartoon:c});
if(C.surface)v.addSurface($3Dmol.SurfaceType.MS,{opacity:0.18,color:"#9ca3af"},{chain:"A"});
C.triad.forEach(x=>{v.addStyle({chain:"A",resi:x[0]},{stick:{color:"#22d3ee",radius:0.22}});
v.addLabel(x[1],lab(0,"#22d3ee"),{chain:"A",resi:x[0],atom:"CA"})});
C.muts.forEach(x=>{v.addStyle({chain:"A",resi:x[0]},{stick:{color:"#ffb020",radius:0.3},sphere:{color:"#ffb020",radius:0.55,opacity:0.45}});
v.addLabel(x[1],lab(0,"#ffb020"),{chain:"A",resi:x[0],atom:"CA"})});
v.zoomTo({chain:"A"});if(C.spin)v.spin(true);v.render();
}).catch(()=>{document.getElementById("v").innerHTML="<p style='color:#fb7185;padding:20px'>Could not load the structure from RCSB.</p>"});
</script>"""

def viewer(muts, mode, surface, triad, spin, colors, h=560):
    cfg = json.dumps({"pdb": PDB_ID, "mode": mode, "surface": surface, "spin": spin, "colors": colors,
                      "triad": [[p, f"{a}{p}"] for p, a in triad.items()],
                      "muts": [[int(m[1:-1]), m] for m in muts]})
    components.html(VIEW_JS.replace("__CFG__", cfg).replace("__H__", str(h)), height=h+10)

# ---------------------------------------------------------------- page
st.markdown("""<div class="hero"><div class="kick">Enzyme engineering for plastic recycling</div>
<h1>Thermo<span>PET</span></h1>
<p>Predict the melting temperature of IsPETase mutants from sequence, scan every substitution
at a position, and see where each mutation sits in the 3D structure.</p>
<span class="chip">ESM-2 150M embeddings</span><span class="chip">XGBoost regressor</span>
<span class="chip">640 features per sequence</span><span class="chip">Tm in °C</span></div>""",
            unsafe_allow_html=True)

try:
    WT, SRC = get_wt()
except Exception as e:
    st.error(f"Could not load the wild type sequence. Add wt_sequence.txt to the repo. Details: {e}")
    st.stop()
bad = [f"{p} should be {a}" for p, a in {121: "S", 186: "D", 280: "R"}.items() if len(WT) < p or WT[p-1] != a]
if bad:
    st.warning("Wild type does not match UniProt numbering (" + "; ".join(bad) + "). Check wt_sequence.txt.")
TR = {p: a for p, a in TRIAD.items() if len(WT) >= p and WT[p-1] == a}

with st.sidebar:
    st.subheader("Wild type")
    st.write(f"Source: {SRC}\n\nLength: {len(WT)} residues")
    st.caption("Mutation numbering follows the full UniProt sequence, for example S121E.")
    with st.expander("Show sequence"): st.code(WT, language=None)
    st.divider(); st.caption("Model: ESM-2 150M + XGBoost (version 1)")

for k, v in {"hist": [], "last": None}.items():
    st.session_state.setdefault(k, v)
t_pred, t_3d, t_scan, t_batch, t_about = st.tabs(["Predict", "3D structure", "Scanner", "Batch", "About"])

with t_pred:
    mode = st.radio("Input type", ["Mutations on wild type", "Full sequence"], horizontal=True)
    seq, label, muts = None, "", []
    try:
        if mode == "Mutations on wild type":
            c1, c2 = st.columns([1, 2])
            pre = c1.selectbox("Example", list(PRESETS), index=1)
            txt = c2.text_input("Mutations (separate with / or commas)", PRESETS[pre], key="m_" + pre)
            seq, muts = apply_muts(WT, txt)
        else:
            seq = clean(st.text_area("Protein sequence (plain text or FASTA)", WT, height=140))
            muts = diff(WT, seq) or []
            if len(seq) > MAX_LEN - 2: st.warning("Long sequences are cut off before prediction.")
        label = "/".join(muts) if muts else "Wild type"
        if diff(WT, seq) is None: label = "Custom sequence"
    except ValueError as e:
        st.error(str(e))
    if st.button("Predict melting temperature", type="primary", disabled=seq is None):
        with st.spinner("Predicting..."):
            wt_tm, tm = predict_many([WT, seq])
            refs = predict_many([apply_muts(WT, v)[0] for v in PRESETS.values()])
        st.session_state.last = dict(label=label, muts=muts, seq=seq, tm=tm, wt=wt_tm, refs=list(refs))
        st.session_state.hist.append({"variant": label, "tm_C": round(tm, 2), "wt_C": round(wt_tm, 2),
                                      "delta_C": round(tm-wt_tm, 2)})
    L = st.session_state.last
    if L:
        d = L["tm"] - L["wt"]
        g, c = st.columns([1.2, 1])
        g.plotly_chart(gauge(L["tm"], L["wt"]), use_container_width=True)
        c.metric("Predicted Tm", f"{L['tm']:.2f} °C", f"{d:+.2f} °C vs wild type")
        c.metric("Wild type (same model)", f"{L['wt']:.2f} °C")
        (c.success if d > 0.5 else c.warning if d < -0.5 else c.info)(
            "More stable than wild type." if d > 0.5 else "Less stable than wild type." if d < -0.5
            else "About as stable as wild type.")
        names = ["Wild type"] + [k for k in PRESETS if k != "Wild type"] + [L["label"]]
        vals = L["refs"] + [L["tm"]]
        fig = go.Figure(go.Bar(x=names, y=vals, marker_color=["#4b5563", CYAN, CYAN, AMBER],
                               text=[f"{v:.1f}" for v in vals], textposition="outside"))
        fig.update_yaxes(range=[min(vals)-8, max(vals)+6], title="Predicted Tm (°C)")
        st.plotly_chart(dark(fig, 300), use_container_width=True)
        st.markdown("**Sequence** (mutations highlighted)")
        st.markdown(seq_html(WT, L["seq"]), unsafe_allow_html=True)
    if st.session_state.hist:
        with st.expander("Session history"):
            h = pd.DataFrame(st.session_state.hist)
            st.dataframe(h, use_container_width=True, hide_index=True); dl(h, "thermopet_session.csv")

with t_3d:
    L = st.session_state.last
    default = "/".join(L["muts"]) if L and L["muts"] else PRESETS["ThermoPETase"]
    c1, c2 = st.columns([2, 1])
    txt = c1.text_input("Mutations to highlight", default, key="m3d_" + default)
    cm = c2.selectbox("Colour the protein by", ["Spectrum", "Scanner effect", "Plain grey"])
    o1, o2, o3 = st.columns(3)
    surf, tri, spin = o1.toggle("Surface"), o2.toggle("Catalytic triad", True), o3.toggle("Spin")
    try:
        _, ms = apply_muts(WT, txt)
    except ValueError as e:
        st.error(str(e)); ms = []
    colors = {}
    sc = st.session_state.get("scan")
    if cm == "Scanner effect":
        if sc is None: st.info("Run the Scanner first, then residues are coloured by mean predicted effect.")
        else:
            g = sc.groupby("position")["delta_vs_wt_C"].mean(); mx = g.abs().max()
            colors = {int(p): dcolor(v, mx) for p, v in g.items()}
    viewer(ms, {"Spectrum": "spectrum", "Plain grey": "grey"}.get(cm, "scan"), surf,
           TR if tri else {}, spin, colors)
    st.markdown(f'<div class="leg"><span><i class="dot" style="background:{AMBER}"></i>Mutated residues</span>'
                f'<span><i class="dot" style="background:{CYAN}"></i>Catalytic triad</span>'
                f'<span><i class="dot" style="background:#3b82f6"></i>Scan: destabilising</span>'
                f'<span><i class="dot" style="background:{AMBER}"></i>Scan: stabilising</span></div>',
                unsafe_allow_html=True)
    st.caption(f"Structure: PDB {PDB_ID}, wild type IsPETase, shown from the RCSB server. Drag to rotate, scroll to zoom.")
    if ms:
        st.dataframe(pd.DataFrame([{"mutation": m, "position": int(m[1:-1]), "from": m[0], "to": m[-1]} for m in ms]),
                     use_container_width=True, hide_index=True)

with t_scan:
    st.write("Predict every single substitution across a range of positions.")
    c1, c2 = st.columns(2)
    a, b = int(c1.number_input("First position", 1, len(WT), 118)), int(c2.number_input("Last position", 1, len(WT), 124))
    if b < a or b - a + 1 > MAX_SCAN:
        st.error(f"Choose a range of 1 to {MAX_SCAN} positions.")
    else:
        st.caption(f"{(b-a+1)*19} variants. This can take a few minutes on a CPU.")
        if st.button("Run scan", type="primary"):
            vs = [(p, WT[p-1], x) for p in range(a, b+1) for x in AA if x != WT[p-1]]
            bar = st.progress(0.0); w = predict_many([WT])[0]
            pr = predict_many([WT[:p-1] + x + WT[p:] for p, _, x in vs], bar); bar.empty()
            st.session_state.scan = pd.DataFrame({"position": [v[0] for v in vs],
                "mutation": [f"{o}{p}{x}" for p, o, x in vs], "substitution": [v[2] for v in vs],
                "predicted_tm_C": pr.round(2), "delta_vs_wt_C": (pr-w).round(2)})
    sc = st.session_state.get("scan")
    if sc is not None:
        z = sc.pivot(index="substitution", columns="position", values="delta_vs_wt_C").reindex(list(AA))
        mx = float(np.nanmax(np.abs(z.values)))
        fig = go.Figure(go.Heatmap(z=z.values, x=[str(c) for c in z.columns], y=list(z.index), zmin=-mx, zmax=mx,
            colorscale=[[0, "#3b82f6"], [0.5, "#1f2937"], [1, AMBER]], colorbar={"title": "ΔTm °C"}))
        fig.update_xaxes(type="category", title="Position"); fig.update_yaxes(autorange="reversed")
        st.plotly_chart(dark(fig, 480), use_container_width=True)
        top = sc.sort_values("delta_vs_wt_C", ascending=False)
        st.subheader("Top 10 predicted stabilising substitutions")
        st.dataframe(top.head(10)[["mutation", "predicted_tm_C", "delta_vs_wt_C"]], use_container_width=True, hide_index=True)
        dl(sc, "thermopet_scan.csv", "Download full scan")
        st.subheader("Stack the best substitutions")
        st.caption("Combines the best substitution at up to 5 different positions and predicts each combined variant. "
                   "Combinations are extrapolation, so treat them as ideas to test.")
        if st.button("Predict stacked variants"):
            best = top.drop_duplicates("position").head(5)["mutation"].tolist()
            sets = ["/".join(best[:n]) for n in range(1, len(best)+1)]
            pr = predict_many([apply_muts(WT, s)[0] for s in sets])
            st.dataframe(pd.DataFrame({"variant": sets, "predicted_tm_C": pr.round(2)}),
                         use_container_width=True, hide_index=True)

with t_batch:
    st.write("Upload a CSV with a column named mutations or sequence (optional name column), or paste one variant per line.")
    up = st.file_uploader("CSV file", type=["csv"])
    pasted = st.text_area("Or paste variants", height=110, placeholder="S121E/D186H\nS121E/D186H/R280A")
    rows = []
    if up is not None:
        try:
            df = pd.read_csv(up); cols = {c.lower().strip(): c for c in df.columns}
            key = cols.get("mutations") or cols.get("sequence")
            if key is None: st.error("The CSV needs a mutations or sequence column.")
            else:
                for i, v in enumerate(df[key].fillna("").astype(str)):
                    rows.append((str(df[cols["name"]].iloc[i]) if "name" in cols else None, v, key.lower() == "sequence"))
        except Exception as e:
            st.error(f"Could not read the file: {e}")
    else:
        rows = [(None, l.strip(), len(l.strip()) > 40) for l in pasted.splitlines() if l.strip()]
    if rows and st.button("Run batch prediction", type="primary"):
        ok, err = [], []
        for n, v, is_seq in rows[:MAX_ROWS]:
            try:
                s = clean(v) if is_seq else apply_muts(WT, v)[0]
                d = diff(WT, s)
                ok.append((n or ("/".join(d) if d else "Wild type") if d is not None else (n or "Custom sequence"), s))
            except ValueError as e:
                err.append({"input": v[:60], "problem": str(e)})
        if ok:
            bar = st.progress(0.0); w = predict_many([WT])[0]
            pr = predict_many([s for _, s in ok], bar); bar.empty()
            out = pd.DataFrame({"variant": [n for n, _ in ok], "predicted_tm_C": pr.round(2),
                                "delta_vs_wt_C": (pr-w).round(2)}).sort_values("predicted_tm_C", ascending=False)
            st.write(f"Wild type from the same model: {w:.2f} °C")
            st.dataframe(out, use_container_width=True, hide_index=True); dl(out, "thermopet_batch.csv")
        if err:
            st.warning(f"{len(err)} row(s) could not be used."); st.dataframe(pd.DataFrame(err), hide_index=True)

with t_about:
    st.subheader("How it works")
    st.write("The sequence goes through the ESM-2 150M protein language model. "
             f"The {'first token (CLS)' if POOLING == 'cls' else 'mean residue'} embedding (640 numbers) "
             "is given to an XGBoost regressor that predicts Tm in °C. The wild type value comes from the same model, "
             "so the change shown for a variant compares like with like.")
    st.subheader("Limits")
    st.write("Predictions are estimates from a model trained on a limited set of PETase variants. They are most "
             "reliable for a few point mutations near the training data. Use them to rank candidates before lab work.")
    st.write("Add your test metrics (MAE, R squared), training set size and data sources here.")
    st.caption("Code: github.com/bigthinker-code/ThermoPet")
