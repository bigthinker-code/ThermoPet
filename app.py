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
