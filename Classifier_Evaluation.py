"""
Version B — 3-class Stage 2 with a "normal" escape hatch.

Current Stage 2 is BINARY (HO vs cong): a normal window that leaks through the
gate is FORCED to become a wrong event label. Version B makes Stage 2 3-class
(normal / HO / congestion), so it can send gate-leaked normals BACK to normal,
potentially reducing false alarms.

KEY DESIGN POINT (this is what makes it a fair test): Stage 2 must be TRAINED on
gate-passed windows INCLUDING the leaked normals, so it learns what a leaked
normal looks like. Training only on true events (as binary Stage 2 does) would
give it no "normal" examples to learn from. So:
  - Stage 2 TRAIN set = windows in the train split whose event_prob > gate
    (true events that pass + normals that leak), labelled with their TRUE class.
  - Stage 2 TEST set  = windows in the test split whose event_prob > gate.
  - Windows below the gate are labelled Normal (unchanged).

Compares macro-F1 and the full per-class table against the current binary
Stage-2 pipeline (macro-F1 0.616).

HONEST EXPECTATION: ~wash (0.61-0.63). Stage 2 reads the same embedding the gate
used, so it has little NEW information to reject leaked normals — it will reject
some but also lose some true events (same fixed PR-AUC curve). Non-zero chance
it helps slightly because XGBoost + scalars carves the boundary differently
than the gate's linear event head.

Read-only. ~3 min.
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import (precision_recall_fscore_support, confusion_matrix,
                             f1_score, balanced_accuracy_score)
from sklearn.preprocessing import RobustScaler
import xgboost as xgb

# ======================= CONFIG =======================
ON_COLAB = Path("/content").exists()
DS_ROOT = Path("/content/drive/MyDrive/Combined_LENS") if ON_COLAB \
    else Path(r"D:\LENS_Dataset\combined\rtt_forecast_ds")
OUT = DS_ROOT / "step8_results"; OUT.mkdir(exist_ok=True, parents=True)
FP_CEILING = 0.05
SEED = 42
CLASS_NAMES = ["Normal", "Handover", "Congestion"]
# =====================================================

idx = pd.read_csv(DS_ROOT / "index_split_v9_6.csv")
emb = pd.read_csv(DS_ROOT / "embeddings.csv")
emb_cols = [c for c in emb.columns if c.startswith("emb_")]
s1 = pd.read_csv(DS_ROOT / "index_stage2.csv")[["sample", "event_prob"]]
SCALARS = [c for c in idx.columns if c in [
    "elevation_mean","elevation_slope","rtt_input_mean","rtt_input_std",
    "rtt_last5s_slope","sinr_input_mean","rtt_slope_full25s","rtt_monotonicity",
    "rtt_range_full25s","sinr_range_full25s"]]
d = idx.merge(emb, on="sample", how="inner").merge(s1, on="sample", how="left")
FEATS = emb_cols + SCALARS

# --- gate threshold: same rule as the reported system (5% FP on val) ---
va = d[d.split=="val"]; ecv=va.event_class.values; epv=va.event_prob.values
isn,isc=(ecv==0),(ecv==2); best=None
for t in np.linspace(0.01,0.99,197):
    fp=((epv>t)&isn).mean(); cr=((epv>t)&isc).sum()/max(isc.sum(),1)
    if fp<=FP_CEILING and (best is None or cr>best[1]): best=(t,cr)
GATE=best[0]
print(f"gate threshold (5% FP): {GATE:.3f}")

def prep(df, sc=None, md=None):
    X=df[FEATS].values.astype(np.float32).copy()
    if md is None: md=np.nanmedian(X,0); md=np.where(np.isnan(md),0,md)
    for j in range(X.shape[1]): X[np.isnan(X[:,j]),j]=md[j]
    if sc is None: sc=RobustScaler(); X=sc.fit_transform(X)
    else: X=sc.transform(X)
    return X,sc,md

def per_class(y_true, y_pred, title):
    print(f"\n=== {title} ===")
    cm=confusion_matrix(y_true,y_pred,labels=[0,1,2])
    print("                 Pred:  Norm    HO  Cong")
    for i,nm in enumerate(CLASS_NAMES):
        print(f"    True {nm:<10} {cm[i,0]:6d} {cm[i,1]:5d} {cm[i,2]:5d}")
    P,R,F,S=precision_recall_fscore_support(y_true,y_pred,labels=[0,1,2],zero_division=0)
    print(f"    {'class':<12}{'P':>7}{'R':>7}{'F1':>7}")
    for i,nm in enumerate(CLASS_NAMES):
        print(f"    {nm:<12}{P[i]:>7.3f}{R[i]:>7.3f}{F[i]:>7.3f}")
    macro=f1_score(y_true,y_pred,labels=[0,1,2],average='macro',zero_division=0)
    bacc=balanced_accuracy_score(y_true,y_pred)
    print(f"    macro-F1={macro:.3f}  balanced-acc={bacc:.3f}")
    return dict(macro_f1=float(macro), per_class={CLASS_NAMES[i]:dict(
        precision=float(P[i]),recall=float(R[i]),f1=float(F[i])) for i in range(3)})

# ============ BINARY baseline (current system) ============
ev = d[d.event_class.isin([1,2])].copy(); ev["lab"]=(ev.event_class==1).astype(int)
tr_b=ev[ev.split=="train"]; va_b=ev[ev.split=="val"]
Xtr,sc,md=prep(tr_b); Xva,_,_=prep(va_b,sc,md)
nH=int(tr_b.lab.sum()); wb=np.where(tr_b.lab==0,nH/max(len(tr_b)-nH,1),1.0)
clf_b=xgb.XGBClassifier(n_estimators=400,max_depth=5,learning_rate=0.05,subsample=0.8,
    colsample_bytree=0.3,min_child_weight=3,reg_alpha=0.5,reg_lambda=2.0,
    eval_metric="logloss",random_state=SEED)
try:
    clf_b.set_params(early_stopping_rounds=30)
    clf_b.fit(Xtr,tr_b.lab.values,sample_weight=wb,eval_set=[(Xva,va_b.lab.values)],verbose=False)
except TypeError:
    clf_b.fit(Xtr,tr_b.lab.values,sample_weight=wb,eval_set=[(Xva,va_b.lab.values)],verbose=False)
te=d[d.split=="test"].copy(); te["pred"]=0
g=te.event_prob.values>GATE
if g.sum()>0:
    Xg,_,_=prep(te[g],sc,md); p=clf_b.predict(Xg); te.loc[g,"pred"]=np.where(p==1,1,2)
r_binary=per_class(te.event_class.values, te["pred"].values, "BINARY Stage 2 (current system)")

# ============ VERSION B: 3-class Stage 2 ============
# Train on train-split windows that PASS the gate, with TRUE 3-class labels
# (true events that pass + normals that leak). This teaches Stage 2 to recognise
# leaked normals and send them back.
tr_all=d[d.split=="train"].copy()
tr_pass=tr_all[tr_all.event_prob.values>GATE].copy()
print(f"\nStage-2 (3-class) train set = gate-passed train windows: {len(tr_pass)}")
print(f"  of which normal={int((tr_pass.event_class==0).sum())} "
      f"HO={int((tr_pass.event_class==1).sum())} "
      f"cong={int((tr_pass.event_class==2).sum())}")
va_all=d[d.split=="val"].copy(); va_pass=va_all[va_all.event_prob.values>GATE].copy()

Xtr3,sc3,md3=prep(tr_pass)
ytr3=tr_pass.event_class.values
# class weights inverse-freq over the 3 classes present
cnt=np.bincount(ytr3,minlength=3).astype(float)
inv=cnt.sum()/np.where(cnt>0,cnt,1); w3=inv[ytr3]
Xva3,_,_=prep(va_pass,sc3,md3)
clf3=xgb.XGBClassifier(n_estimators=400,max_depth=5,learning_rate=0.05,subsample=0.8,
    colsample_bytree=0.3,min_child_weight=3,reg_alpha=0.5,reg_lambda=2.0,
    objective="multi:softprob",num_class=3,eval_metric="mlogloss",random_state=SEED)
try:
    clf3.set_params(early_stopping_rounds=30)
    clf3.fit(Xtr3,ytr3,sample_weight=w3,eval_set=[(Xva3,va_pass.event_class.values)],verbose=False)
except TypeError:
    clf3.fit(Xtr3,ytr3,sample_weight=w3,eval_set=[(Xva3,va_pass.event_class.values)],verbose=False)

te2=d[d.split=="test"].copy(); te2["pred"]=0            # below-gate -> Normal
g2=te2.event_prob.values>GATE
if g2.sum()>0:
    Xg2,_,_=prep(te2[g2],sc3,md3)
    te2.loc[g2,"pred"]=clf3.predict(Xg2)               # 3-class: can output 0/1/2
r_vb=per_class(te2.event_class.values, te2["pred"].values,
               "VERSION B: 3-class Stage 2 (normal escape hatch)")

# ============ verdict ============
print("\n"+"="*60); print("VERDICT"); print("="*60)
print(f"  BINARY  (current) macro-F1 = {r_binary['macro_f1']:.3f}")
print(f"  VERSION B (3-class) macro-F1 = {r_vb['macro_f1']:.3f}")
gain=r_vb['macro_f1']-r_binary['macro_f1']
print(f"  change = {gain:+.3f}")
# did it actually reduce false-normal->event alarms?
for cls in ["Handover","Congestion"]:
    b=r_binary['per_class'][cls]; v=r_vb['per_class'][cls]
    print(f"  {cls}: F1 {b['f1']:.3f} -> {v['f1']:.3f}  "
          f"(precision {b['precision']:.3f} -> {v['precision']:.3f})")
if gain>0.02:
    print("  -> the normal escape hatch HELPED. Real gain.")
elif gain<-0.02:
    print("  -> it HURT (rejected true events along with leaked normals).")
else:
    print("  -> WASH. Stage 2 has ~no new info to reject leaked normals that the")
    print("     gate didn't already use (same embedding, same ~0.19 separability).")

json.dump({"binary":r_binary,"version_b":r_vb,"gate":float(GATE)},
          open(OUT/"version_b_3class_stage2.json","w"),indent=2,default=str)
print(f"\nsaved {OUT/'version_b_3class_stage2.json'}")

# ============ PER-EVENT (3-class) ============
te2["event_id"] = idx.loc[te2.index, "event_id"].values

ev_mask = te2.event_class.isin([1, 2])
ev_df = te2[ev_mask].copy()
g = ev_df.groupby("event_id")

ev_true = g["event_class"].first()          # 1 or 2
ev_pred = g["pred"].apply(lambda s: 2 if (s == 2).any() else (1 if (s == 1).any() else 0))

# congestion only
cong_true = (ev_true == 2)
cong_pred = (ev_pred == 2)

tp = (cong_true & cong_pred).sum()
fp = (~cong_true & cong_pred).sum()
fn = (cong_true & ~cong_pred).sum()
P = tp / max(tp + fp, 1)
R = tp / max(tp + fn, 1)
F1 = 2 * P * R / max(P + R, 1e-9)

print(f"\n=== PER-EVENT congestion (3-class Stage 2) ===")
print(f"  events: {len(ev_true):,}  (cong={int(cong_true.sum())}, HO={int((ev_true==1).sum())})")
print(f"  TP={tp}  FP={fp}  FN={fn}")
print(f"  P={P:.3f}  R={R:.3f}  F1={F1:.3f}")

# ============ PER-EVENT (all classes) ============
te2["event_id"] = idx.loc[te2.index, "event_id"].values

ev_mask = te2.event_class.isin([1, 2])
ev_df = te2[ev_mask].copy()
g = ev_df.groupby("event_id")

ev_true = g["event_class"].first()
ev_pred = g["pred"].apply(lambda s: 2 if (s == 2).any() else (1 if (s == 1).any() else 0))

print(f"\n=== PER-EVENT (3-class Stage 2) ===")
print(f"  total events: {len(ev_true):,}  "
      f"(HO={int((ev_true==1).sum())}, cong={int((ev_true==2).sum())})")

for cls, name in [(1, "Handover"), (2, "Congestion")]:
    true_c = (ev_true == cls)
    pred_c = (ev_pred == cls)
    tp = (true_c & pred_c).sum()
    fp = (~true_c & pred_c).sum()
    fn = (true_c & ~pred_c).sum()
    P = tp / max(tp + fp, 1)
    R = tp / max(tp + fn, 1)
    F1 = 2 * P * R / max(P + R, 1e-9)
    print(f"  {name:<12}  P={P:.3f}  R={R:.3f}  F1={F1:.3f}  (TP={tp} FP={fp} FN={fn})")

    # ============ PER-EVENT full table (3-class Stage 2) ============
te2["event_id"] = idx.loc[te2.index, "event_id"].values

# --- events: group by event_id ---
ev_df = te2[te2.event_class.isin([1, 2])].copy()
g = ev_df.groupby("event_id")
ev_true = g["event_class"].first().values
ev_pred = g["pred"].apply(lambda s: 2 if (s == 2).any() else (1 if (s == 1).any() else 0)).values

# --- normals: each window is its own "event" ---
norm_df = te2[te2.event_class == 0]
n_true = norm_df["event_class"].values
n_pred = norm_df["pred"].values

# --- combine ---
all_true = np.concatenate([n_true, ev_true])
all_pred = np.concatenate([n_pred, ev_pred])

cm = confusion_matrix(all_true, all_pred, labels=[0, 1, 2])
P, R, F, _ = precision_recall_fscore_support(all_true, all_pred, labels=[0, 1, 2], zero_division=0)
macro = f1_score(all_true, all_pred, labels=[0, 1, 2], average='macro', zero_division=0)
bacc = balanced_accuracy_score(all_true, all_pred)

print(f"\n=== PER-EVENT full pipeline (3-class Stage 2) ===")
print(f"                 Pred:  Norm    HO  Cong")
for i, nm in enumerate(["Normal", "Handover", "Congestion"]):
    print(f"    True {nm:<10} {cm[i,0]:6d} {cm[i,1]:5d} {cm[i,2]:5d}")
print(f"    {'class':<12}{'P':>7}{'R':>7}{'F1':>7}")
for i, nm in enumerate(["Normal", "Handover", "Congestion"]):
    print(f"    {nm:<12}{P[i]:>7.3f}{R[i]:>7.3f}{F[i]:>7.3f}")
print(f"    macro-F1={macro:.3f}  balanced-acc={bacc:.3f}")

# ============ PER-EVENT full table (3-class Stage 2) ============
te2["event_id"] = idx.loc[te2.index, "event_id"].values

# --- events: group by event_id ---
ev_df = te2[te2.event_class.isin([1, 2])].copy()
g = ev_df.groupby("event_id")
ev_true = g["event_class"].first().values
ev_pred = g["pred"].apply(lambda s: 2 if (s == 2).any() else (1 if (s == 1).any() else 0)).values

# --- normals: each window is its own "event" ---
norm_df = te2[te2.event_class == 0]
n_true = norm_df["event_class"].values
n_pred = norm_df["pred"].values

# --- combine ---
all_true = np.concatenate([n_true, ev_true])
all_pred = np.concatenate([n_pred, ev_pred])

cm = confusion_matrix(all_true, all_pred, labels=[0, 1, 2])
P, R, F, _ = precision_recall_fscore_support(all_true, all_pred, labels=[0, 1, 2], zero_division=0)
macro = f1_score(all_true, all_pred, labels=[0, 1, 2], average='macro', zero_division=0)
bacc = balanced_accuracy_score(all_true, all_pred)

print(f"\n=== PER-EVENT full pipeline (3-class Stage 2) ===")
print(f"                 Pred:  Norm    HO  Cong")
for i, nm in enumerate(["Normal", "Handover", "Congestion"]):
    print(f"    True {nm:<10} {cm[i,0]:6d} {cm[i,1]:5d} {cm[i,2]:5d}")
print(f"    {'class':<12}{'P':>7}{'R':>7}{'F1':>7}")
for i, nm in enumerate(["Normal", "Handover", "Congestion"]):
    print(f"    {nm:<12}{P[i]:>7.3f}{R[i]:>7.3f}{F[i]:>7.3f}")
print(f"    macro-F1={macro:.3f}  balanced-acc={bacc:.3f}")

"""
Visualize false positives: normal windows predicted as congestion or handover.
Uses the 3-class Stage 2 pipeline predictions.

Run after version_b_3class_stage2.py (needs te2 DataFrame in memory),
OR run standalone by re-computing predictions.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json

# ── PATHS ─────────────────────────────────────────────────────────────────────
DS_ROOT    = Path(r'D:\LENS_Dataset\combined\rtt_forecast_ds')
SAMPLE_DIR = DS_ROOT / 'samples'
INDEX_FILE = DS_ROOT / 'index_split_v9_6.csv'

DT_MS     = 10
STEPS_IN  = 1500
STEPS_OUT = 500
BASE_COL_H = 13
DEV_COL_H  = 14

N_EXAMPLES = 8


def load_rtt_curve(sample_name):
    d = np.load(SAMPLE_DIR / sample_name, allow_pickle=False)
    with open(DS_ROOT / 'feature_schema.json') as f:
        schema = json.load(f)
    select_idx = schema['x_num_select_indices']
    num_features = schema['num_features']
    rtt_col = num_features.index('rtt_ms')
    raw_rtt_col = select_idx[rtt_col]
    input_rtt_ms = d['X_num'][:, raw_rtt_col].copy()

    y_rtt = d['y_rtt']
    mask  = d['rtt_mask']
    horizon_rtt_ms = np.full(STEPS_OUT, np.nan)
    horizon_rtt_ms[mask == 1] = np.expm1(y_rtt[mask == 1])

    baseline  = d['y_aim_num'][:, BASE_COL_H]
    deviation = d['y_aim_num'][:, DEV_COL_H]

    t_input   = np.arange(STEPS_IN) * DT_MS / 1000.0
    t_horizon = STEPS_IN * DT_MS / 1000.0 + np.arange(STEPS_OUT) * DT_MS / 1000.0

    return dict(input_rtt_ms=input_rtt_ms, horizon_rtt_ms=horizon_rtt_ms,
                baseline=baseline, deviation=deviation,
                t_input=t_input, t_horizon=t_horizon)

groups = [
    (0, 1, "Normal → HO", 1377),
    (0, 2, "Normal → Cong", 1589),
    (1, 0, "Handover → Normal", 1761),
    (2, 0, "Congestion → Normal", 983),
]

for true_c, pred_c, name, expected in groups:
    g = te2[(te2.event_class == true_c) & (te2.pred == pred_c)]
    print(f"\n=== {name} (n={len(g):,}) ===")
    print(f"  event_prob   median={g.event_prob.median():.3f}  mean={g.event_prob.mean():.3f}")
    if 'dev_p90' in g.columns:
        print(f"  dev_p90      median={g.dev_p90.median():.1f}")
    if 'dev_max' in g.columns:
        print(f"  dev_max      median={g.dev_max.median():.1f}")
    if 'cong_frac_40' in g.columns:
        print(f"  cong_frac_40 median={g.cong_frac_40.median():.3f}")
    if 'longest_run_bins' in g.columns:
        print(f"  longest_run  median={g.longest_run_bins.median():.0f}")
    if 'baseline_med' in g.columns:
        print(f"  baseline_med median={g.baseline_med.median():.1f}")
    if 'input_has_recent_ho' in g.columns:
        print(f"  recent_ho    sum={g.input_has_recent_ho.sum()}")
    if 'horizon_handover' in g.columns:
        print(f"  horizon_ho   sum={g.horizon_handover.sum()}")


