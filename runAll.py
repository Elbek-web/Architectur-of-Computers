import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

FILE_2017 = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX (1).csv"
DROP = ["Destination Port", "Fwd Header Length.1"]
SEEDS = [1, 2, 3, 4, 5]
ATTACK_FRAC = 0.01          # 2019 attack files to be sampled
OUT = Path("results")
OUT.mkdir(exist_ok=True)


def make_models(seed):
    return {
        "LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "DecisionTree": DecisionTreeClassifier(random_state=seed),
        "RandomForest": RandomForestClassifier(n_estimators=50, n_jobs=1, random_state=seed),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    }


def is_attack(label_series):
    return (label_series.astype(str).str.strip().str.upper() != "BENIGN").astype(int)


def clean(df):
    df.columns = df.columns.str.strip()
    df = df.drop(columns=DROP, errors="ignore")
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df.drop_duplicates()


def find_file(prefix):
    hits = sorted(Path(".").glob(prefix + "*"))
    if not hits:
        raise FileNotFoundError(f"'{prefix}' started file not found (check the folder)")
    return hits[0]


# ----------------------------------------------------------------- IDS2017
df17 = clean(pd.read_csv(FILE_2017))
y17 = is_attack(df17["Label"])
feats = [c for c in df17.columns if c != "Label"]
X17 = df17[feats]
print(f"[IDS2017] cleaned size: {X17.shape} | attack ratio: {y17.mean():.3f}")

# --------------------------------------------------------- A) base: IDS2017
rows = []
for seed in SEEDS:
    X_tr, X_te, y_tr, y_te = train_test_split(
        X17, y17, test_size=0.3, stratify=y17, random_state=seed)
    for name, m in make_models(seed).items():
        t0 = time.perf_counter(); m.fit(X_tr, y_tr); t_train = time.perf_counter() - t0
        t0 = time.perf_counter(); pred = m.predict(X_te); t_pred = time.perf_counter() - t0
        tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
        rows.append({"model": name, "seed": seed,
                     "F1": f1_score(y_te, pred), "FPR": fp / (fp + tn),
                     "train_s": t_train,
                     "pred_ms_per_1k": t_pred / len(X_te) * 1e6,
                     "size_MB": len(pickle.dumps(m)) / 1e6})
    print(f"[A] base results, seed {seed} ready")

runs = pd.DataFrame(rows)
runs.to_csv(OUT / "A_baseline_runs.csv", index=False)
summ = runs.drop(columns="seed").groupby("model").agg(["mean", "std"])
summ.columns = ["_".join(c) for c in summ.columns]
summ.to_csv(OUT / "A_baseline_summary.csv")
print("\n[A] IDS2017 base results (mean and std, 5-fold cross-validation):")
print(summ.round(4).T.to_string(), "\n")

# ------------------------------------------------------------ 2019 upload
paths = {k: find_file(k) for k in ["Syn", "DrDoS_NTP", "DrDoS_LDAP", "UDPLag"]}
cols_ok = None
for p in paths.values():
    cols = set(pd.read_csv(p, nrows=0).columns.str.strip())
    cols_ok = cols if cols_ok is None else cols_ok & cols
common = [c for c in feats if c in cols_ok]
missing = [c for c in feats if c not in common]
print(f"[2019] general features: {len(common)} / {len(feats)} | missing: {missing}")
keep = set(common) | {"Label"}


def load_2019(path, seed=42):
    ben, att = [], []
    for chunk in pd.read_csv(path, usecols=lambda c: c.strip() in keep, chunksize=200_000):
        chunk.columns = chunk.columns.str.strip()
        chunk = chunk.replace([np.inf, -np.inf], np.nan).dropna()
        a = is_attack(chunk["Label"]) == 1
        ben.append(chunk[~a][common])
        att.append(chunk[a][common].sample(frac=ATTACK_FRAC, random_state=seed))
    return pd.concat(ben), pd.concat(att)


ben_s, att_s = load_2019(paths["Syn"])
ben_n, att_n = load_2019(paths["DrDoS_NTP"])
ben_l, _ = load_2019(paths["DrDoS_LDAP"])     # LDAP va UDPLag: only benign traffic, no attacks
ben_u, _ = load_2019(paths["UDPLag"])
X_ben = pd.concat([ben_s, ben_n, ben_l, ben_u]).drop_duplicates()
X_syn, X_ntp = att_s.drop_duplicates(), att_n.drop_duplicates()
print(f"[2019] test: benign {len(X_ben)} | Syn {len(X_syn)} | NTP {len(X_ntp)}\n")

# ------------------------------------------- B) Test within dataset (2017->2019)
X_tr = X17[common]
rows = []
for name, m in make_models(1).items():
    m.fit(X_tr, y17)
    p_ben, p_syn, p_ntp = m.predict(X_ben), m.predict(X_syn), m.predict(X_ntp)
    rows.append({"model": name, "FPR": p_ben.mean(),
                 "recall_Syn": p_syn.mean(), "recall_NTP": p_ntp.mean()})
    print(f"[B] {name} ready")
resB = pd.DataFrame(rows).round(4)
resB.to_csv(OUT / "B_cross_2017_to_2019.csv", index=False)
print("\n[B] Test within dataset (Read in IDS2017 -> Tested in 2019):")
print(resB.to_string(index=False), "\n")

# --------------------------------------------- C) nazorat: 2019 ning ichida
X19 = pd.concat([X_ben, X_syn, X_ntp])
tag = np.array(["benign"] * len(X_ben) + ["Syn"] * len(X_syn) + ["NTP"] * len(X_ntp))
y19 = (tag != "benign").astype(int)
Xa, Xb, ya, yb, ta, tb = train_test_split(
    X19, y19, tag, test_size=0.3, stratify=tag, random_state=1)
rows = []
for name, m in make_models(1).items():
    p = m.fit(Xa, ya).predict(Xb)
    rows.append({"model": name, "FPR": p[tb == "benign"].mean(),
                 "recall_Syn": p[tb == "Syn"].mean(), "recall_NTP": p[tb == "NTP"].mean()})
resC = pd.DataFrame(rows).round(4)
resC.to_csv(OUT / "C_within_2019.csv", index=False)
print("[C] Check (In 2019 its learned and checked):")
print(resC.to_string(index=False), "\n")

# ------------------------------------------------ D) 
chk = [c for c in ["Flow Duration", "Total Fwd Packets", "Flow Bytes/s", "Packet Length Mean"]
       if c in common]
med = pd.DataFrame({
    "IDS2017_benign_median": X17.loc[y17 == 0, chk].median(),
    "2019_benign_median": X_ben[chk].median()})
med.to_csv(OUT / "D_benign_medians.csv")
print("[D] Benign traffic medians:")
print(med.round(2).to_string(), "\n")
print("All is done. Results are in the 'results' folder.")

