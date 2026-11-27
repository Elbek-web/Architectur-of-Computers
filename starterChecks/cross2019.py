from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

FILE_2017 = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX (1).csv"
DROP = ["Destination Port", "Fwd Header Length.1"]

def clean(df):
    df.columns = df.columns.str.strip()
    df = df.drop(columns=DROP, errors="ignore")
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df.drop_duplicates()

# 1) IDS2017: cleaning and preparing the training set
df17 = clean(pd.read_csv(FILE_2017))
y_tr = (df17["Label"].str.upper() != "BENIGN").astype(int)
feats = [c for c in df17.columns if c != "Label"]

# 2) finding common features between 2017 and 2019 datasets
def header(prefix):
    path = next(Path(".").glob(prefix + "*"))
    return path, set(pd.read_csv(path, nrows=0).columns.str.strip())

path_syn, cols_syn = header("Syn")
path_ntp, cols_ntp = header("DrDoS_NTP")
common = [c for c in feats if c in cols_syn and c in cols_ntp]
print("IDS2017 properties:", len(feats), "| total:", len(common))
print("in 2019 no columns:", [c for c in feats if c not in common])
keep = set(common) | {"Label"}

# 3) 2019: benign all, 1% example from attacks
def load_2019(path, attack_frac=0.01, seed=42):
    ben, att = [], []
    for chunk in pd.read_csv(path, usecols=lambda c: c.strip() in keep,
                             chunksize=200_000):
        chunk.columns = chunk.columns.str.strip()
        chunk = chunk.replace([np.inf, -np.inf], np.nan).dropna()
        is_ben = chunk["Label"].astype(str).str.upper() == "BENIGN"
        ben.append(chunk[is_ben])
        att.append(chunk[~is_ben].sample(frac=attack_frac, random_state=seed))
    return pd.concat(ben), pd.concat(att)

ben_s, att_s = load_2019(path_syn)
ben_n, att_n = load_2019(path_ntp)
benign = pd.concat([ben_s, ben_n]).drop_duplicates()
att_s, att_n = att_s.drop_duplicates(), att_n.drop_duplicates()
print("test: benign", len(benign), "| Syn", len(att_s), "| NTP", len(att_n))

X_tr = df17[common]
X_ben, X_syn, X_ntp = benign[common], att_s[common], att_n[common]
y_true = np.r_[np.zeros(len(X_ben)), np.ones(len(X_syn)), np.ones(len(X_ntp))]

# 4) models
models = {
    "LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "DecisionTree": DecisionTreeClassifier(random_state=1),
    "RandomForest": RandomForestClassifier(n_estimators=50, n_jobs=1, random_state=1),
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
}

rows = []
for name, m in models.items():
    m.fit(X_tr, y_tr)
    p_ben, p_syn, p_ntp = m.predict(X_ben), m.predict(X_syn), m.predict(X_ntp)
    y_pred = np.r_[p_ben, p_syn, p_ntp]
    rows.append({"model": name,
                 "F1_all": f1_score(y_true, y_pred),
                 "FPR": p_ben.mean(),
                 "recall_Syn": p_syn.mean(),
                 "recall_NTP": p_ntp.mean()})
    print("ready:", name)

res = pd.DataFrame(rows).round(4)
print(res.to_string(index=False))
res.to_csv("cross_2017_to_2019.csv", index=False)

# --- 1) Checking: is benign traffic similar in both datasets?
chk = ["Flow Duration", "Total Fwd Packets", "Flow Bytes/s", "Packet Length Mean"]
print("IDS2017 benign median:", df17.loc[y_tr == 0, chk].median().round(2).to_dict())
print("2019 benign median:  ", X_ben[chk].median().round(2).to_dict())

# --- 2) Control: train on 2019, test on 2019 (within the same dataset)
from sklearn.model_selection import train_test_split
from sklearn.base import clone

X19 = pd.concat([X_ben, X_syn, X_ntp])
tag = np.array(["benign"] * len(X_ben) + ["Syn"] * len(X_syn) + ["NTP"] * len(X_ntp))
Xa, Xb, ya, yb, ta, tb = train_test_split(
    X19, y_true, tag, test_size=0.3, stratify=tag, random_state=1)

rows2 = []
for name, m in models.items():
    p = clone(m).fit(Xa, ya).predict(Xb)
    rows2.append({"model": name,
                  "FPR": p[tb == "benign"].mean(),
                  "recall_Syn": p[tb == "Syn"].mean(),
                  "recall_NTP": p[tb == "NTP"].mean()})
print("Control (in 2019):")
res2 = pd.DataFrame(rows2).round(4)
print(res2.to_string(index=False))
res2.to_csv("within_2019.csv", index=False)