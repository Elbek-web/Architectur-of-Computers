import pickle, time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, confusion_matrix

FILE = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX (1).csv"

def load_clean(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df.drop(columns=["Destination Port", "Fwd Header Length.1"], errors="ignore")
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df.drop_duplicates()
    y = (df["Label"].str.upper() != "BENIGN").astype(int) 
    X = df.drop(columns=["Label"])
    return X, y

X, y = load_clean(FILE)
print("cleaned size:", X.shape, "| attack rate:", round(y.mean(), 3))

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)

models = {
    "LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42),
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
}

rows = []
for name, m in models.items():
    t0 = time.perf_counter(); m.fit(X_tr, y_tr); t_train = time.perf_counter() - t0
    t0 = time.perf_counter(); pred = m.predict(X_te); t_pred = time.perf_counter() - t0
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    rows.append({
        "model": name,
        "F1": f1_score(y_te, pred),
        "FPR": fp / (fp + tn),
        "train_s": t_train,
        "pred_ms_per_1k": t_pred / len(X_te) * 1000 * 1000,
        "size_MB": len(pickle.dumps(m)) / 1e6,
    })
    print("trained:", name)

print(pd.DataFrame(rows).round(4).to_string(index=False))

pd.DataFrame(rows).to_csv("baseline_ids2017.csv", index=False)