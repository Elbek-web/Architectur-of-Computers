from collections import Counter
from pathlib import Path
import pandas as pd

KEYS = ["Syn", "DrDoS_NTP", "DrDoS_LDAP", "UDPLag"]
WANTED = ("Label", "Destination Port")

for key in KEYS:
    path = next(Path(".").glob(f"{key}*"), None)
    if path is None:
        print(key, "-> not found")
        continue

    lab, port_att, port_ben = Counter(), Counter(), Counter()
    for chunk in pd.read_csv(path, usecols=lambda c: c.strip() in WANTED,
                             chunksize=200_000):
        chunk.columns = chunk.columns.str.strip()
        is_ben = chunk["Label"].astype(str).str.upper() == "BENIGN"
        lab.update(chunk["Label"].value_counts().to_dict())
        port_ben.update(chunk.loc[is_ben, "Destination Port"].value_counts().to_dict())
        port_att.update(chunk.loc[~is_ben, "Destination Port"].value_counts().to_dict())

    print(path.name, "| rows:", sum(lab.values()))
    print("  Label:", dict(lab))
    print("  attack ports:", dict(port_att.most_common(3)))
    print("  benign ports:", dict(port_ben.most_common(3)))