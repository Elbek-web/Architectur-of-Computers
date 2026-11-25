# Lightweight ML models for DDoS detection: cross-dataset test and computational cost

Code and results for the paper *"Comparing Lightweight Machine Learning Models for DDoS Detection:
Generalization and Computational Cost"* (Student Research Challenge 2026/27, Architecture of Computers,
Vistula University).

Author: [Name], student ID [ID]

## What the script does

`runAll.py` runs four experiments with Logistic Regression, Decision Tree, Random Forest (50 trees) and KNN:

| Block | Experiment | Output file |
|---|---|---|
| A | Within-dataset baseline on CIC-IDS2017 (five random stratified 70/30 splits; F1, FPR, training time, inference time, model size) | `results/A_baseline_runs.csv`, `results/A_baseline_summary.csv` |
| B | Train on CIC-IDS2017, test on CIC-DDoS2019 (FPR, recall for Syn and NTP) | `results/B_cross_2017_to_2019.csv` |
| C | Control: train and test inside CIC-DDoS2019 | `results/C_within_2019.csv` |
| D | Scale check: medians of four features in benign traffic of both datasets | `results/D_benign_medians.csv` |

## Requirements

- Python 3.14.7, scikit-learn 1.9.1, pandas 3.1.1, numpy [version]
- Install: `python -m pip install -r requirements.txt`
- Paper results were produced on an AMD Ryzen 7 260 (8 cores, 16 threads) with 16 GB RAM.

## Data (not included, public)

The datasets are provided by the Canadian Institute for Cybersecurity (UNB) after a short download form.
They are not redistributed here. Please cite Sharafaldin et al. (2018, 2019) when using them.

- CIC-IDS2017: https://www.unb.ca/cic/datasets/ids-2017.html
  -> `CSVs/MachineLearningCSV.zip` -> `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`
- CIC-DDoS2019: https://www.unb.ca/cic/datasets/ddos-2019.html
  -> `CSV-01-12.zip` -> `Syn.csv`, `DrDoS_NTP.csv`, `DrDoS_LDAP.csv`, `UDPLag.csv`

Put these five CSV files in the same folder as `runAll.py`. If your copy of the IDS2017 file has a
different name, edit `FILE_2017` at the top of the script.

## Run

```
python runAll.py
```

Takes roughly 10-25 minutes on the machine above. The 2019 files are read in chunks, so 16 GB of RAM is enough.

## Sanity checks

A correct run prints:

- `cleaned size: (221264, 76)` for IDS2017 after cleaning
- `general features: 76 / 76` shared with CIC-DDoS2019
- `test: benign 18414 | Syn 2096 | NTP 11456` for the CIC-DDoS2019 test set

## Reproducibility notes

- Random seeds: block A uses seeds 1-5, blocks B and C use seed 1, the 1% attack sample of CIC-DDoS2019 uses seed 42.
- Timings depend on the machine and on background load; ratios between models are more reliable than absolute values.
- Model size is the size of the pickled model, not runtime memory.
- Destination Port and a duplicated column (`Fwd Header Length.1`) are removed; infinite values, missing values and duplicate rows are dropped.

## Limitations

- Training attack (HTTP flood) and test attacks (SYN flood, NTP reflection) differ, so dataset shift and attack-type shift are confounded.
- Benign traffic differs in scale between the datasets (see block D).
- Blocks B and C are single runs on a 1% sample of the attack flows (Syn has many duplicate flows).
- Default hyperparameters; original CIC-IDS2017 files, not corrected versions.

## AI assistance

Claude (Anthropic) was used for code drafting and review; see the declaration in the paper.
