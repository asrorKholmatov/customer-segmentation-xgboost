"""Download and load the Kaggle "Customer Segmentation" dataset (vetrirah/customer)."""
from pathlib import Path
import subprocess
import urllib.request

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TRAIN_CSV = DATA_DIR / "Train.csv"

KAGGLE_DATASET = "vetrirah/customer"
# Public mirror of the same Kaggle files, used when the Kaggle CLI is not configured.
MIRROR_URL = "https://raw.githubusercontent.com/Atikahdr/ArchTech_DS_CustomerSegmentation/main/Train.csv"

TARGET = "Segmentation"
CLASSES = ["A", "B", "C", "D"]

PROFESSIONS = ["Artist", "Doctor", "Engineer", "Entertainment", "Executive",
               "Healthcare", "Homemaker", "Lawyer", "Marketing"]
CATEGORIES = [f"Cat_{i}" for i in range(1, 8)]
SPENDING = {"Low": 0, "Average": 1, "High": 2}
YES_NO = {"No": 0, "Yes": 1}


def download() -> Path:
    """Fetch Train.csv via the Kaggle CLI, falling back to the public mirror."""
    if TRAIN_CSV.exists():
        return TRAIN_CSV
    DATA_DIR.mkdir(exist_ok=True)
    try:
        subprocess.run(["kaggle", "datasets", "download", "-d", KAGGLE_DATASET,
                        "-p", str(DATA_DIR), "--unzip"], check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("Kaggle CLI unavailable - downloading from mirror")
        urllib.request.urlretrieve(MIRROR_URL, TRAIN_CSV)
    return TRAIN_CSV


def load() -> pd.DataFrame:
    return pd.read_csv(download())


def feature_names() -> list[str]:
    return (["Gender", "Ever_Married", "Age", "Graduated", "Work_Experience",
             "Spending_Score", "Family_Size"]
            + [f"Profession_{p}" for p in PROFESSIONS]
            + [f"Var_1_{c}" for c in CATEGORIES])


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw customer rows into the numeric matrix the model expects.

    Missing values stay NaN: XGBoost learns a default branch for them, and the
    web app mirrors this exact encoding in JavaScript.
    """
    X = pd.DataFrame(index=df.index)
    X["Gender"] = df["Gender"].map({"Female": 0, "Male": 1})
    X["Ever_Married"] = df["Ever_Married"].map(YES_NO)
    X["Age"] = df["Age"].astype(float)
    X["Graduated"] = df["Graduated"].map(YES_NO)
    X["Work_Experience"] = df["Work_Experience"].astype(float)
    X["Spending_Score"] = df["Spending_Score"].map(SPENDING)
    X["Family_Size"] = df["Family_Size"].astype(float)
    for p in PROFESSIONS:
        X[f"Profession_{p}"] = (df["Profession"] == p).astype(float).where(df["Profession"].notna())
    for c in CATEGORIES:
        X[f"Var_1_{c}"] = (df["Var_1"] == c).astype(float).where(df["Var_1"].notna())
    return X[feature_names()].astype(float)
