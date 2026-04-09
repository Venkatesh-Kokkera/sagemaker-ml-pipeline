import os
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

def load_data(input_dir: str) -> pd.DataFrame:
    """Load raw data from input directory."""
    dfs = []
    for file in os.listdir(input_dir):
        if file.endswith(".csv"):
            path = os.path.join(input_dir, file)
            df = pd.read_csv(path)
            dfs.append(df)
            print(f"Loaded: {file} — {df.shape}")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"Total data shape: {combined.shape}")
    return combined

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess dataframe."""
    # Drop duplicates
    before = len(df)
    df = df.drop_duplicates()
    print(f"Removed {before - len(df)} duplicates")

    # Drop columns with too many nulls
    null_threshold = 0.5
    null_pct = df.isnull().mean()
    cols_to_drop = null_pct[null_pct > null_threshold].index
    df = df.drop(columns=cols_to_drop)
    print(f"Dropped {len(cols_to_drop)} high-null columns")

    return df

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical columns."""
    cat_cols = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    if "target" in cat_cols:
        cat_cols.remove("target")

    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
        print(f"Encoded: {col}")

    return df

def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values."""
    num_cols = df.select_dtypes(
        include=["float64", "int64"]
    ).columns.tolist()

    if "target" in num_cols:
        num_cols.remove("target")

    imputer = SimpleImputer(strategy="median")
    df[num_cols] = imputer.fit_transform(df[num_cols])
    print(f"Imputed {len(num_cols)} numeric columns")

    return df

def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
) -> tuple:
    """Scale numeric features."""
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns
    )
    print("✅ Features scaled")
    return X_train_scaled, X_test_scaled

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new features from existing ones."""
    num_cols = df.select_dtypes(
        include=["float64", "int64"]
    ).columns.tolist()

    if "target" in num_cols:
        num_cols.remove("target")

    # Interaction features
    if len(num_cols) >= 2:
        df["feature_interaction"] = (
            df[num_cols[0]] * df[num_cols[1]]
        )

    # Log transform for skewed features
    for col in num_cols[:3]:
        if df[col].min() > 0:
            df[f"{col}_log"] = np.log1p(df[col])

    print(f"New shape after feature engineering: {df.shape}")
    return df

def process(args):
    """Main feature engineering pipeline."""
    print("Loading data...")
    df = load_data(args.input_dir)

    print("Cleaning data...")
    df = clean_data(df)

    print("Encoding categoricals...")
    df = encode_categoricals(df)

    print("Imputing missing values...")
    df = impute_missing(df)

    print("Engineering features...")
    df = engineer_features(df)

    # Split features and target
    X = df.drop("target", axis=1)
    y = df["target"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Scale features
    X_train, X_test = scale_features(X_train, X_test)

    # Save outputs
    os.makedirs(args.train_output, exist_ok=True)
    os.makedirs(args.test_output, exist_ok=True)

    train_df = X_train.copy()
    train_df["target"] = y_train.values
    train_df.to_csv(
        os.path.join(args.train_output, "train.csv"),
        index=False
    )

    test_df = X_test.copy()
    test_df["target"] = y_test.values
    test_df.to_csv(
        os.path.join(args.test_output, "test.csv"),
        index=False
    )

    print(f"✅ Train set: {train_df.shape}")
    print(f"✅ Test set: {test_df.shape}")
    print("✅ Feature engineering complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default="/opt/ml/processing/input"
    )
    parser.add_argument(
        "--train-output",
        default="/opt/ml/processing/train"
    )
    parser.add_argument(
        "--test-output",
        default="/opt/ml/processing/test"
    )
    args = parser.parse_args()
    process(args)
