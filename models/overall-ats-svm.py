import pandas as pd
import numpy as np
import torch
from torch import nn
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    Trainer, 
    TrainingArguments
)
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
import os
import copy
import warnings
warnings.filterwarnings("ignore")

# -------------------------
# Device / distributed init
# -------------------------
def init_process_group():
    rank = int(os.getenv('RANK', 0))
    local_rank = int(os.getenv('LOCAL_RANK', 0))
    num_gpus = int(os.getenv('WORLD_SIZE', 1))
    return rank == 0, torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'), num_gpus

is_rank0, DEVICE, num_gpus = init_process_group()

# -------------------------
# Configuration
# -------------------------
MODEL_NAME = "deepset/gelectra-large"   # keep same as your SVR script; change if desired
N_FOLDS = 7
SEED = 23
TARGET_COLUMN = "overall"               # predict this column for ATS
N_SELECTED_FEATURES = 20                # SFS picks this many features
BATCH_SIZE = 8
NUM_EPOCHS = 30                         # same as your original
LR = 2e-5

np.random.seed(SEED)
torch.manual_seed(SEED)

# -------------------------
# Data Loading (ATS)
# -------------------------
def load_ats():
    df = pd.read_csv("CSV/ats-new_features.csv", header=0, delimiter=';')
    # drop columns 3..6 like in your script
    df.drop(columns=df.columns[2:6], inplace=True)
    df = df.groupby('sentence').mean().reset_index()
    # remove problematic features if present
    df = df.drop(columns=['dugast_uber_index', 'herdans_vm', 'yules_i'], errors='ignore')
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    return df

# -------------------------
# Dataset wrapper for Trainer (keeps your original behavior)
# -------------------------
class RegressionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        # encodings: dict of lists (tokenizer output with strings/lists)
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # Mirror your original: convert each encoding entry to tensor per item
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)

# -------------------------
# Feature selection (train only)
# -------------------------
def select_features(train_data, feature_cols, target_col, n_features=N_SELECTED_FEATURES):
    X = train_data[feature_cols].fillna(train_data[feature_cols].mean())
    y = train_data[target_col].values.astype(float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)

    selector = SequentialFeatureSelector(
        LinearRegression(),
        n_features_to_select=n_features,
        direction='forward',
        cv=5,
        n_jobs=-1
    ).fit(X_scaled, y)

    selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
    feature_scaler = StandardScaler().fit(train_data[selected_features].fillna(0).values.astype(float))

    return selected_features, feature_scaler

# -------------------------
# Compute metrics (same as your SVR script)
# -------------------------
def compute_metrics_for_regression(eval_pred):
    logits, labels = eval_pred
    logits = logits.reshape(-1, 1)
    labels = labels.reshape(-1, 1)

    mse = mean_squared_error(labels, logits)
    mae = mean_absolute_error(labels, logits)
    rmse = np.sqrt(mse)
    r2 = r2_score(labels, logits)
    accuracy = (abs(logits - labels) < 0.5).mean()

    epsilon = 0.25
    errors = logits - labels
    squared_errors = np.where(np.abs(errors)**2 > epsilon,
                              (np.abs(errors)**2 - epsilon),
                              0)
    eise = squared_errors.mean()

    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "seei": eise,
        "accuracy": accuracy
    }

# -------------------------
# Embedding extraction (CLS)
# -------------------------
def extract_embeddings(model, tokenizer, texts, batch_size=8):
    model.eval()
    embeddings = []
    # use backbone attribute like model.electra (as in your original)
    backbone = getattr(model, "electra", None) or getattr(model, "bert", None) or getattr(model, "roberta", None) or getattr(model, "base_model", None) or model
    backbone = backbone.to(DEVICE)

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            outputs = backbone(**inputs)
            # hidden state may be outputs.last_hidden_state or outputs[0]
            hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
            cls_embedding = hidden[:, 0, :].cpu().numpy()
            embeddings.append(cls_embedding)

    if len(embeddings) == 0:
        return np.zeros((0, backbone.config.hidden_size))
    return np.vstack(embeddings)

# -------------------------
# Main training loop (SVR pipeline)
# -------------------------
def main():
    df = load_ats()
    # identify feature columns (exclude sentence and target)
    feature_cols = [c for c in df.columns if c not in ['sentence', TARGET_COLUMN]]

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    # store per-fold metrics
    fold_metrics = []

    splits = list(kf.split(df))
    for fold, (train_idx, test_idx) in enumerate(splits):
        print(f"\n=== Fold {fold + 1}/{N_FOLDS} ===")

        train_data = df.iloc[train_idx].reset_index(drop=True)
        test_data = df.iloc[test_idx].reset_index(drop=True)

        # validation: previous fold's test (or last fold for fold 0)
        val_idx = splits[-1][1] if fold == 0 else splits[fold - 1][1]
        val_data = df.iloc[val_idx].reset_index(drop=True)

        # remove validation rows from train_data (like in your original)
        # (here indices are from original df, so drop by matching sentences to be safe)
        val_sentences = set(val_data['sentence'].tolist())
        train_data = train_data[~train_data['sentence'].isin(val_sentences)].reset_index(drop=True)

        # 1) Feature selection on TRAIN only
        selected_features, feature_scaler = select_features(train_data, feature_cols, TARGET_COLUMN, n_features=N_SELECTED_FEATURES)
        print("Selected features:", selected_features)

        # 2) Prepare ELECTRA datasets (tokenizer used as in your script)
        def prepare_dataset_for_trainer(df_part):
            texts = df_part['sentence'].tolist()
            labels = df_part[TARGET_COLUMN].values.astype(float)
            encodings = tokenizer(texts, truncation=True, padding='max_length', max_length=512)
            return RegressionDataset(encodings, labels)

        train_dataset = prepare_dataset_for_trainer(train_data)
        val_dataset = prepare_dataset_for_trainer(val_data)

        # 3) Fine-tune the Transformer for this fold
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=1,
            problem_type="regression"
        ).to(DEVICE)

        training_args = TrainingArguments(
            output_dir=f"./results/fold_{fold}",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=LR,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            num_train_epochs=NUM_EPOCHS,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="rmse",
            greater_is_better=False,
            logging_dir=f"./logs/fold_{fold}"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics_for_regression
        )

        print("  Training Transformer (may take time)...")
        trainer.train()
        # ensure best model loaded
        model = trainer.model
        model.to(DEVICE)

        # 4) Extract [CLS] embeddings
        train_val_df = pd.concat([train_data, val_data], ignore_index=True)
        train_val_texts = train_val_df['sentence'].tolist()
        test_texts = test_data['sentence'].tolist()

        print("  Extracting embeddings...")
        train_val_bert = extract_embeddings(model, tokenizer, train_val_texts, batch_size=BATCH_SIZE)
        test_bert = extract_embeddings(model, tokenizer, test_texts, batch_size=BATCH_SIZE)

        # 5) Prepare linguistic features (scale using feature_scaler fit on train)
        train_val_hand = feature_scaler.transform(train_val_df[selected_features].fillna(0).values.astype(float))
        test_hand = feature_scaler.transform(test_data[selected_features].fillna(0).values.astype(float))

        # 6) Combine embeddings + handcrafted features
        X_train_val = np.hstack([train_val_bert, train_val_hand])
        X_test = np.hstack([test_bert, test_hand])
        y_train_val = train_val_df[TARGET_COLUMN].values.astype(float).reshape(-1, 1).ravel()
        y_test = test_data[TARGET_COLUMN].values.astype(float).reshape(-1, 1).ravel()

        # 7) Scale combined features and train SVR
        combined_scaler = StandardScaler().fit(X_train_val)
        X_train_val_scaled = combined_scaler.transform(X_train_val)
        X_test_scaled = combined_scaler.transform(X_test)

        svm = SVR(kernel='rbf', C=1.0, epsilon=0.1).fit(X_train_val_scaled, y_train_val)

        # 8) Evaluate
        test_pred = svm.predict(X_test_scaled)

        mse = mean_squared_error(y_test, test_pred)
        mae = mean_absolute_error(y_test, test_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, test_pred)
        epsilon = 0.25
        squared_error = (test_pred.reshape(-1, 1) - y_test.reshape(-1, 1)) ** 2
        eise = np.mean(np.where(squared_error > epsilon, squared_error - epsilon, 0))

        fold_result = {
            'fold': fold + 1,
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'eise': eise,
            'n_test': len(y_test)
        }
        fold_metrics.append(fold_result)

        print(f"  Fold {fold+1} Test Metrics: MSE={mse:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}, EISE={eise:.4f}, n={len(y_test)}")

        # free memory
        torch.cuda.empty_cache()

    # Final averages across folds
    avg_mse = np.mean([m['mse'] for m in fold_metrics])
    avg_mae = np.mean([m['mae'] for m in fold_metrics])
    avg_rmse = np.mean([m['rmse'] for m in fold_metrics])
    avg_r2 = np.mean([m['r2'] for m in fold_metrics])
    avg_eise = np.mean([m['eise'] for m in fold_metrics])
    total_n = np.sum([m['n_test'] for m in fold_metrics])

    print("\n=== Final average across folds ===")
    print(f"Avg MSE: {avg_mse:.4f}")
    print(f"Avg MAE: {avg_mae:.4f}")
    print(f"Avg RMSE: {avg_rmse:.4f}")
    print(f"Avg R2: {avg_r2:.4f}")
    print(f"Avg EISE: {avg_eise:.4f}")
    print(f"Total test samples across folds: {total_n}")

if __name__ == "__main__":
    main()
