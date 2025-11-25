import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import os
import copy
import warnings
warnings.filterwarnings("ignore")

# ===== Device init =====
def init_process_group():
    rank = int(os.getenv('RANK', 0))
    local_rank = int(os.getenv('LOCAL_RANK', 0))
    num_gpus = int(os.getenv('WORLD_SIZE', 1))
    return rank == 0, torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'), num_gpus

is_rank0, DEVICE, num_gpus = init_process_group()

# ===== Load ATS + MT datasets =====
ats = pd.read_csv("CSV/ats-new_features.csv", header=0, delimiter=';')
ats.drop(columns=ats.columns[2:6], inplace=True)
ats = ats.groupby('sentence').mean().reset_index()
ats["source"] = "ats"

mt = pd.read_csv("CSV/mt-new_features.csv", header=0, delimiter=';')
mt.drop(columns=mt.columns[2:6], inplace=True)
mt = mt.groupby('sentence').mean().reset_index()
mt["source"] = "mt"

df = pd.concat([ats, mt], ignore_index=True)
df = df.drop(columns=['dugast_uber_index', 'herdans_vm', 'yules_i'], errors='ignore')
df = df.sample(frac=1, random_state=23).reset_index(drop=True)

# ===== Settings =====
k = 7
kf = KFold(n_splits=k, shuffle=True, random_state=23)
target_column = "overall"
model_name = "deepset/gelectra-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)

LEARNING_RATE = 2e-5
MAX_LENGTH = 512
BATCH_SIZE = 8
EPOCHS = 30
N_FEATURES = 20

# ===== Dataset wrapper =====
class RegressionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item
    def __len__(self):
        return len(self.labels)

# ===== Feature selection =====
def select_features(train_data, feature_cols, target_col, n_features=N_FEATURES):
    X = train_data[feature_cols].fillna(train_data[feature_cols].mean())
    y = train_data[target_col].values.astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)
    sfs = SequentialFeatureSelector(LinearRegression(),
                                    n_features_to_select=n_features,
                                    direction='forward',
                                    cv=5,
                                    n_jobs=-1).fit(X_scaled, y)
    selected = [feature_cols[i] for i in sfs.get_support(indices=True)]
    feature_scaler = StandardScaler().fit(train_data[selected].fillna(0).values.astype(float))
    return selected, feature_scaler

# ===== Embedding extraction =====
def extract_embeddings(model, tokenizer, texts, batch_size=8):
    model.eval()
    embeddings = []
    backbone = getattr(model, "electra", None) or getattr(model, "bert", None) or getattr(model, "roberta", None) or getattr(model, "base_model", None) or model
    backbone = backbone.to(DEVICE)
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = backbone(**inputs)
            hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
            cls_embedding = hidden[:, 0, :].cpu().numpy()
            embeddings.append(cls_embedding)
    return np.vstack(embeddings)

# ===== Metrics =====
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.reshape(-1)
    labels = labels.reshape(-1)

    mse = mean_squared_error(labels, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(labels, preds)
    r2 = r2_score(labels, preds)
    acc = (np.abs(preds - labels) < 0.5).mean()

    epsilon = 0.25
    errors = preds - labels
    squared_errors = np.where(np.abs(errors) ** 2 > epsilon, (np.abs(errors) ** 2 - epsilon), 0)
    eise = squared_errors.mean()

    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "seei": eise, "accuracy": acc}

# ===== Main training loop =====
def main():
    k = 7
    feature_cols = [c for c in df.columns if c not in ["sentence", target_column, "source"]]
    results = []

    splits = list(kf.split(df))
    for fold, (train_idx, test_idx) in enumerate(splits):
        print(f"\n=== Fold {fold+1}/{k} ===")

        train_data = df.iloc[train_idx]
        test_data = df.iloc[test_idx]
        val_data = df.iloc[splits[fold-1][1]] if fold > 0 else df.iloc[splits[-1][1]]
        train_data = train_data.drop(val_data.index)

        # Feature selection
        selected, scaler = select_features(train_data, feature_cols, target_column, n_features=N_FEATURES)
        print("Selected features:", selected)

        # Fine-tune Transformer (on merged ATS+MT)
        train_dataset = RegressionDataset(tokenizer(train_data["sentence"].tolist(),
                                                    truncation=True, padding="max_length", max_length=MAX_LENGTH),
                                          train_data[target_column].values.astype(float))
        val_dataset = RegressionDataset(tokenizer(val_data["sentence"].tolist(),
                                                  truncation=True, padding="max_length", max_length=MAX_LENGTH),
                                        val_data[target_column].values.astype(float))
        base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
        base_model.config.problem_type = "regression"
        training_args = TrainingArguments(
            output_dir=f"./Outputs/overall-mt-svm/fold_{fold}",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=LEARNING_RATE,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            num_train_epochs=EPOCHS,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="rmse",   # now valid
            greater_is_better=False,
            logging_dir=f"./logs/fold_{fold}"
        )
        trainer = Trainer(
            model=base_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,   # added
        )
        print("  Training Transformer...")
        trainer.train()
        model = trainer.model.to(DEVICE)

        # Extract embeddings
        train_val_df = pd.concat([train_data, val_data], ignore_index=True)
        train_val_texts = train_val_df["sentence"].tolist()
        test_texts = test_data["sentence"].tolist()
        print("  Extracting embeddings...")
        train_val_emb = extract_embeddings(model, tokenizer, train_val_texts, batch_size=BATCH_SIZE)
        test_emb = extract_embeddings(model, tokenizer, test_texts, batch_size=BATCH_SIZE)

        # Handcrafted features
        train_val_hand = scaler.transform(train_val_df[selected].fillna(0).values.astype(float))
        test_hand = scaler.transform(test_data[selected].fillna(0).values.astype(float))

        # Combine
        X_train_val = np.hstack([train_val_emb, train_val_hand])
        X_test = np.hstack([test_emb, test_hand])
        y_train_val = train_val_df[target_column].values.astype(float).ravel()
        y_test = test_data[target_column].values.astype(float).ravel()

        # Scale combined features
        comb_scaler = StandardScaler().fit(X_train_val)
        X_train_val = comb_scaler.transform(X_train_val)
        X_test = comb_scaler.transform(X_test)

        # Train SVR
        svm = SVR(kernel='rbf', C=1.0, epsilon=0.1).fit(X_train_val, y_train_val)

        # Evaluate only on MT subset
        mt_mask = test_data["source"] == "mt"
        X_test_mt = X_test[mt_mask]
        y_test_mt = y_test[mt_mask]
        if len(y_test_mt) == 0:
            print("  No MT samples in this fold's test set.")
            continue

        preds = svm.predict(X_test_mt)
        mse = mean_squared_error(y_test_mt, preds)
        mae = mean_absolute_error(y_test_mt, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_mt, preds)
        epsilon = 0.25
        eise = np.mean(np.where((preds - y_test_mt) ** 2 > epsilon, (preds - y_test_mt) ** 2 - epsilon, 0))
        acc = (np.abs(preds - y_test_mt) < 0.5).mean()

        fold_result = {"fold": fold+1, "mse": mse, "mae": mae, "rmse": rmse,
                       "r2": r2, "seei": eise, "accuracy": acc, "n_mt": len(y_test_mt)}
        results.append(fold_result)
        print(f"  MT Test Metrics: MSE={mse:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}, "
              f"R2={r2:.4f}, SEEI={eise:.4f}, ACC={acc:.4f}, n={len(y_test_mt)}")

        torch.cuda.empty_cache()

    # Averages
    avg = {m: np.mean([r[m] for r in results]) for m in ["mse", "mae", "rmse", "r2", "seei", "accuracy"]}
    total_n = np.sum([r["n_mt"] for r in results])

    print("\n=== Final MT-only averages across folds ===")
    for k, v in avg.items():
        print(f"Avg {k.upper()}: {v:.4f}")
    print(f"Total MT test samples: {total_n}")

if __name__ == "__main__":
    main()
