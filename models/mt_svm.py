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

def init_process_group():
    rank = int(os.getenv('RANK', 0))
    local_rank = int(os.getenv('LOCAL_RANK', 0))
    num_gpus = int(os.getenv('WORLD_SIZE', 1))
    return rank == 0, torch.device(f'cuda:{local_rank}'), num_gpus

is_rank0, DEVICE, num_gpus = init_process_group()

# Configuration
MODEL_NAME = "deepset/gelectra-large"
N_FOLDS = 7
SEED = 23
TARGET_COLUMNS = ["präzision", "komplexität", "transparenz", "grammatikalität"]

# Data Loading
def load_data():
    features_df = pd.read_csv("CSV/mt_features.csv", header=0, delimiter=';') 
    features_df = features_df.groupby('sentence').mean().reset_index()
    features_df = features_df.drop(columns=['dugast_uber_index', 'herdans_vm', 'yules_i'], errors='ignore')
    shuffled_df = features_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    feature_cols = [col for col in features_df.columns if col not in ['sentence'] + TARGET_COLUMNS]
    return shuffled_df, feature_cols

# Custom Dataset Class
class RegressionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)

# Feature Processing
def select_features(train_data, feature_cols, target_col, n_features=20):
    X = train_data[feature_cols].fillna(train_data[feature_cols].mean())
    y = train_data[target_col]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    selector = SequentialFeatureSelector(
        LinearRegression(),
        n_features_to_select=n_features,
        direction='forward',
        cv=5,
        n_jobs=-1
    ).fit(X_scaled, y)
    
    selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
    feature_scaler = StandardScaler().fit(train_data[selected_features])
    
    return selected_features, feature_scaler

# Compute Metrics
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

# Embedding Extraction
def extract_embeddings(model, tokenizer, texts, batch_size=8):
    model.eval()
    embeddings = []
    
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
            outputs = model.electra(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_embedding)
    
    return np.vstack(embeddings)

# Main Training Loop
def main():
    # Load and prepare data
    df, feature_cols = load_data()
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    svm_results = {target: [] for target in TARGET_COLUMNS}

    splits = list(kf.split(df))
    for fold, (train_idx, test_idx) in enumerate(splits):
        print(f"\n=== Fold {fold + 1}/{N_FOLDS} ===")
        
        # Split data
        train_data = df.iloc[train_idx]
        test_data = df.iloc[test_idx]
        
        if fold == 0:
            val_data = df.iloc[splits[-1][1]]
        else:
            val_data = df.iloc[splits[fold - 1][1]]
        
        train_val_data = pd.concat([train_data, val_data])
        train_data = train_data.drop(val_data.index)
        
        # Process each target individually
        for target_column in TARGET_COLUMNS:
            print(f"\nTraining for target: {target_column}")
            
            # 1. Feature selection on TRAIN only
            selected_features, feature_scaler = select_features(
                train_data, feature_cols, target_column
            )
            
            # 2. Prepare ELECTRA datasets
            def prepare_dataset(df):
                texts = df['sentence'].tolist()
                labels = df[target_column].values.astype(float)
                encodings = tokenizer(texts, truncation=True, padding='max_length', max_length=512)
                return RegressionDataset(encodings, labels)
            
            train_dataset = prepare_dataset(train_data)
            val_dataset = prepare_dataset(val_data)
            
            # 3. Fine-tune ELECTRA
            model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME,
                num_labels=1,
                problem_type="regression"
            ).to(DEVICE)
            
            training_args = TrainingArguments(
                output_dir=f"./results/fold_{fold}/{target_column}",
                evaluation_strategy="epoch",
                save_strategy="epoch",
                learning_rate=2e-5,
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                num_train_epochs=30,
                weight_decay=0.01,
                load_best_model_at_end=True,
                metric_for_best_model="rmse",
                greater_is_better=False
            )
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=compute_metrics_for_regression
            )
            
            trainer.train()
            
            # 4. Extract [CLS] embeddings
            train_val_texts = train_val_data['sentence'].tolist()
            test_texts = test_data['sentence'].tolist()
            
            train_val_bert = extract_embeddings(model, tokenizer, train_val_texts)
            test_bert = extract_embeddings(model, tokenizer, test_texts)
            
            # 5. Prepare linguistic features
            train_val_ling = feature_scaler.transform(
                train_val_data[selected_features].fillna(0)
            )
            test_ling = feature_scaler.transform(
                test_data[selected_features].fillna(0)
            )
            
            # 6. Combine features
            X_train_val = np.hstack([train_val_bert, train_val_ling])
            X_test = np.hstack([test_bert, test_ling])
            y_train_val = train_val_data[target_column].values.reshape(-1, 1)
            y_test = test_data[target_column].values.reshape(-1, 1)
            
            # 7. Scale combined features
            combined_scaler = StandardScaler().fit(X_train_val)
            X_train_val_scaled = combined_scaler.transform(X_train_val)
            X_test_scaled = combined_scaler.transform(X_test)
            
            # 8. Train SVM on train+val
            svm = SVR(kernel='rbf', C=1.0, epsilon=0.1).fit(X_train_val_scaled, y_train_val.ravel())
            
            # 9. Evaluate
            test_pred = svm.predict(X_test_scaled)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, test_pred)
            mae = mean_absolute_error(y_test, test_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, test_pred)
            
            epsilon = 0.25
            squared_error = (test_pred.reshape(-1, 1) - y_test) ** 2
            eise = np.mean(np.where(squared_error > epsilon, squared_error - epsilon, 0))
            
            metrics = {
                'fold': fold + 1,
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'epsilon_insensitive_squared_error': eise
            }
            svm_results[target_column].append(metrics)
            
            print(f"{target_column} - Fold {fold + 1} Test Metrics:")
            print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
            print(f"R²: {r2:.4f}, EISE: {eise:.4f}")
    
    # Final evaluation
    for target_column in TARGET_COLUMNS:
        avg_metrics = {
            'target': target_column,
            'avg_mse': np.mean([r['mse'] for r in svm_results[target_column]]),
            'avg_mae': np.mean([r['mae'] for r in svm_results[target_column]]),
            'avg_rmse': np.mean([r['rmse'] for r in svm_results[target_column]]),
            'avg_r2': np.mean([r['r2'] for r in svm_results[target_column]]),
            'avg_eise': np.mean([r['epsilon_insensitive_squared_error'] for r in svm_results[target_column]])
        }
        
        print(f"\n=== Final Results for {target_column} ===")
        print(f"Average MSE: {avg_metrics['avg_mse']:.4f}")
        print(f"Average MAE: {avg_metrics['avg_mae']:.4f}")
        print(f"Average RMSE: {avg_metrics['avg_rmse']:.4f}")
        print(f"Average R²: {avg_metrics['avg_r2']:.4f}")
        print(f"Average EISE: {avg_metrics['avg_eise']:.4f}")

if __name__ == "__main__":
    main()