import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
from torch import distributed as dist
from sklearn.model_selection import KFold
import numpy as np
import copy
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from torch import nn
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Device initialization
def init_process_group():
    rank = int(os.getenv('RANK', 0))
    local_rank = int(os.getenv('LOCAL_RANK', 0))
    num_gpus = int(os.getenv('WORLD_SIZE', 1))
    return rank == 0, torch.device(f'cuda:{local_rank}'), num_gpus

is_rank0, device, num_gpus = init_process_group()

dataset_name = "big_data_averaged_correct_format"

# Load features
features_df = pd.read_csv("CSV/ats_features.csv", header=0, delimiter=';') 
features_df = features_df.groupby('sentence').mean().reset_index()

data = pd.read_csv("CSV/ats-new_features.csv", header=0, delimiter=';')
new_df = pd.DataFrame(data)
new_df.drop(columns=new_df.columns[1:2], inplace=True)
#Groupby sentence and average the labels 
new_df = new_df.groupby('sentence').mean().reset_index()
 
features_df = pd.concat([features_df, new_df], ignore_index=True)
# Remove problematic features
features_df = features_df.drop(columns=['dugast_uber_index', 'herdans_vm', 'yules_i'], errors='ignore')

# Shuffle the combined dataset
shuffled_df = features_df.sample(frac=1, random_state=23).reset_index(drop=True)

# Initialize k-fold cross-validation
k = 7
kf = KFold(n_splits=k, shuffle=True, random_state=23)

model_name = "deepset/gbert-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)
base_model.config.problem_type = "regression"

# Simplified BERT model for regression (without features)
class BERTRegressionModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.bert = base_model.bert
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(base_model.config.hidden_size, 4)
        
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
    
        loss = None
        if labels is not None:
            loss = torch.nn.functional.mse_loss(logits, labels.view(-1, 4))
        
        return (loss, logits) if loss is not None else logits

    def get_cls_embedding(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        return pooled_output

# Compute metrics for BERT training
def compute_metrics_for_regression(eval_pred):
    logits, labels = eval_pred
    logits = logits.reshape(-1, 4)
    labels = labels.reshape(-1, 4)

    mse = mean_squared_error(labels, logits, multioutput='raw_values')
    mae = mean_absolute_error(labels, logits, multioutput='raw_values')
    rmse = np.sqrt(mse)
    r2 = r2_score(labels, logits, multioutput='raw_values')
    
    return {
        "mse": mse.mean(),
        "mae": mae.mean(),
        "rmse": rmse.mean(),
        "r2": r2.mean(),
    }

# Training arguments
LEARNING_RATE = 2e-5
MAX_LENGTH = 512
BATCH_SIZE = 8
EPOCHS = 30

def create_training_args(fold):
    model_name_clean = model_name.replace("/", "_")
    return TrainingArguments(
        output_dir=f"Outputs/ats-new-svm/fold_{fold}",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS, 
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        metric_for_best_model="rmse",
        greater_is_better=False,
        load_best_model_at_end=True,
        weight_decay=0.01,
    )

# Get all feature columns (excluding text and labels)
feature_columns = [col for col in shuffled_df.columns 
                   if col not in ['sentence', 'sprachliche logik', 'komplexität', 
                                 'eindeutigkeit', 'vorhersehbarkeit']]

# Precompute the splits for all folds
splits = list(kf.split(shuffled_df))

def select_features(train_data, feature_columns, target_columns, n_features=20):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_data[feature_columns].values.astype(float))
    y_train = train_data[target_columns].values
    
    sfs = SequentialFeatureSelector(LinearRegression(),
                                  n_features_to_select=n_features,
                                  direction='forward',
                                  cv=5)
    sfs.fit(X_train_scaled, y_train)
    
    selected_indices = [i for i, selected in enumerate(sfs.get_support()) if selected]
    selected_features = [feature_columns[i] for i in selected_indices]
    
    selected_scaler = StandardScaler()
    selected_scaler.fit(train_data[selected_features].values.astype(float))
    
    return selected_features, selected_scaler

def preprocess_function(examples, tokenizer):
    labels = [
        float(examples["sprachliche logik"]),
        float(examples["komplexität"]),
        float(examples["eindeutigkeit"]),
        float(examples["vorhersehbarkeit"])
    ]

    tokenized_inputs = tokenizer(examples["sentence"], truncation=True,
                               padding="max_length", max_length=512)
    tokenized_inputs["labels"] = torch.tensor(labels, dtype=torch.float)
    
    return tokenized_inputs

def make_dataset(df, tokenizer):
    processed = df.apply(
        preprocess_function,
        tokenizer=tokenizer,
        axis=1
    )
    return processed.tolist()

# Store performance metrics
bert_scores = []
svm_scores = []

initial_model_weights = copy.deepcopy(base_model.state_dict())

for fold, (train_idx, test_idx) in enumerate(splits):
    print(f"Fold {fold + 1}/{k}")
    
    train_data = shuffled_df.iloc[train_idx]
    test_data = shuffled_df.iloc[test_idx]
    
    if fold == 0:
        val_data = shuffled_df.iloc[splits[-1][1]]
    else:
        val_data = shuffled_df.iloc[splits[fold - 1][1]]
    
    train_data = train_data.drop(val_data.index)
    
    # Select features for this fold
    selected_features, selected_scaler = select_features(
        train_data, 
        feature_columns,
        ['sprachliche logik', 'komplexität', 'eindeutigkeit', 'vorhersehbarkeit']
    )
    
    # 1. Fine-tune BERT model
    print("  Fine-tuning BERT model...")
    model = BERTRegressionModel(base_model)
    model.load_state_dict(initial_model_weights)
    model.to(device)
    
    trainer = Trainer(
        model=model,
        args=create_training_args(fold),
        train_dataset=make_dataset(train_data, tokenizer),
        eval_dataset=make_dataset(val_data, tokenizer),
        compute_metrics=compute_metrics_for_regression
    )
    
    trainer.train()
    bert_test_results = trainer.evaluate(eval_dataset=make_dataset(test_data, tokenizer))
    bert_scores.append(bert_test_results)
    
    # 2. Extract [CLS] embeddings and combine with features for SVM
    print("  Training SVM with BERT embeddings and features...")
    
    # Function to get [CLS] embeddings
    def get_cls_embeddings(model, df):
        model.eval()
        embeddings = []
        with torch.no_grad():
            for _, row in df.iterrows():
                inputs = tokenizer(row['sentence'], return_tensors='pt', 
                                 truncation=True, padding='max_length', max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                # Use the new method to get embeddings
                embedding = model.get_cls_embedding(**inputs)
                embeddings.append(embedding.cpu().numpy())
        return np.vstack(embeddings)    
    # Get embeddings for all datasets
    train_embeddings = get_cls_embeddings(model, train_data)
    val_embeddings = get_cls_embeddings(model, val_data)
    test_embeddings = get_cls_embeddings(model, test_data)
    
    # Get selected features for all datasets
    def get_selected_features(df):
        features = selected_scaler.transform(df[selected_features].values.astype(float))
        return features
    
    train_features = get_selected_features(train_data)
    val_features = get_selected_features(val_data)
    test_features = get_selected_features(test_data)
    
    # Combine embeddings and features
    X_train = np.hstack([train_embeddings, train_features])
    X_val = np.hstack([val_embeddings, val_features])
    X_test = np.hstack([test_embeddings, test_features])
    
    # Get labels
    y_train = train_data[['sprachliche logik', 'komplexität', 'eindeutigkeit', 'vorhersehbarkeit']].values
    y_val = val_data[['sprachliche logik', 'komplexität', 'eindeutigkeit', 'vorhersehbarkeit']].values
    y_test = test_data[['sprachliche logik', 'komplexität', 'eindeutigkeit', 'vorhersehbarkeit']].values
    
    # Scale the combined features (important for SVM)
    combined_scaler = StandardScaler()
    X_train_scaled = combined_scaler.fit_transform(X_train)
    X_val_scaled = combined_scaler.transform(X_val)
    X_test_scaled = combined_scaler.transform(X_test)
    
    # Train SVM with hyperparameter tuning
    param_grid = {
        'multioutputregressor__estimator__C': [1],
         'multioutputregressor__estimator__epsilon': [0.1],
        'multioutputregressor__estimator__kernel': ['rbf']
    }

    # Create the pipeline with proper naming
    svm_pipeline = make_pipeline(
        StandardScaler(),
        MultiOutputRegressor(SVR())
    )

    grid_search = GridSearchCV(
        svm_pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train_scaled, y_train)
    best_svm = grid_search.best_estimator_  
  
    # Evaluate on validation set to select best model
    val_pred = best_svm.predict(X_val_scaled)
    
    # Final evaluation on test set
    test_pred = best_svm.predict(X_test_scaled)
    
    # Calculate metrics
    svm_mse = mean_squared_error(y_test, test_pred, multioutput='raw_values').mean()
    svm_mae = mean_absolute_error(y_test, test_pred, multioutput='raw_values').mean()
    svm_rmse = np.sqrt(mean_squared_error(y_test, test_pred, multioutput='raw_values')).mean()
    svm_r2 = r2_score(y_test, test_pred, multioutput='raw_values').mean()
    
    svm_scores.append({
        'mse': svm_mse,
        'mae': svm_mae,
        'rmse': svm_rmse,
        'r2': svm_r2,
    })
    
    torch.cuda.empty_cache()

# Calculate average performance across all folds
avg_bert_metrics = {
    "model": f"{model_name} (BERT only)",
    "mse": np.mean([score["eval_mse"] for score in bert_scores]),
    "mae": np.mean([score["eval_mae"] for score in bert_scores]),
    "rmse": np.mean([score["eval_rmse"] for score in bert_scores]),
    "r2": np.mean([score["eval_r2"] for score in bert_scores]),
}

avg_svm_metrics = {
    "model": f"{model_name} + SVM with features",
    "mse": np.mean([score["mse"] for score in svm_scores]),
    "mae": np.mean([score["mae"] for score in svm_scores]),
    "rmse": np.mean([score["rmse"] for score in svm_scores]),
    "r2": np.mean([score["r2"] for score in svm_scores]),
}

# Print results
print("\nAverage performance across all folds:")
print("BERT-only model:")
print(avg_bert_metrics)
print("\nBERT + SVM with features:")
print(avg_svm_metrics)